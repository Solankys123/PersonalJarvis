"""Agent Chat, the real managed worker, and SQLite FORGE lifecycle."""

from __future__ import annotations

import asyncio
import json
import os
import re
import threading
from html import escape
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from types import SimpleNamespace
from urllib.parse import parse_qs, urlsplit

import pytest

from jarvis.agent_chat.service import AgentChatService
from jarvis.agent_chat.store import AgentChatStore
from jarvis.brain.manager import BrainManager
from jarvis.core.bus import EventBus
from jarvis.core.config import JarvisConfig
from jarvis.core.protocols import BrainDelta
from jarvis.forge.engine import Limits
from jarvis.forge.store import Store
from jarvis.forge.tool import ForgeMindTool
from jarvis.safety.approval import ApprovalWorkflow
from jarvis.safety.risk_tier import RiskTierEvaluator
from jarvis.safety.tool_executor import ToolExecutor
from jarvis.society.approvals import Approvals
from jarvis.society.browser import install
from jarvis.society.browser.live import LiveSessions
from jarvis.society.events import ApprovalState
from jarvis.society.store import SocietyStore

pytestmark = pytest.mark.skipif(
    not (os.getenv("JARVIS_BROWSER_TEST_PYTHON") and os.getenv("JARVIS_BROWSER_TEST_EXECUTABLE")),
    reason="requires a browser worker Python and Chromium",
)


@pytest.fixture
def search_site():
    fixture = SimpleNamespace(requests=[], list_tag="ul", dynamic=False, moved=False)

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            parts = urlsplit(self.path)
            query = parse_qs(parts.query).get("q", [""])[0]
            fixture.requests.append((parts.path, query, self.headers.get("Cookie") or None))
            if parts.path == "/redirect":
                self.send_response(302)
                other_origin = fixture.origin.replace("127.0.0.1", "localhost")
                self.send_header("Location", f"{other_origin}/other")
                self.end_headers()
                return
            self.send_response(200)
            self.send_header("Content-Type", "text/html")
            if fixture.dynamic:
                self.send_header("Set-Cookie", "fixture_session=private cookie")
            self.end_headers()
            if query:
                tag = fixture.list_tag
                body = (
                    f"<h1>Results for {escape(query)}</h1>"
                    f"<{tag}><li>private fixture text</li></{tag}>"
                )
            elif fixture.dynamic and parts.path == "/catalog":
                extra = "<button>Other control</button>" if fixture.moved else ""
                body = (
                    extra
                    + """<h1>Catalog</h1>
                <button aria-label="Open catalog search">Open catalog search</button>
                <section></section><script>
                document.querySelector('button:last-of-type').onclick = () => {
                    const area = document.querySelector('section');
                    area.innerHTML = '<input type=search aria-label="Catalog query">'
                        + '<button aria-label="Find items">Find items</button>';
                    area.querySelector('button').onclick = async () => {
                        const query = area.querySelector('input').value;
                        await fetch('/results?q=' + encodeURIComponent(query));
                        const result = document.createElement('h2');
                        result.textContent = 'Catalog matches for ' + query;
                        area.replaceChildren(result,
                            document.createTextNode('private fixture text'));
                    };
                };
                </script>"""
                )
            else:
                credential = (
                    '<input type="password" name="password" value="fixture credential">'
                    if parts.path == "/sensitive"
                    else ""
                )
                body = (
                    '<h1>Catalog</h1><form method="get">'
                    '<input type="search" name="q" autocomplete="off">'
                    f'{credential}<button type="submit">Search</button></form>'
                )
            self.wfile.write(f"<!doctype html><main>{body}</main>".encode())

        def log_message(self, *args):
            pass  # Keep transient search input out of the test server log.

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    fixture.origin = f"http://127.0.0.1:{server.server_port}"
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield fixture
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


@pytest.mark.parametrize("dynamic", [False, True], ids=["legacy", "dynamic"])
async def test_web_search_through_agent_chat(tmp_path, monkeypatch, search_site, dynamic):
    search_site.dynamic = dynamic
    for name, env in (
        ("venv_python", "JARVIS_BROWSER_TEST_PYTHON"),
        ("browser_executable", "JARVIS_BROWSER_TEST_EXECUTABLE"),
    ):
        monkeypatch.setattr(install, name, lambda _, env=env: Path(os.environ[env]))
    monkeypatch.setattr(install, "is_installed", lambda _: True)
    config = JarvisConfig()
    config.forge.enabled = True
    bus = EventBus()
    executor = ToolExecutor(bus, RiskTierEvaluator(config.safety), ApprovalWorkflow(bus))
    db = Store(tmp_path / "forge.sqlite")
    society = SocietyStore(tmp_path / "society.sqlite")
    await society.open()
    approvals = Approvals(society)
    caller = SimpleNamespace(
        agent_id="lead",
        model="",
        provider="openai-codex",
        state="active",
        denies=[],
        grant_mode="default",
        approval_mode="always_ask",
        approval_rules={},
        browser_allowed_domains=["http*://127.0.0.1"],
    )

    async def roster_get(_):
        return caller

    async def unexpected(*args, **kwargs):
        pytest.fail("FORGE web search reached a provider or the desktop fixture")

    runtime = SimpleNamespace(
        data_dir=tmp_path,
        lead_id=caller.agent_id,
        store=society,
        approvals=approvals,
        roster=SimpleNamespace(get=roster_get),
        browser=SimpleNamespace(_python=None),
    )
    # Patch both bindings after the surface imports the real runtime accessor.
    monkeypatch.setattr("jarvis.society.surface.current_runtime", lambda: runtime)
    monkeypatch.setattr("jarvis.society.runtime.current_runtime", lambda: runtime)

    observed_controls = []
    model_calls = []

    class FixtureModel:
        calls = 0

        async def complete(self, request):
            self.calls += 1
            model_calls.append(self.calls)
            observation = "\n".join(
                str(message.content) for message in request.messages if message.role == "user"
            )
            lines = observation.splitlines()
            observed_controls.append(
                (self.calls, [line for line in lines if "<input" in line or "<button" in line])
            )

            def indexed(label, tag):
                for line in reversed(lines):
                    match = re.search(rf"\[(\d+)\]:?\s*<{tag}\b", line)
                    if match and label in line:
                        return int(match[1])
                return None

            query = re.search(
                r"public search interface to display results for '([^']+)'", observation
            )
            assert query
            value = query[1]
            if self.calls == 3:
                action = {"click": {"index": indexed("Find items", "button")}}
            elif self.calls == 2:
                action = {"input": {"index": indexed("Catalog query", "input"), "text": value}}
            else:
                action = {"click": {"index": indexed("Open catalog search", "button")}}
            yield BrainDelta(
                content=json.dumps(
                    {
                        "thinking": "",
                        "evaluation_previous_goal": "Ready",
                        "memory": "",
                        "next_goal": "Find results",
                        "action": [action],
                    }
                )
            )

    def start():
        runtime.browser.live = LiveSessions(tmp_path)
        runtime.browser.live.executor = executor
        if dynamic:
            runtime.browser.live.model_resolver = lambda _: FixtureModel()
        forge = ForgeMindTool(Store(db.path), unexpected, Limits(seconds=60))
        brain = BrainManager(config, bus, tools={forge.name: forge}, tool_executor=executor)
        monkeypatch.setattr(brain, "_generate", unexpected)
        monkeypatch.setattr("jarvis.agent_chat.runner_brain.brain_manager", lambda: brain)
        service = AgentChatService(AgentChatStore(":memory:"), bus=lambda: bus)
        session = service.create_session(
            surface="jarvis", provider="openai-codex", permission_mode="ask"
        )
        return service, session.session_id

    service, session_id = start()

    async def send(text, *, deny=None, cancel=None):
        approved = []
        await service.send(session_id, text)
        async with asyncio.timeout(220 if dynamic else 65):
            while service.is_running(session_id):
                for approval_id in service.pending_approvals(session_id):
                    payload = next(
                        event["payload"]
                        for event in reversed(service.store.list_events(session_id))
                        if event["kind"] == "approval_required"
                        and event["payload"]["approval_id"] == approval_id
                    )
                    name = next(iter(payload["input"]["action"]))
                    approved.append(name)
                    if name == cancel:
                        assert await service.cancel(session_id)
                    else:
                        assert service.resolve_approval(
                            session_id, approval_id, "deny" if name == deny else "allow"
                        )
                for row in await approvals.items(state=ApprovalState.PENDING):
                    name = next(iter(row.action["action"]))
                    approved.append(name)
                    if name == cancel:
                        assert await service.cancel(session_id)
                    else:
                        await approvals.resolve(row.id, approve=name != deny)
                await asyncio.sleep(0.03)
        events = service.store.list_events(session_id)
        turn = [row for row in events if row["kind"] == "turn_finished"][-1]
        if cancel:
            assert turn["payload"]["status"] == "cancelled"
            return "", approved
        assert turn["payload"]["status"] == "done"
        answer = [row for row in events if row["kind"] == "assistant_text"][-1]
        return answer["payload"]["text"], approved

    origin = search_site.origin
    start_url = f"{origin}/catalog"
    learn = "forge learn web" if dynamic else "forge learn web-legacy"
    steps = (
        ["navigate", "click", "input_text", "click", "done"]
        if dynamic
        else ["navigate", "input_text", "click", "verify_results"]
    )
    try:
        objective = (
            f'Learn to search {start_url} for "cobalt"'
            if dynamic
            else f"{learn} {start_url} cobalt"
        )
        learned = await send(objective)
        assert learned == (
            "FORGE procedure 1 is draft. Verification succeeded.",
            steps,
        ), observed_controls
        assert await send("forge inspect 1") == (
            "FORGE procedure 1 is draft. Verified public search steps are "
            + (
                "navigate, click, input text, click, verify results."
                if dynamic
                else "navigate, input text, click, verify results."
            ),
            [],
        )
        assert (await send(f"forge replay 1 web {start_url} violet"))[0].endswith(
            "(promoted_version_required)."
        )
        assert (await send(f"forge validate 1 web {start_url} violet"))[0].endswith(
            "(independent_validation_required)."
        )
        calls_after_learning = len(model_calls)
        if dynamic:
            assert calls_after_learning == 3
            caller.approval_mode = "ask"
            caller.approval_rules = {
                "always_allow": [
                    "core:forge:promote",
                    *(f"core:browser:{step}" for step in steps),
                ],
            }
        state = "promoted" if dynamic else "validated"
        assert await send(f"forge validate 1 web {start_url} amber") == (
            f"FORGE procedure 1 is {state}. Verification succeeded.",
            [] if dynamic else steps,
        )
        if not dynamic:
            assert await send("forge promote 1") == ("FORGE procedure 1 promoted.", [])

        await runtime.browser.live.close()
        service, session_id = start()
        command = (
            f'Use what you learned to search {start_url} for "green marble"'
            if dynamic
            else f"forge replay 1 web {start_url} green marble"
        )
        assert await send(command) == (
            "FORGE procedure 1 is promoted. Verification succeeded.",
            [] if dynamic else steps,
        )
        assert len(model_calls) == calls_after_learning
        # Revocation takes effect on the next action without clearing the skill.
        caller.approval_mode = "always_ask"
        caller.approval_rules = {}
        assert ("/results" if dynamic else "/catalog", "green marble", None) in search_site.requests

        search_site.list_tag = "ol"
        search_site.moved = True
        assert (await send(f"forge replay 1 web {start_url} plum"))[0].endswith(
            "(search_page_drift)."
        )
        search_site.moved = False
        assert (await send(f"forge replay 1 web {origin}/redirect mint"))[0].endswith(
            "(browser_navigation_blocked)." if dynamic else "(search_page_drift)."
        )
        assert not any(path == "/other" for path, _, _ in search_site.requests)
        assert await send(
            f"forge replay 1 web {start_url.replace('127.0.0.1', 'localhost')} mint"
        ) == (
            "FORGE stopped (browser_origin_mismatch).",
            [],
        )

        assert (await send(f"{learn} {origin}/sensitive lemon"))[0].endswith(
            "(browser_restricted_page)." if dynamic else "(search_page_drift)."
        )
        assert (await send(f"{learn} {start_url}?token=secret lemon"))[0].endswith(
            "(public_search_start_url_required)."
        )
        denied_prefix = steps[:2] if dynamic else steps[:3]
        assert (await send(f"{learn} {start_url} peppermint", deny="click")) == (
            "FORGE stopped (browser_approval_denied)."
            if dynamic
            else "FORGE stopped (browser_search_stopped).",
            denied_prefix,
        )
        assert (await send(f"{learn} {start_url} mangosteen", cancel="click"))[1] == denied_prefix
        assert not await approvals.items(state=ApprovalState.PENDING)
        assert not {"peppermint", "mangosteen", "lemon"}.intersection(
            query for _, query, _ in search_site.requests
        )
        assert db.list() == [{"version": 1, "state": "promoted"}]
        raw = db.path.read_bytes() + society.path.read_bytes()
        for private_value in (
            b"/catalog",
            b"private fixture text",
            b"fixture credential",
            b"green marble",
            b"cobalt",
            b"amber",
            b"peppermint",
            b"mangosteen",
            b"token=secret",
            b"private cookie",
        ):
            assert private_value not in raw
    finally:
        await service.cancel_all()
        await runtime.browser.live.close()
        await society.close()
