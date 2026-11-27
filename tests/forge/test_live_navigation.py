import asyncio
from dataclasses import asdict
from types import SimpleNamespace
from uuid import uuid4

import pytest

from jarvis.forge.engine import Evidence, Refused, State, Verdict
from jarvis.forge.live_navigation import DynamicSearch, Navigation, PageShape
from jarvis.forge.store import Store


def test_navigation_uses_ephemeral_parameters():
    origin = "http://127.0.0.1:8765"
    first = origin + "/private-path?q=private-value"
    second = origin + "/other?query=second-value"
    shape = PageShape(main=True, heading=True)
    procedure = Navigation.capture(
        allowed_origin=origin,
        requested_url=first,
        final_url=first,
        after=shape,
        succeeded=True,
    )
    assert asdict(procedure) == {"origin": origin, "after": {"main": True, "heading": True}}
    assert procedure.bind(second) == second
    procedure.verify(second, second, shape)
    with pytest.raises(Refused, match="browser_navigation_drift"):
        procedure.verify(second, second, PageShape(main=False, heading=True))
    with pytest.raises(Refused, match="browser_navigation_drift"):
        procedure.verify(second, origin + "/redirect", shape)


@pytest.mark.parametrize(
    "url",
    [
        "https://other.example/",
        "https://example.test:8443/",
        "https://example.test@other.example/",
        "https://user:secret@example.test/",
        "https://example.test\\@other.example/",
        "https://example.test\n/",
        "file:///index.html",
    ],
)
def test_navigation_refuses_invalid_or_other_origins(url):
    procedure = Navigation("https://example.test", PageShape(main=True, heading=True))
    with pytest.raises(Refused):
        procedure.bind(url)


def test_navigation_requires_successful_observed_destination():
    url = "https://example.test/"
    with pytest.raises(Refused, match="navigation_unverified"):
        Navigation.capture(
            allowed_origin="https://example.test",
            requested_url=url,
            final_url=url,
            after=PageShape(main=True, heading=True),
            succeeded=False,
        )


def test_promoted_dynamic_search_lookup_after_reopen(tmp_path):
    path = tmp_path / "forge.sqlite"
    store = Store(path)
    origin = "https://example.test"
    steps = ("navigate", "input_text:" + "a" * 64, "click:" + "b" * 64, "verify_results")
    spec = DynamicSearch(origin, steps)

    def evidence(target, length):
        return [
            Evidence(
                target,
                step,
                f"length:{length}",
                State.BLANK,
                State.MATCH,
                Verdict.SUCCESS,
                "verified",
                index,
            )
            for index, step in enumerate(steps, 1)
        ]

    assert store.record("learn", "learn", None, Verdict.SUCCESS, evidence("first", 6), spec) == (
        1,
        Verdict.SUCCESS,
    )
    assert store.promoted_search(origin) is None
    assert store.record(
        "validate", "validate", 1, Verdict.SUCCESS, evidence("second", 4), spec
    ) == (
        1,
        Verdict.SUCCESS,
    )
    store.promote(1, authorized=True)
    reopened = Store(path)
    assert reopened.promoted_search(origin) == 1
    assert reopened.promoted_search("https://other.test") is None
    assert reopened.get(1) == (spec, "promoted")


def test_web_deadline_configuration_preserves_fixture_limit():
    from pydantic import ValidationError

    from jarvis.core.config import ForgeConfig

    config = ForgeConfig.model_validate({"run_seconds": 60, "web_run_seconds": "180"})
    assert config.run_seconds == 60
    assert config.web_run_seconds == 180
    with pytest.raises(ValidationError, match="web_run_seconds"):
        ForgeConfig.model_validate({"web_run_seconds": 541})


@pytest.mark.asyncio
@pytest.mark.parametrize("state", ["pending", "denied", "approved"])
async def test_browser_approval_deadline_releases_pending_card(state):
    from jarvis.society.browser.bridge import wait_for_browser_approval

    resolved = []
    events = []

    async def get(_):
        return SimpleNamespace(state=state)

    async def resolve(approval_id, *, approve, note):
        resolved.append((approval_id, approve, note))

    runtime = SimpleNamespace(approvals=SimpleNamespace(get=get, resolve=resolve))
    session = SimpleNamespace(publish=events.append)
    if state == "pending":
        with pytest.raises(TimeoutError):
            await wait_for_browser_approval(runtime, "one", session, seconds=0.02)
        assert resolved == [("one", False, "Browser task ended")]
    else:
        assert await wait_for_browser_approval(runtime, "one", session, seconds=1) == (
            state == "approved"
        )
        assert not resolved
    assert events == [{"kind": "approval_cleared", "id": "one"}]


@pytest.mark.asyncio
@pytest.mark.parametrize("seconds, expected", [(0.25, "browser_model_timeout"), (2, None)])
async def test_dynamic_model_deadline(seconds, expected):
    from jarvis.core.protocols import BrainDelta
    from jarvis.society.browser.bridge import execute_live

    class SlowModel:
        async def complete(self, _request):
            await asyncio.sleep(0.4)
            yield BrainDelta(content="{}")

    async def run(_caller, *, llm, **_kwargs):
        answer = await llm({"messages": [{"role": "user", "content": "fixture"}]})
        return {"ok": answer["ok"], "error": answer.get("error")}

    live = SimpleNamespace(
        stopped_turns=set(), executor=object(), model_resolver=lambda _: SlowModel(), run=run
    )
    caller = SimpleNamespace(agent_id="lead", provider="openai", model="test")
    result = await execute_live(
        SimpleNamespace(data_dir="."),
        caller,
        SimpleNamespace(live=live),
        {},
        SimpleNamespace(config={}),
        dynamic_search={"model_seconds": seconds},
    )
    assert result.success is (expected is None)
    assert result.error == expected


@pytest.mark.asyncio
async def test_dynamic_session_deadline_cancels_the_browser_run(tmp_path, monkeypatch):
    from jarvis.core.protocols import ExecutionContext
    from jarvis.forge.engine import Limits
    from jarvis.forge.tool import ForgeMindTool

    started = asyncio.Event()
    cleaned = asyncio.Event()

    async def roster_get(_):
        return SimpleNamespace(state="active")

    async def stalled_run(*_args, **_kwargs):
        try:
            started.set()
            await asyncio.Future()
        finally:
            cleaned.set()

    runtime = SimpleNamespace(
        lead_id="lead",
        browser=SimpleNamespace(_python=None),
        roster=SimpleNamespace(get=roster_get),
    )
    monkeypatch.setattr("jarvis.society.runtime.current_runtime", lambda: runtime)
    monkeypatch.setattr("jarvis.society.browser.bridge.execute_live", stalled_run)
    store = Store(tmp_path / "forge.sqlite")
    forge = ForgeMindTool(store, None, Limits(), web_seconds=0.2)
    ctx = ExecutionContext(
        trace_id=uuid4(),
        user_utterance="forge learn web https://example.test/search fixture",
        config={"forge_interactive": True, "approval_ref": "agent-chat:test"},
        memory_read=None,
    )
    result = await forge.execute({}, ctx)
    assert result.error == "browser_session_timeout"
    assert started.is_set() and cleaned.is_set()
    assert store.list() == []
