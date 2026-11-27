"""Exercise mastery through the real Agent Chat service and tool executor."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest

from jarvis.agent_chat.service import AgentChatService
from jarvis.agent_chat.store import AgentChatStore
from jarvis.brain.manager import BrainManager
from jarvis.core.bus import EventBus
from jarvis.core.config import JarvisConfig
from jarvis.core.protocols import BrainDelta, HarnessResult, ToolResult
from jarvis.forge.tool import ForgeMindTool
from jarvis.harness.computer_use_context import ComputerUseContext
from jarvis.plugins.tool.computer_use_tool import ComputerUseTool
from jarvis.safety.approval import ApprovalWorkflow
from jarvis.safety.risk_tier import RiskTierEvaluator
from jarvis.safety.tool_executor import ToolExecutor
from jarvis.skills.registry import SkillRegistry
from jarvis.skills.runner import SkillRunner
from jarvis.skills.schema import SkillLifecycleState
from jarvis.skills.skill_context import SkillContext, set_skill_context, try_get_skill_context

DOCS = "https://docs.python.org/3/library/csv.html"
TASK = "Learn and master creating an orchard CSV"
CONTENT = "fruit,count\npear,2\nplum,7\n"


class SearchTool:
    name = "search_web"
    description = "Search documentation"
    schema = {}
    risk_tier = "safe"

    async def execute(self, args, ctx):
        assert args["query"] == "Python CSV official documentation"
        return ToolResult(True, {"results": [{"url": DOCS, "title": "CSV"}]})


class PlanningModel:
    def __init__(self, file_name, expected):
        self.file_name = file_name
        self.expected = expected
        self.calls = 0

    async def complete(self, request):
        self.calls += 1
        if self.calls % 2 == 1:
            yield BrainDelta(content="Python CSV official documentation")
            return
        assert DOCS in request.messages[0].content
        yield BrainDelta(content=json.dumps({
            "name": "orchard-csv",
            "description": "Create and verify an orchard CSV.",
            "phrases": ["create an orchard CSV", "make an orchard CSV"],
            "knowledge": f"Use standard CSV quoting. See [CSV documentation]({DOCS}).",
            "method": "Write the requested header and rows, then read the artifact back.",
            "steps": [{"tool": "Write", "args": {
                "file_path": self.file_name, "content": CONTENT,
            }}],
            "check": {"tool": "Read", "args": {"file_path": self.file_name},
                      "expected": [self.expected]},
        }))


@pytest.fixture
def mastery_chat(tmp_path, monkeypatch):
    bus = EventBus()
    config = JarvisConfig()
    executor = ToolExecutor(bus, RiskTierEvaluator(config.safety), ApprovalWorkflow(bus))
    brain = BrainManager(config, bus=bus, tools={"search_web": SearchTool()},
                         tool_executor=executor)
    monkeypatch.setattr("jarvis.agent_chat.runner_brain.brain_manager", lambda: brain)
    root = tmp_path / "skills"
    registry = SkillRegistry(root)
    previous = try_get_skill_context()
    set_skill_context(SkillContext(registry, SkillRunner(registry)))
    service = AgentChatService(AgentChatStore(":memory:"), bus=lambda: bus)
    cwd = tmp_path / "exercise"
    cwd.mkdir()
    try:
        yield service, cwd, registry
    finally:
        set_skill_context(previous)


async def run_mastery(service, cwd, decision, *, existing=False, revoke=False):
    session = service.create_session(
        surface="jarvis", provider="openai", cwd=str(cwd), permission_mode="ask",
    )
    sid = session.session_id
    events = service.subscribe(sid)
    await service.send(sid, TASK)
    async with asyncio.timeout(5):
        while True:
            event = await events.get()
            if event["kind"] == "turn_finished":
                pytest.fail(event["payload"]["error"])
            if event["kind"] == "approval_required":
                break
    card = event["payload"]
    assert card["name"] == "forge_mastery"
    target = Path(card["input"]["exercise"][0]["args"]["file_path"])
    assert await asyncio.to_thread(target.exists) == existing
    if revoke:
        service.store.update_session(sid, permission_mode="plan")
    assert service.resolve_approval(sid, card["approval_id"], decision)
    await asyncio.wait_for(service.wait_turn(sid), timeout=5)
    emitted = []
    while not events.empty():
        emitted.append(events.get_nowait())
    assert not any(event["kind"] == "approval_required" for event in emitted)
    assert "forge_mastery" not in service.always_allowed(sid)
    return target


@pytest.mark.parametrize("case", ["denied", "wrong_readback", "existing", "revoked"])
async def test_skill_requires_completed_exercise(mastery_chat, monkeypatch, case):
    service, cwd, registry = mastery_chat
    target = cwd / "practice.csv"
    expected = "mango,999" if case == "wrong_readback" else "plum,7"
    if case == "existing":
        target.write_text("keep", encoding="utf-8")
    model = PlanningModel(target.name, expected)
    monkeypatch.setattr("jarvis.agent_chat.browser_model.browser_model_for", lambda *_: model)
    await run_mastery(service, cwd, "deny" if case == "denied" else "allow",
                      existing=case == "existing", revoke=case == "revoked")
    assert not registry.list()
    if case == "existing":
        assert target.read_text(encoding="utf-8") == "keep"
    else:
        assert target.exists() == (case == "wrong_readback")


async def test_mastery_reloads_and_archives_verified_revision(mastery_chat, monkeypatch):
    service, cwd, registry = mastery_chat
    model = PlanningModel("practice.csv", "plum,7")
    monkeypatch.setattr("jarvis.agent_chat.browser_model.browser_model_for", lambda *_: model)
    target = await run_mastery(service, cwd, "allow_always")
    assert target.read_text(encoding="utf-8") == CONTENT

    restarted = SkillRegistry(registry.root)
    restarted.reload_sync()
    skill = restarted.get("orchard-csv")
    assert skill.state is SkillLifecycleState.ACTIVE
    assert "Researched guidance" in skill.body
    assert "Practically verified method" in skill.body
    assert "plum,7" not in skill.body
    set_skill_context(SkillContext(restarted, SkillRunner(restarted)))
    from jarvis.forge.mastery import mastery_match

    assert mastery_match("Please create an orchard CSV for apple and lychee") == skill

    model.file_name = "round-two.csv"
    model.expected = "pear,2"
    await run_mastery(service, cwd, "allow")
    updated = restarted.get("orchard-csv")
    assert updated.frontmatter.version == "0.1.1"
    assert "Verified experience" in updated.body
    assert (skill.path.parent / "history" / "0.1.0.md").is_file()


async def test_subscription_reuses_reloaded_skill_with_changed_inputs(mastery_chat, monkeypatch):
    service, cwd, registry = mastery_chat
    model = PlanningModel("practice.csv", "plum,7")
    monkeypatch.setattr("jarvis.agent_chat.browser_model.browser_model_for", lambda *_: model)
    await run_mastery(service, cwd, "allow")
    restarted = SkillRegistry(registry.root)
    restarted.reload_sync()
    set_skill_context(SkillContext(restarted, SkillRunner(restarted)))

    class BindingModel:
        calls = 0

        async def complete(self, request):
            self.calls += 1
            text = request.messages[0].content
            assert "Write the requested header and rows" in text
            assert "apple,11" in text
            yield BrainDelta(content=json.dumps({
                "steps": [{"tool": "Write", "args": {
                    "file_path": "reuse.csv", "content": "fruit,count\napple,11\n",
                }}],
                "check": {"tool": "Read", "args": {"file_path": "reuse.csv"},
                          "expected": ["apple,11"]},
            }))

    binding_model = BindingModel()
    monkeypatch.setattr(
        "jarvis.agent_chat.browser_model.browser_model_for", lambda *_: binding_model,
    )
    session = service.create_session(
        surface="jarvis", provider="openai-codex", cwd=str(cwd), permission_mode="accept-edits",
    )
    sid = session.session_id
    await service.send(sid, "Please create an orchard CSV named reuse.csv with row apple,11.")
    await asyncio.wait_for(service.wait_turn(sid), timeout=5)
    assert binding_model.calls == 1
    assert (cwd / "reuse.csv").read_text(encoding="utf-8") == "fruit,count\napple,11\n"
    assert not service.pending_approvals(sid)


@pytest.mark.parametrize("verified", [False, True])
async def test_adaptive_browser_mastery_and_reuse(mastery_chat, monkeypatch, verified):
    service, cwd, registry = mastery_chat
    url = "https://example.org/search"
    queries = []

    class BrowserForge(ForgeMindTool):
        async def _run_live_search(self, mode, version, procedure, state, entry, query, ctx):
            assert mode == "mastery" and entry == url
            ctx.config["forge_search_guard"]()
            if not queries:
                assert await ctx.config["forge_ask"]("id", "action", {}, "search") == "allow"
            queries.append(query)
            return ToolResult(verified, None, None if verified else "search_results_unverified")

    class BrowserModel:
        calls = 0

        async def complete(self, request):
            self.calls += 1
            if self.calls == 1:
                yield BrainDelta(content="Python CSV official documentation")
                return
            prompt = request.messages[0].content
            assert '"browser_search"' in prompt
            result = {"browser_search": {
                "url": url, "query": "sample jobs" if self.calls == 2 else "design jobs",
            }}
            if self.calls == 2:
                result.update(
                    name="public-jobs", description="Find jobs using a public search form.",
                    phrases=["search public jobs"], knowledge=f"Reference {DOCS}.",
                    method="Search the public job board and check the rendered results.",
                )
            yield BrainDelta(content=json.dumps(result))

    from jarvis.agent_chat.runner_brain import brain_manager

    brain_manager()._tools.update({
        "forge_mind": BrowserForge(None, None, None),
        "society_browser": SearchTool(),
    })
    model = BrowserModel()
    monkeypatch.setattr("jarvis.agent_chat.browser_model.browser_model_for", lambda *_: model)
    session = service.create_session(
        surface="jarvis", provider="openai-codex", cwd=str(cwd), permission_mode="ask",
    )
    sid = session.session_id
    events = service.subscribe(sid)
    await service.send(sid, "Learn and master how to search public jobs")
    async with asyncio.timeout(5):
        while (event := await events.get())["kind"] != "approval_required":
            assert event["kind"] != "turn_finished", event
    card = event["payload"]
    assert card["name"] == "forge_mastery"
    assert card["input"]["exercise"] == {"url": url, "query": "sample jobs"}
    assert not queries
    assert service.resolve_approval(sid, card["approval_id"], "allow")
    await asyncio.wait_for(service.wait_turn(sid), timeout=5)
    assert queries == ["sample jobs"]
    assert [row["payload"]["name"] for row in service.store.list_events(sid)
            if row["kind"] == "approval_required"] == ["forge_mastery"]
    assert bool(registry.list_active()) == verified
    if verified:
        restarted = SkillRegistry(registry.root)
        restarted.reload_sync()
        set_skill_context(SkillContext(restarted, SkillRunner(restarted)))
        await service.send(sid, "Search public jobs for design jobs")
        await asyncio.wait_for(service.wait_turn(sid), timeout=5)
        assert queries == ["sample jobs", "design jobs"]
        assert model.calls == 3


@pytest.mark.parametrize("proof", ["", "Matching jobs are visible for the search query."])
async def test_signed_in_mastery_awaits_desktop_verifier(mastery_chat, monkeypatch, proof):
    from jarvis.agent_chat.runner_brain import brain_manager

    service, cwd, registry = mastery_chat
    goals = []

    class DesktopHarness:
        async def dispatch(self, name, task):
            assert name == "screenshot"
            goals.append(task.prompt)
            stdout = f"[cu] done (verified: {proof})" if proof else "[cu] done"
            yield HarnessResult(stdout=stdout, exit_code=0, is_final=True)

    class DesktopModel:
        calls = 0

        async def complete(self, request):
            self.calls += 1
            if self.calls == 1:
                yield BrainDelta(content="Python CSV official documentation")
                return
            prompt = request.messages[0].content
            assert '"desktop"' in prompt
            assert '"browser_search":' not in prompt
            body = {"desktop": {
                "application": "Brave on Naukri",
                "goal": "search for sample jobs" if self.calls == 2 else "search for design jobs",
                "evidence": "Visible job cards match the search query.",
            }}
            if self.calls == 2:
                body.update(
                    name="signed-in-job-search", description="Search the open job site.",
                    phrases=["search Naukri jobs"], knowledge=f"Reference {DOCS}.",
                    method="Search in the open application and verify visible job cards.",
                )
            yield BrainDelta(content=json.dumps(body))

    brain = brain_manager()
    monkeypatch.setattr(
        "jarvis.plugins.tool.computer_use_tool.peek_computer_use_context",
        lambda: ComputerUseContext(
            vision_engine=object(), brain_manager=brain, tool_executor=brain._tool_executor,
        ),
    )
    brain._tools.update({
        "computer_use": ComputerUseTool(bus=brain._bus, manager=DesktopHarness()),
        "forge_mind": ForgeMindTool(None, None, None),
        "society_browser": SearchTool(),
    })
    model = DesktopModel()
    monkeypatch.setattr("jarvis.agent_chat.browser_model.browser_model_for", lambda *_: model)
    session = service.create_session(
        surface="jarvis", provider="openai-codex", cwd=str(cwd), permission_mode="ask",
    )
    sid = session.session_id
    events = service.subscribe(sid)
    await service.send(sid, "Learn and master how to search Naukri jobs in my signed-in Brave")
    async with asyncio.timeout(5):
        while (event := await events.get())["kind"] != "approval_required":
            assert event["kind"] != "turn_finished", event
    card = event["payload"]
    assert card["name"] == "forge_mastery"
    assert card["input"]["exercise"]["tool"] == "computer_use"
    assert not goals
    assert service.resolve_approval(sid, card["approval_id"], "allow")
    await asyncio.wait_for(service.wait_turn(sid), timeout=5)
    assert goals == [card["input"]["exercise"]["goal"]]
    assert [
        row["payload"]["name"] for row in service.store.list_events(sid)
        if row["kind"] == "approval_required"
    ] == ["forge_mastery"]
    assert bool(registry.list_active()) == bool(proof)
    if proof:
        await service.send(sid, "Search Naukri jobs for design jobs")
        await asyncio.wait_for(service.wait_turn(sid), timeout=5)
        assert "search for design jobs" in goals[1]
        assert model.calls == 3
