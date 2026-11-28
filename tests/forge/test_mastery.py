"""Mastery wraps ordinary Agent Chat execution and learns from its receipts."""

from __future__ import annotations

import asyncio

import pytest

from jarvis.agent_chat.events import make_event
from jarvis.agent_chat.service import AgentChatService
from jarvis.agent_chat.store import AgentChatStore
from jarvis.forge.mastery import Lesson, MasteryVerdict
from jarvis.skills.registry import SkillRegistry
from jarvis.skills.runner import SkillRunner
from jarvis.skills.skill_context import SkillContext, set_skill_context, try_get_skill_context

TASK = "Create an orchard record with pear,2"
GUIDANCE = "Write the requested row to a new file and read the file back separately."


@pytest.mark.parametrize("case", ["declined", "revoked", "unverified", "private", "verified"])
async def test_mastery_uses_normal_runner_and_reloads_guidance(tmp_path, monkeypatch, case):
    registry = SkillRegistry(tmp_path / "skills")
    previous = try_get_skill_context()
    set_skill_context(SkillContext(registry, SkillRunner(registry)))
    monkeypatch.setattr("jarvis.core.task_agent.subscription_seat_off_loop", _no_reseat)
    service = AgentChatService(AgentChatStore(":memory:"))
    session = service.create_session(
        surface="jarvis", provider="openai-codex", cwd=str(tmp_path), permission_mode="ask",
    )
    sid = session.session_id
    prompts = []

    async def run(handle, prompt, runner, **_):
        assert runner == "codex-cli"
        prompts.append(prompt)
        for kind, data in (
            ("tool_call", {"call_id": "1", "name": "Write", "input": {}}),
            ("tool_result", {"call_id": "1", "output": "written", "is_error": False}),
            ("tool_call", {"call_id": "2", "name": "Read", "input": {}}),
            ("tool_result", {"call_id": "2", "output": "pear,2", "is_error": False}),
            ("turn_finished", {"status": "done", "usage": {}}),
        ):
            await handle.emit(make_event(kind, {"turn_id": handle.turn_id, **data}))

    async def verify(selected, task, events, *, learning):
        assert selected.provider == "openai-codex"
        assert learning and task == TASK
        assert [e["payload"]["output"] for e in events if e["kind"] == "tool_result"] == [
            "written", "pear,2",
        ]
        return MasteryVerdict(
            status="continue" if case == "unverified" else "complete",
            reason="The separate readback verified the result.",
            evidence=["9"],
            lesson=Lesson(
                name="orchard-record",
                description="Create an orchard record.",
                phrases=["create an orchard record"],
                body=GUIDANCE if case != "private" else "Use password: synthetic-secret-value",
            ),
        )

    monkeypatch.setattr("jarvis.agent_chat.service.run_cli_turn", run)
    monkeypatch.setattr("jarvis.agent_chat.control_eval.evaluate_goal", verify)
    stream = service.subscribe(sid)
    cards = []
    try:
        await service.send(sid, "Learn and master " + TASK)
        async with asyncio.timeout(5):
            while (event := await stream.get())["kind"] != "turn_finished":
                if event["kind"] != "approval_required":
                    continue
                payload = event["payload"]
                cards.append(payload["name"])
                if payload["name"] == "forge_mastery":
                    assert not prompts
                    assert payload["input"] == {"objective": TASK}
                    if case == "revoked":
                        service.store.update_session(sid, permission_mode="plan")
                else:
                    assert payload["input"]["body"] == GUIDANCE
                    assert not registry.list_active()
                assert service.resolve_approval(
                    sid, payload["approval_id"], "deny" if case == "declined" else "allow",
                )
        await service.wait_turn(sid)
        assert bool(registry.list_active()) == (case == "verified")
        assert cards == (
            ["forge_mastery", "forge_mastery_save"] if case == "verified" else ["forge_mastery"]
        )
        assert len(prompts) == (0 if case in {"declined", "revoked"} else 1)

        if case == "verified":
            reloaded = SkillRegistry(registry.root)
            reloaded.reload_sync()
            assert len(reloaded.list_active()) == 1
            set_skill_context(SkillContext(reloaded, SkillRunner(reloaded)))
            await service.send(sid, "Create an orchard record with plum,7")
            await asyncio.wait_for(service.wait_turn(sid), timeout=5)
            assert len(prompts) == 2
            assert "plum,7" in prompts[1] and GUIDANCE in prompts[1]
            assert "pear,2" not in prompts[1]
    finally:
        await service.cancel_all()
        set_skill_context(previous)


async def _no_reseat(_):
    return None
