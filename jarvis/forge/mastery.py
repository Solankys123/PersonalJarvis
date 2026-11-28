"""Authorize learning and persist verified lessons from ordinary chat turns."""

from __future__ import annotations

import asyncio
import re
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from jarvis.agent_chat.control_types import GoalVerdict
from jarvis.core.redact import redact_secrets
from jarvis.skills.authoring.draft_writer import safe_lint_skill_body
from jarvis.skills.authoring.service import SkillAuthoringService, SkillCreateRequest, slugify
from jarvis.skills.skill_context import try_get_skill_context
from jarvis.skills.trigger_matcher import TriggerMatcher

TAG = "forge-mastery"
_COMMAND = re.compile(
    r"(?:master (this skill|the previous task)|learn and master\s+(.+?))[.!]?\s*$",
    re.IGNORECASE | re.DOTALL,
)


def request(text: str) -> str | None:
    match = _COMMAND.fullmatch(text.strip())
    return (match.group(2) or "").strip() if match else None


def mastery_match(text: str) -> Any | None:
    context = try_get_skill_context()
    if context is not None:
        skill = TriggerMatcher(context.registry).match_voice(text)
        if skill and skill.frontmatter and TAG in skill.frontmatter.tags:
            return skill
    return None


def reuse_guidance(text: str) -> str:
    skill = mastery_match(text)
    context = try_get_skill_context()
    if skill is None or context is None:
        return ""
    instructions = context.runner.render_instructions(
        skill, args={"utterance": text, "_trigger": "chat"},
    )
    return (
        "\n\nRelevant saved skill /" + skill.name + " (guidance only). "
        "Use current inputs and ordinary tool permissions. Verify the current result.\n"
        + instructions
    )


class Lesson(BaseModel):
    model_config = ConfigDict(extra="forbid")
    name: str = Field(min_length=1, max_length=64)
    description: str = Field(min_length=1, max_length=400)
    phrases: list[str] = Field(min_length=1, max_length=5)
    body: str = Field(min_length=1, max_length=6000)


class MasteryVerdict(GoalVerdict):
    lesson: Lesson | None = None


LEARNING_INSTRUCTIONS = (
    "This is a user-approved practical learning exercise. Use existing knowledge and "
    "research where relevant. Choose tools as in an ordinary task. Use disposable inputs "
    "for practice unless the user explicitly scoped real data. Stay within the approved "
    "objective. Preserve existing data and stop at login or permission barriers. "
    "Do not send, publish, purchase, apply, delete, or change account settings. "
    "When a method has no effect, inspect the current state and try an authorized "
    "alternative. Use tool observations to verify the result before reporting completion.\n\n"
)


async def learn(handle: Any, task: str, events: list[dict[str, Any]]) -> str | None:
    """Review the actual receipts before offering the exact durable instructions."""
    from jarvis.agent_chat.control_eval import evaluate_goal

    context = try_get_skill_context()
    if context is None:
        return None

    def check_permission() -> bool:
        if handle.cancel.is_set():
            raise asyncio.CancelledError
        current = handle.control_service.store.get_session(handle.session.session_id)
        return current is not None and current.permission_mode == handle.session.permission_mode

    if not check_permission() or not any(
        e["kind"] == "tool_result" and not e["payload"].get("is_error") for e in events
    ):
        return None
    async with asyncio.timeout(120):
        verdict = await evaluate_goal(handle.session, task, events, learning=True)
    if (
        not check_permission()
        or not isinstance(verdict, MasteryVerdict)
        or verdict.status != "complete"
        or verdict.lesson is None
    ):
        return None
    lesson = verdict.lesson
    name = slugify(lesson.name)
    text = "\n".join((lesson.name, lesson.description, *lesson.phrases, lesson.body))
    if (
        not name
        or redact_secrets(text) != text
        or "<redacted:" in text
        or safe_lint_skill_body(lesson.body)
        or re.search(r"(?im)^\s*TOOL:", lesson.body)
        or any(
            not re.fullmatch(r"[\w -]{3,90}", phrase) or len(phrase.split()) < 2
            for phrase in lesson.phrases
        )
    ):
        return None

    prior = mastery_match(task)
    if prior is None and any(skill.name == name for skill in context.registry.list()):
        return None
    from jarvis.agent_chat.runner_brain import APPROVAL_TIMEOUT_S

    async with asyncio.timeout(APPROVAL_TIMEOUT_S):
        decision = await handle.request_approval(
            handle.turn_id, "forge_mastery_save",
            {"name": prior.name if prior else name, **lesson.model_dump(exclude={"name"})},
            "Review these reusable instructions and approve activation if they are safe to save.",
        )
    if decision != "allow" or not check_permission():
        return None
    writer = SkillAuthoringService(registry=context.registry, bus=handle.bus)
    if prior is not None:
        await writer.append_verified_experience(prior, lesson.body)
        return prior.name
    triggers = tuple(
        {"type": "voice", "pattern": r"\b" + r"\s+".join(map(re.escape, phrase.split())) + r"\b"}
        for phrase in lesson.phrases
    )
    draft = await writer.create(SkillCreateRequest(
        name=name, description=lesson.description, tags=(TAG,), triggers=triggers,
        state="draft", body=lesson.body,
    ))
    if not check_permission():
        return None
    context.registry.promote(draft.name)
    return draft.name
