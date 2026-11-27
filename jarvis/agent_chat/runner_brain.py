"""The Jarvis surface's runner: Jarvis' own brain, driven by text.

This is what the front page's chat runs on — every seat of it, since that
surface gave up its CLI seats (2026-08-26): a provider API behind a key, or
a server on this machine. A typed turn is the same turn a spoken one is —
``BrainManager.generate`` with its router, its tools, its memory, its wiki
and its safety tiers — with the keyboard in place of the microphone. Nothing
here spawns a coding CLI or a second agent loop; there is one assistant in
this app and this is how you type at it (maintainer, 2026-08-25).

What the composer's pick changes is which model Jarvis thinks with FOR THIS
CHAT, and how hard: provider, model and effort travel as a
:class:`jarvis.brain.turn_override.TurnOverride` on the one ``generate``
call, and nothing else. The live brain the voice runs on — its active
provider, its config, its dead-lists — is never touched; the first version
of this runner (2026-08-24) switched it and was reverted for exactly that.

The chat's folder is Jarvis' working folder for the turn: the folder tools
(:mod:`jarvis.agent_chat.folder_tools`) join Jarvis' own tools on the turn's
surface, every call goes through ``ToolExecutor``, and the chat's permission
stance decides what asks — through a clickable card, not a spoken "yes"
(:mod:`jarvis.agent_chat.approval_bridge`).
"""

from __future__ import annotations

import asyncio
import logging
import re
import time
import uuid
from collections.abc import Callable, Mapping
from dataclasses import replace
from pathlib import Path
from typing import TYPE_CHECKING, Any, Final, get_args

from jarvis.agent_chat.approval_bridge import ChatApprovalBridge, ChatGrant, approval_ref
from jarvis.agent_chat.effort import normalize_effort
from jarvis.agent_chat.events import make_event
from jarvis.agent_chat.folder_tools import PLAN_STANCE, folder_tools, plan_filter
from jarvis.agent_chat.runner_api import messages_from_events
from jarvis.agent_chat.surface_kits import kit_for
from jarvis.agent_chat.tool_catalog import ToolChoice
from jarvis.agent_chat.tools import summarize_call
from jarvis.brain.turn_override import TurnOverride
from jarvis.core.protocols import BrainMessage, ReasoningEffort, Tool

if TYPE_CHECKING:
    from uuid import UUID

    from jarvis.agent_chat.runner_api import TurnHandle
    from jarvis.agent_chat.store import AgentChatSession

log = logging.getLogger(__name__)

#: The runner id this module answers to, in the catalog and in ``turn_started``.
RUNNER: Final[str] = "brain"

#: ``MessageSent.source_layer`` for a turn typed into the chat surface. The
#: brain's router reads it; it is a conversational layer, so the router may
#: still spawn a worker on an explicit request (``_NON_SPAWN_SOURCE_LAYERS``
#: does not list it — typing at Jarvis is not a background channel).
SOURCE_LAYER: Final[str] = "ui.chat.typed"

#: How long a card may stay open. A person reading a chat is not on a spoken
#: turn's clock; the executor clamps this to its own ceiling.
APPROVAL_TIMEOUT_S: Final[float] = 600.0

#: The tool-loop ceiling of a chat turn — the API runner's own round budget.
MAX_TURNS: Final[int] = 40

#: How much of the chat's own log rides along as history.
_HISTORY_MAX: Final[int] = 40

_EFFORTS: Final[frozenset[str]] = frozenset(get_args(ReasoningEffort))


def brain_manager() -> Any | None:
    """The live BrainManager, or ``None`` before the brain finished building."""
    from jarvis.core import runtime_refs

    return runtime_refs.get_brain_manager()


# ------------------------------------------------------------------ history


def brain_history_from_events(
    events: list[dict[str, Any]], *, max_messages: int | None = _HISTORY_MAX
) -> list[BrainMessage]:
    """The chat's own log as the turn's history: what was said, in prose.

    User and assistant TEXT only. A tool round is folded into the assistant
    message as one line per call ("[used tool X: …]") and its results are
    dropped: Jarvis' loop rebuilds its own tool rounds, and replaying a
    foreign provider's ``tool_use`` ids into a different provider fails its
    validation. Capped to the last ``_HISTORY_MAX`` messages.
    """
    out: list[BrainMessage] = []
    for message in messages_from_events(events, max_events=None if max_messages is None else 600):
        if message.role == "user":
            text = message.content if isinstance(message.content, str) else ""
            if text.strip():
                out.append(BrainMessage(role="user", content=text))
        elif message.role == "assistant":
            text = _assistant_prose(message.content)
            if not text.strip():
                continue
            if out and out[-1].role == "assistant":
                # One answer per turn: the tool round and the prose after it
                # are two provider messages but one thing Jarvis said.
                merged = f"{out[-1].content}\n{text}"
                out[-1] = BrainMessage(role="assistant", content=merged)
            else:
                out.append(BrainMessage(role="assistant", content=text))
    return out[-max_messages:] if max_messages else out


def _assistant_prose(content: str | list[dict[str, Any]]) -> str:
    if isinstance(content, str):
        return content
    lines: list[str] = []
    for block in content:
        kind = block.get("type")
        if kind == "text":
            text = str(block.get("text") or "")
            if text.strip():
                lines.append(text)
        elif kind == "tool_use":
            name = str(block.get("name") or "tool")
            args = block.get("input") or {}
            summary = summarize_call(name, args) if isinstance(args, dict) else ""
            lines.append(f"[used tool {name}: {summary}]" if summary else f"[used tool {name}]")
    return "\n".join(lines)


# ----------------------------------------------------------------- override


def build_override(
    session: AgentChatSession,
    brain: Any,
    *,
    stance: str,
    cwd: Path,
    ref: str,
    kit_tools: Mapping[str, Tool] | None = None,
    system_extra: str = "",
) -> TurnOverride:
    """The pick for this turn: the session's provider / model / effort, the
    surface's hands (its kit's tools, else the folder's), and the tool context
    that routes approvals to the card."""
    kit = kit_for(session.surface)
    provider = session.provider
    model = (session.model or "").strip() or _default_model(brain, provider)
    effort = normalize_effort(provider, session.effort)
    reasoning_effort: ReasoningEffort | None = (
        effort if effort in _EFFORTS else None  # type: ignore[assignment]
    )
    return TurnOverride(
        provider=provider,
        model=model,
        reasoning_effort=reasoning_effort,
        tools_extra=(
            dict(kit_tools) if kit_tools is not None else folder_tools(cwd, stance=stance)
        ),
        tool_filter=_compose_filters(
            _typed_chat_tools,
            kit.tool_filter,
            kit.session_tool_filter(session) if kit.session_tool_filter is not None else None,
            plan_filter if stance == PLAN_STANCE else None,
        ),
        credential_scope=kit.credential_scope,
        system_extra=system_extra,
        tool_context={
            # A typed answer is read, not heard: no spoken two-liners.
            "delivery": "written",
            # The executor asks the card, never the voice sentinel.
            "approval_surface": "interactive",
            "approval_ref": ref,
            "approval_timeout_s": APPROVAL_TIMEOUT_S,
            "tool_origin": kit.tool_origin,
            **({"chat_read_only": True} if stance == PLAN_STANCE else {}),
            "cwd": str(cwd),
        },
        max_turns=kit.max_turns or MAX_TURNS,
    )


def _typed_chat_tools(tools: dict[str, Tool]) -> dict[str, Tool]:
    """Keep voice transcript retrieval off the typed model's tool surface."""
    return {name: tool for name, tool in tools.items() if name != "session-latest-turn"}


def _compose_filters(
    *filters: Callable[[dict[str, Tool]], dict[str, Tool]] | None,
) -> Callable[[dict[str, Tool]], dict[str, Tool]] | None:
    """The kit's filter first, then the stance's; ``None`` when neither applies."""
    active = [f for f in filters if f is not None]
    if not active:
        return None

    def _apply(tools: dict[str, Tool]) -> dict[str, Tool]:
        for f in active:
            tools = f(tools)
        return tools

    return _apply


async def kit_payload(session: AgentChatSession, brain: Any) -> tuple[dict[str, Tool] | None, str]:
    """The surface kit's per-turn hands and briefing: ``(tools, system_extra)``.

    ``(None, "")`` on a surface without its own kit tools (the folder tools
    then apply). A kit builder that fails loses only its part — the turn
    still runs, and the failure is in the log.
    """
    kit = kit_for(session.surface)
    cfg = getattr(brain, "_config", None)
    tools: dict[str, Tool] | None = None
    extra = ""
    coding_tool: Tool | None = None
    if session.surface == "jarvis":
        from jarvis.society.surface import coding_tool_for_session

        try:
            coding_tool = await coding_tool_for_session(session.session_id)
            if coding_tool is not None:
                tools = folder_tools(Path(session.cwd), stance=session.permission_mode or "ask")
                tools[coding_tool.name] = coding_tool
        except Exception:
            log.warning("Jarvis chat: coding-session capability unavailable", exc_info=True)
    if kit.session_tools is not None:
        try:
            tools = kit.session_tools(cfg, brain, session)
        except Exception as exc:  # noqa: BLE001 - the turn runs without the kit's hands
            log.warning("surface %s: kit tools not built: %s", session.surface, exc, exc_info=True)
            tools = {}
        if coding_tool is not None:
            # The kit's own hands replace the folder set; the granted
            # coding-session control must survive that replacement.
            tools = {**(tools or {}), coding_tool.name: coding_tool}
    elif kit.tools is not None:
        try:
            tools = kit.tools(cfg, brain)
        except Exception as exc:  # noqa: BLE001 — the turn runs without the kit's hands
            log.warning("surface %s: kit tools not built: %s", session.surface, exc, exc_info=True)
            tools = {}
    if kit.session_system_extra is not None:
        try:
            extra = await kit.session_system_extra(cfg, brain, session)
        except Exception as exc:  # noqa: BLE001 - the turn runs without the briefing
            log.warning(
                "surface %s: kit briefing not built: %s", session.surface, exc, exc_info=True
            )
    elif kit.system_extra is not None:
        try:
            extra = await kit.system_extra(cfg, brain)
        except Exception as exc:  # noqa: BLE001 — the turn runs without the briefing
            log.warning(
                "surface %s: kit briefing not built: %s", session.surface, exc, exc_info=True
            )
    return tools, extra


def _default_model(brain: Any, provider: str) -> str | None:
    fast = getattr(brain, "_fast_model", None)
    if callable(fast):
        try:
            return fast(provider) or None
        except Exception:  # noqa: BLE001 — the provider's own default then
            return None
    return None


# ------------------------------------------------------------- step mirror


class _StepMirror:
    """Turns THIS turn's bus events into the timeline's step rows.

    The brain does not report its work to the chat; it publishes it on the
    app bus, where the voice lane and the run inspector read it too. This
    subscribes for the length of one turn and translates — for the events
    that carry this turn's trace only, so a voice turn running alongside
    never draws rows in the chat:

    * ``ActionProposed`` -> a tool row, plus its ``rationale`` as reasoning
      text. That sentence is the model's OWN words for why it reaches for the
      tool — never a second "explain yourself" call.
    * ``ActionExecuted`` / ``ActionDenied`` -> that row's result.
    * ``ToolCallStarted`` / ``ToolCallCompleted`` -> the same for the paths
      that publish those instead.

    Rows are paired per tool name, oldest open first. Read-only and
    defensive: a malformed event never reaches the brain (AP-18).
    """

    __slots__ = ("_bus", "_children", "_emit", "_open", "_seen_text", "_trace_id", "_turn_id")

    def __init__(self, emit: Any, turn_id: str, bus: Any | None, trace_id: UUID) -> None:
        self._emit = emit
        self._turn_id = turn_id
        self._bus = bus
        self._trace_id = trace_id
        self._open: list[tuple[str, str]] = []  # (tool name, call id)
        self._children: set[UUID] = set()
        self._seen_text: set[str] = set()

    def start(self) -> None:
        if self._bus is not None and hasattr(self._bus, "subscribe_all"):
            self._bus.subscribe_all(self._on_event)

    def stop(self) -> None:
        if self._bus is not None and hasattr(self._bus, "unsubscribe_all"):
            try:
                self._bus.unsubscribe_all(self._on_event)
            except Exception:  # noqa: BLE001 — detaching must never raise
                log.debug("brain runner: could not detach the step mirror", exc_info=True)

    def open_call_id(self, tool_name: str) -> str:
        """The open row for ``tool_name`` (the bridge's card hangs off it), else ""."""
        for name, call_id in self._open:
            if name == tool_name:
                return call_id
        return ""

    async def _on_event(self, event: Any) -> None:
        try:
            await self._translate(event)
        except Exception:  # noqa: BLE001 — AP-18: never leave a subscriber
            log.debug("brain runner: step mirror skipped an event", exc_info=True)

    def _mine(self, event: Any) -> bool:
        return getattr(event, "trace_id", None) == self._trace_id

    async def _open_call(self, tool_name: str, args: Any) -> None:
        call_id = uuid.uuid4().hex
        self._open.append((tool_name, call_id))
        if isinstance(args, dict):
            payload_in: Any = args
        elif args:
            payload_in = {"arguments": str(args)}
        else:
            payload_in = {}
        await self._emit(
            "tool_call",
            {"turn_id": self._turn_id, "call_id": call_id, "name": tool_name, "input": payload_in},
        )

    async def _close_call(self, tool_name: str, *, ok: bool, output: str, duration_ms: Any) -> None:
        index = next((i for i, (name, _) in enumerate(self._open) if name == tool_name), None)
        if index is None:
            # A result with no row of its own (another path's, or a report
            # without a start) would draw a headless row; drop it.
            return
        _, call_id = self._open.pop(index)
        await self._emit(
            "tool_result",
            {
                "turn_id": self._turn_id,
                "call_id": call_id,
                "output": str(output)[:2000],
                "is_error": not ok,
                "duration_ms": int(duration_ms or 0),
            },
        )

    async def _translate(self, event: Any) -> None:
        name = type(event).__name__
        if name == "ActionProposed" and self._mine(event):
            text = (getattr(event, "rationale", "") or "").strip()
            if text and text not in self._seen_text:
                self._seen_text.add(text)
                await self._emit("reasoning_delta", {"turn_id": self._turn_id, "text": text})
            await self._open_call(
                getattr(event, "tool_name", "") or "tool", getattr(event, "args", None)
            )
        elif name == "ActionExecuted" and self._mine(event):
            await self._close_call(
                getattr(event, "tool_name", "") or "tool",
                ok=bool(getattr(event, "success", False)),
                output=getattr(event, "error", None) or getattr(event, "output_preview", "") or "",
                duration_ms=getattr(event, "duration_ms", 0),
            )
        elif name == "ActionDenied" and self._mine(event):
            await self._close_call(
                getattr(event, "tool_name", "") or "tool",
                ok=False,
                output=str(getattr(event, "reason", "") or "denied"),
                duration_ms=0,
            )
        elif (
            name == "ToolCallStarted" and getattr(event, "parent_trace_id", None) == self._trace_id
        ):
            self._children.add(event.trace_id)
            await self._open_call(
                getattr(event, "tool_name", "") or "tool", getattr(event, "args_preview", "") or ""
            )
        elif name == "ToolCallCompleted" and getattr(event, "trace_id", None) in self._children:
            self._children.discard(event.trace_id)
            tool_name = getattr(event, "tool_name", "") or (
                self._open[0][0] if self._open else "tool"
            )
            await self._close_call(
                tool_name,
                ok=bool(getattr(event, "success", False)),
                output=getattr(event, "error", None) or getattr(event, "output_preview", "") or "",
                duration_ms=getattr(event, "duration_ms", 0),
            )


# ------------------------------------------------------------------- turn


async def run_brain_turn(
    handle: TurnHandle,
    text: str,
    *,
    bridge: ChatApprovalBridge | None,
    always_allowed: set[str],
    tool_choices: list[ToolChoice] | None = None,
    forge_command: bool = False,
    mastery_command: bool = False,
    mastered_reuse: bool = False,
) -> None:
    """Run one typed turn on Jarvis' brain, streaming it into the timeline."""
    started = time.monotonic()
    turn_id = handle.turn_id
    message_id = uuid.uuid4().hex
    session = handle.session

    async def emit(kind: str, payload: dict[str, Any]) -> None:
        await handle.emit(make_event(kind, payload))

    mirror = _StepMirror(emit, turn_id, handle.bus, handle.trace_id)
    ref = approval_ref(session.session_id)

    async def finish(status: str, usage: dict[str, Any], error: str | None = None) -> None:
        await emit(
            "turn_finished",
            {
                "turn_id": turn_id,
                "status": status,
                "duration_ms": int((time.monotonic() - started) * 1000),
                "usage": usage,
                "error": error,
            },
        )

    brain = brain_manager()
    if brain is None or not callable(getattr(brain, "generate", None)):
        await finish(
            "error",
            {},
            "FORGE is unavailable while Jarvis is starting up."
            if forge_command
            else "Jarvis' brain is still starting up. Give it a moment and send again.",
        )
        return

    cwd = Path(session.cwd or Path.home())
    stance = handle.stance or "ask"
    if forge_command:
        stance = PLAN_STANCE if stance == "read-only" else stance
        kit_tools, system_extra = {}, ""
    else:
        kit_tools, system_extra = await kit_payload(session, brain)
    if tool_choices:
        from jarvis.agent_chat.tool_catalog import (
            live_catalog,
            resolve_choices,
            selection_briefing,
            selection_tools,
        )

        try:
            current = await asyncio.to_thread(live_catalog, brain, cwd=str(cwd), stance=stance)
            choices = resolve_choices([row.id for row in tool_choices], current)
            base = dict(getattr(brain, "_tools", {}) or {})
            extra = dict(kit_tools) if kit_tools is not None else folder_tools(cwd, stance=stance)
            base.update(extra)
            extra.update(selection_tools(choices, base))
            kit_tools = extra
            system_extra += selection_briefing(choices)
        except ValueError as exc:
            await finish("error", {}, str(exc))
            return
    override = build_override(
        session,
        brain,
        stance=stance,
        cwd=cwd,
        ref=ref,
        kit_tools=kit_tools,
        system_extra=system_extra,
    )

    async def forge_progress(message: str) -> None:
        await emit("reasoning_delta", {"turn_id": turn_id, "text": message + "\n"})

    override = replace(
        override,
        tool_context={
            **override.tool_context,
            "forge_interactive": forge_command,
            **(
                {
                    "forge_browser_model": (override.provider, override.model),
                    "forge_progress": forge_progress,
                    "forge_ask": handle.request_approval,
                }
                if forge_command
                else {}
            ),
        },
    )
    if not (forge_command or mastery_command or mastered_reuse):
        _note_skill_trigger(brain, text)

    if bridge is not None:
        from jarvis.society.surface import requires_explicit_approval

        bridge.arm(
            ref,
            ChatGrant(
                session_id=session.session_id,
                turn_id=turn_id,
                stance=stance,
                always_allowed=always_allowed,
                ask=handle.request_approval,
                call_id_for=mirror.open_call_id,
                force_ask=(
                    lambda name, args: requires_explicit_approval(session.session_id, name, args)
                )
                if session.surface == "society"
                else lambda _name, _args: False,
            ),
        )

    # The "thinking" line the timeline shows until the first words arrive,
    # and the watcher that fills it: every tool Jarvis reaches for and every
    # sentence it writes next to one lands in the timeline as it happens.
    await emit("reasoning_started", {"turn_id": turn_id, "message_id": message_id})
    mirror.start()

    loop = asyncio.get_running_loop()
    seen: list[str] = []

    def feed(chunk: str) -> None:
        """Called by the brain as the answer is produced — a CHUNK, not the whole so far."""
        if not chunk:
            return
        seen.append(chunk)
        # The consumer is synchronous (the brain's contract); hand the emit
        # back to the loop rather than blocking the producer.
        asyncio.run_coroutine_threadsafe(
            emit("text_delta", {"turn_id": turn_id, "message_id": message_id, "text": chunk}),
            loop,
        )

    status = "done"
    error: str | None = None
    reply = ""
    try:
        if mastery_command or mastered_reuse:
            from jarvis.forge.mastery import MasteryError

            try:
                task = asyncio.create_task(
                    _master_skill(
                        brain, text, handle, override, mirror, bridge, ref, reuse=mastered_reuse,
                    ),
                    name=f"agent-chat-mastery-{turn_id[:8]}",
                )
                reply = await _wait_for_turn_task(task, handle)
            except MasteryError as exc:
                reply = str(exc)
                status, error = "error", "skill_mastery_unverified"
        else:
            reply = await _generate(brain, text, handle, override, feed)
    except asyncio.CancelledError:
        status = "cancelled"
    except Exception as exc:  # noqa: BLE001 — the timeline must never spin forever
        log.exception("brain chat turn %s failed", turn_id)
        status, error = "error", f"{type(exc).__name__}: {exc}"
    finally:
        if bridge is not None:
            bridge.disarm(ref)
        mirror.stop()

    if status == "done" and handle.cancel.is_set():
        status = "cancelled"
    if status == "done" or reply:
        answer = (reply or "").strip() or "".join(seen).strip()
        if answer:
            await emit(
                "assistant_text", {"turn_id": turn_id, "message_id": message_id, "text": answer}
            )
        # Keep the safe explanation visible, but distinguish provider success
        # from an answer rejected by the execution/evidence backstop.
        if override.receipt.guard_failure:
            status = "error"
            error = override.receipt.failure_reason or "guarded_response"
    await finish(status, override.receipt.usage(), error)


async def _master_skill(brain, text, handle, override, mirror, bridge, ref, *, reuse=False) -> str:
    from jarvis.agent_chat.browser_model import browser_model_for
    from jarvis.agent_chat.folder_tools import FolderTool
    from jarvis.core.config import get_jarvis_agent_secret, override_provider_secrets
    from jarvis.core.redact import redact_secrets
    from jarvis.forge import mastery
    from jarvis.forge.tool import ForgeMindTool
    from jarvis.skills.skill_context import try_get_skill_context
    from jarvis.society.browser.tool import lead_browser_tools
    from jarvis.voice.action_phrases import _extract_cu_proof

    context = try_get_skill_context()
    executor = getattr(brain, "_tool_executor", None)
    if context is None or executor is None or bridge is None:
        raise mastery.MasteryError("Skill mastery is unavailable while Jarvis is starting.")
    stance = handle.stance or "ask"
    if stance in ("plan", "read-only"):
        raise mastery.MasteryError("Switch out of read-only mode to approve a practical exercise.")
    task = text if reuse else mastery.request(text)
    if not task:
        task = next(
            (message.content for message in reversed(brain_history_from_events(handle.history))
             if message.role == "user" and isinstance(message.content, str)
             and mastery.request(message.content) is None),
            "",
        )
    task = redact_secrets(task)
    if not task:
        raise mastery.MasteryError(
            "Describe the skill to master or complete a task in this chat first."
        )
    prior = mastery.mastery_match(task)
    available = brain._apply_turn_override_tools(None, override)
    forge = available.get("forge_mind")
    tools = mastery.exercise_tools(available)
    live_session = mastery.requires_live_session(task)
    browser_search = (
        not live_session and isinstance(forge, ForgeMindTool) and "society_browser" in tools
    )
    if live_session and "computer_use" not in tools:
        raise mastery.MasteryError("The authorized live desktop agent is unavailable in this chat.")
    if not reuse and "search_web" not in tools:
        raise mastery.MasteryError("Current documentation search is unavailable in this chat.")
    expires = time.monotonic() + APPROVAL_TIMEOUT_S

    def check_permission() -> None:
        mastery.check_cancel(handle.cancel)
        if handle.control_service is not None:
            current = handle.control_service.store.get_session(handle.session.session_id)
            if current is None or current.permission_mode != handle.session.permission_mode:
                raise mastery.MasteryError("The exercise authorization was revoked.")
        if time.monotonic() >= expires:
            raise mastery.MasteryError("The exercise authorization expired.")

    for name, observer in lead_browser_tools(handle.session, read_only=True).items():
        if name in tools:
            tools[f"{name}_observe"] = observer

    async def execute_call(call: mastery.Call, check: mastery.Call, rationale: str):
        check_permission()
        snapshot = dict(override.tool_context)
        if call is check and not isinstance(tools[call.tool], FolderTool):
            snapshot["chat_read_only"] = True
        return await executor.execute(
            tools[call.tool], dict(call.args), config_snapshot=snapshot,
            trace_id=handle.trace_id, rationale=rationale,
        )

    def check_active_skill() -> None:
        current_skill = mastery.mastery_match(task)
        if current_skill is None or current_skill.body_hash != prior.body_hash:
            raise mastery.MasteryError("The mastered skill changed during reuse.")

    async def run_desktop(practice: mastery.DesktopPractice, *, scoped: bool = False) -> None:
        args = practice.tool_args()

        def guard() -> None:
            check_permission()
            if reuse:
                check_active_skill()

        def scoped_approval(name: str, proposed: dict[str, Any]) -> bool:
            try:
                guard()
            except (mastery.MasteryError, asyncio.CancelledError):
                return False
            return name == "computer_use" and proposed == args

        guard()
        if scoped:
            bridge.arm(ref, ChatGrant(
                handle.session.session_id, handle.turn_id, "always_ask", set(),
                handle.request_approval, call_id_for=mirror.open_call_id,
                scoped_approval=scoped_approval,
            ))
        pending = asyncio.create_task(executor.execute(
            tools["computer_use"], args, trace_id=handle.trace_id, user_utterance=task,
            rationale="Verify the practical exercise in the existing desktop session",
            config_snapshot={**override.tool_context, "forge_mastery_desktop": True},
        ))
        try:
            while not pending.done():
                await asyncio.wait((pending,), timeout=0.25)
                guard()
            result = pending.result()
        finally:
            if not pending.done():
                pending.cancel()
                await asyncio.gather(pending, return_exceptions=True)
        stdout = result.output.get("stdout") if isinstance(result.output, dict) else None
        if not result.success or not isinstance(stdout, str) or not _extract_cu_proof(stdout):
            raise mastery.MasteryError("The desktop agent did not verify the visible outcome.")

    async def run_browser(search: mastery.BrowserSearch, *, scoped: bool = False) -> None:
        def guard() -> None:
            check_permission()
            if reuse:
                check_active_skill()

        guard()

        async def approve_action(*args):
            guard()
            if scoped:
                return "allow"
            return await handle.request_approval(*args)

        def scoped_approval(name: str, args: dict[str, Any]) -> bool:
            try:
                check_permission()
            except (mastery.MasteryError, asyncio.CancelledError):
                return False
            if name == "forge_mind":
                return not args
            if name != "society_browser_action":
                return False
            return args in (
                {"action": {verb: {}}}
                for verb in ("navigate", "input_text", "click", "send_keys", "done")
            )

        if scoped:
            bridge.arm(ref, ChatGrant(
                handle.session.session_id, handle.turn_id, "always_ask", set(),
                handle.request_approval, call_id_for=mirror.open_call_id,
                scoped_approval=scoped_approval,
            ))
        result = await executor.execute(
            forge, {}, trace_id=handle.trace_id, rationale="Verify a public browser search",
            config_snapshot={
                **override.tool_context,
                "forge_interactive": True,
                "forge_browser_model": (override.provider, override.model),
                "forge_mastery_search": (search.url, search.query),
                "forge_search_guard": guard,
                "forge_ask": approve_action,
            },
        )
        guard()
        if not result.success:
            raise mastery.MasteryError(f"Browser search verification failed ({result.error}).")

    check_permission()
    secret = _agent_secret(get_jarvis_agent_secret, handle.session.provider)
    with override_provider_secrets({handle.session.provider: secret} if secret else {}):
        model = browser_model_for(brain._config, override.provider, override.model or "")
        check_permission()
        if reuse:
            if prior is None:
                raise mastery.MasteryError("The mastered skill is no longer active.")
            instructions = context.runner.render_instructions(
                prior, args={"utterance": task, "_trigger": "chat"},
            )
            reply = await mastery.complete(
                model, mastery.reuse_prompt(task, instructions, tools), max_tokens=3000,
            )
            check_permission()
            calls = mastery.parse_proposal(
                reply, tools, reuse=True,
                browser_search=browser_search and mastery.BROWSER_SEARCH_GUIDANCE in instructions,
            )
            if calls is None:
                return instructions
            if mastery.DESKTOP_GUIDANCE in instructions or live_session:
                if not isinstance(calls, mastery.DesktopPractice) or not calls.matches_skill(
                    instructions
                ):
                    raise mastery.MasteryError("The requested task is outside the verified app.")
                await run_desktop(calls)
                return f"Applied /{prior.name} and verified the visible result."
            if isinstance(calls, mastery.DesktopPractice):
                raise mastery.MasteryError("This skill has no verified desktop method.")
            if isinstance(calls, mastery.BrowserSearch):
                if not mastery.matches_browser_scope(instructions, calls, task):
                    raise mastery.MasteryError(
                        "The requested search is outside the verified scope."
                    )
                await run_browser(calls)
                return f"Applied /{prior.name} and verified the displayed search results."
            await asyncio.to_thread(
                mastery.bind_folder_scope, calls, tools, Path(override.tool_context["cwd"]),
                reuse=True,
            )

            async def run_saved(call):
                check_permission()
                check_active_skill()
                return await execute_call(call, calls.check, "Apply verified skill " + prior.name)

            await mastery.exercise(calls, run_saved, check_permission, reuse=True)
            return f"Applied /{prior.name} to the current inputs and verified the result."
        query = await mastery.research_topic(model, task)
        check_permission()
        research = await executor.execute(
            tools["search_web"],
            {"query": query, "queries": [" ".join(query.split()[:4])], "max_results": 5},
            config_snapshot={**override.tool_context, "chat_read_only": True},
            trace_id=handle.trace_id, rationale="Research public documentation for skill mastery",
        )
        references = mastery.search_references(research)
        check_permission()
        reply = await mastery.complete(
            model, mastery.planning_prompt(
                task, tools, prior, references, browser_search=browser_search,
            ), max_tokens=4000,
        )
    check_permission()
    proposal = mastery.parse_proposal(reply, tools, browser_search=browser_search)
    assert isinstance(proposal, mastery.Proposal)
    if not any(row["url"] in proposal.knowledge for row in references):
        raise mastery.MasteryError("The proposed guidance did not cite the research results.")
    if prior is None:
        prior = next((s for s in context.registry.list_active() if
                      s.name == proposal.name and mastery.is_mastered(s)), None)
        if prior is None and any(s.name == proposal.name for s in context.registry.list()):
            raise mastery.MasteryError("That skill name is already in use.")

    practice = proposal.practice
    if live_session and not isinstance(practice, mastery.DesktopPractice):
        raise mastery.MasteryError("This exercise requires the existing desktop session.")
    if isinstance(practice, mastery.Exercise):
        await asyncio.to_thread(
            mastery.bind_folder_scope, practice, tools, Path(override.tool_context["cwd"]),
        )

    async with asyncio.timeout(max(0, expires - time.monotonic())):
        decision = await handle.request_approval(
            handle.turn_id, "forge_mastery",
            {**proposal.approval_input(), **({"update_skill": prior.name} if prior else {})},
            "Run the scoped exercise and save the displayed guidance if verification passes.",
        )
    check_permission()
    if decision not in {"allow", "allow_always"}:
        return "Skill mastery declined. The exercise did not run."

    if isinstance(practice, mastery.BrowserSearch):
        await run_browser(practice, scoped=True)
        name = await mastery.save(proposal, context.registry, handle.bus, prior)
        return f"Verified and activated /{name}. The skill is available for future matching tasks."
    if isinstance(practice, mastery.DesktopPractice):
        await run_desktop(practice, scoped=True)
        name = await mastery.save(proposal, context.registry, handle.bus, prior)
        return f"Verified and activated /{name}. The skill is available for future matching tasks."

    # Only the exact new-file operation can consume the scoped approval.
    # Other confirmation gates, including plausibility, retain their cards.
    active: mastery.Call | None = None

    def scoped(name: str, args: dict[str, Any]) -> bool:
        try:
            check_permission()
        except (mastery.MasteryError, asyncio.CancelledError):
            return False
        return (
            active is not None and name == active.tool == "Write"
            and isinstance(tools[name], FolderTool) and args == active.args
            and args.get("create_only") is True and not handle.cancel.is_set()
        )

    bridge.arm(ref, ChatGrant(
        handle.session.session_id, handle.turn_id, "always_ask", set(),
        handle.request_approval, call_id_for=mirror.open_call_id, scoped_approval=scoped,
    ))

    async def execute(call: mastery.Call):
        nonlocal active
        active = call
        try:
            return await execute_call(call, practice.check, "FORGE mastery exercise")
        finally:
            active = None

    await mastery.exercise(practice, execute, check_permission)
    name = await mastery.save(proposal, context.registry, handle.bus, prior)
    return f"Verified and activated /{name}. The skill is available for future matching tasks."


async def _generate(
    brain: Any,
    text: str,
    handle: TurnHandle,
    override: TurnOverride,
    feed: Callable[[str], None],
) -> str:
    """The one ``generate`` call, under the chat's credentials, cancellable.

    ``allow_voice_confirm=False`` is load-bearing: the executor must never
    park a chat's consequential action in the manager-wide voice slot that
    the NEXT spoken turn would consume. The chat's stance and card decide
    (``approval_surface`` in the tool context).
    """
    from jarvis.core.config import get_jarvis_agent_secret, override_provider_secrets

    session = handle.session
    history = brain_history_from_events(handle.history)
    output_language = getattr(handle, "output_language", "")
    if session.surface == "society":
        from jarvis.society.reply_preference import resolve_agent_reply_language

        output_language = await resolve_agent_reply_language(
            session.session_id, text, output_language
        )
    kwargs: dict[str, Any] = {
        "use_history": False,
        "history_override": history,
        "conversation_id": session.session_id,
        "source_layer": SOURCE_LAYER,
        "trace_id": handle.trace_id,
        "allow_voice_confirm": False,
        "emit_tool_ack": False,
        "publish_response": False,
        "text_consumer": feed,
        "turn_override": override,
    }
    if output_language:
        kwargs["force_output_language"] = output_language
    secret = (
        None
        if override.tool_context.get("forge_interactive")
        else _agent_secret(get_jarvis_agent_secret, session.provider)
    )
    overrides = {session.provider: secret} if secret else {}
    # The task inherits the credential override through its context copy, so
    # the scoped brain instance resolves the Agents-tab key on first use.
    with override_provider_secrets(overrides):
        if session.surface == "society" and not override.tool_context.get("forge_interactive"):
            from jarvis.society.conversation import prepare_history
            from jarvis.society.runtime import current_runtime

            runtime = current_runtime()
            getter = getattr(brain, "_get_brain", None)
            if runtime is not None and callable(getter):
                provider = getter(
                    session.provider,
                    session.model or None,
                    scope=f"society-compaction:{session.session_id}",
                )
                kwargs["history_override"] = await prepare_history(
                    runtime,
                    session,
                    handle.history,
                    text,
                    provider,
                    prompt_chars=len(override.system_extra) + 16000,
                )
        task: asyncio.Task[str] = asyncio.create_task(
            brain.generate(text, **kwargs), name=f"agent-chat-brain-{handle.turn_id[:8]}"
        )
    return await _wait_for_turn_task(task, handle)


async def _wait_for_turn_task(task: asyncio.Task[str], handle: TurnHandle) -> str:
    waiter = asyncio.create_task(handle.cancel.wait())
    try:
        done, _ = await asyncio.wait({task, waiter}, return_when=asyncio.FIRST_COMPLETED)
        if task in done:
            return await task
        task.cancel()
        try:
            await task
        except (asyncio.CancelledError, Exception) as exc:  # noqa: BLE001 — the turn is over
            log.debug("brain runner: cancelled turn ended with %r", exc)
        raise asyncio.CancelledError
    finally:
        waiter.cancel()
        if not task.done():
            task.cancel()


def _agent_secret(resolver: Callable[[str], str | None], provider: str) -> str | None:
    try:
        return resolver(provider)
    except Exception:  # noqa: BLE001 — no dedicated key: the brain resolves as usual
        log.debug("brain runner: no agent-tier secret for %s", provider, exc_info=True)
        return None


#: A leading ``/slug`` — the composer's slash pick — and whatever follows it.
_EXPLICIT_SKILL_RE = re.compile(r"^\s*/([A-Za-z0-9][\w.:-]*)(?:\s+|$)(.*)$", re.DOTALL)


def explicit_skill(text: str) -> tuple[str, str] | None:
    """``("slug", "the rest")`` when the message opens with ``/slug``, else None.

    The composer's ``/`` typeahead lists the registry's active skills by
    slug; picking one puts ``/slug`` at the head of the sentence. That pick
    is an order, not a hint, so it is honoured before any trigger-phrase
    matching — the same precedence a CLI gives its slash commands.
    """
    m = _EXPLICIT_SKILL_RE.match(text or "")
    if m is None:
        return None
    return m.group(1), m.group(2).strip()


def _note_skill_trigger(brain: Any, text: str) -> None:
    """The desktop text path's pre-brain hook, verbatim in effect: a skill
    whose trigger matches is noted on the brain, which injects its
    instructions into this turn (desktop_app._on_user_message). A leading
    ``/slug`` names the skill outright and wins over the trigger matcher."""
    try:
        from jarvis.skills.skill_context import try_get_skill_context
        from jarvis.skills.trigger_matcher import TriggerMatcher

        skill_ctx = try_get_skill_context()
        note = getattr(brain, "note_skill_trigger", None)
        if skill_ctx is None or not callable(note):
            return
        explicit = explicit_skill(text)
        if explicit is not None:
            slug, rest = explicit
            try:
                skill = skill_ctx.registry.resolve(slug)
            except KeyError:  # not a skill's name: the trigger matcher gets its turn
                log.debug("brain runner: /%s names no active skill", slug)
            else:
                note(skill.name, content=rest, source="chat")
                return
        match_result = TriggerMatcher(skill_ctx.registry).match_voice_with_match(text, lang="auto")
        if match_result is None:
            return
        matched, regex_match = match_result
        content = ""
        for group in reversed(regex_match.groups()):
            if group and group.strip():
                content = group.strip()
                break
        note(matched.name, content=content, source="chat")
    except Exception:  # noqa: BLE001 — the hook is defensive; a crash must never cost the turn
        log.debug("brain runner: skill pre-hook skipped", exc_info=True)


__all__ = [
    "APPROVAL_TIMEOUT_S",
    "MAX_TURNS",
    "RUNNER",
    "SOURCE_LAYER",
    "brain_history_from_events",
    "brain_manager",
    "build_override",
    "kit_payload",
    "run_brain_turn",
]
