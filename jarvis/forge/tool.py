"""Jarvis tool and command vocabulary for the local FORGE fixture."""

from __future__ import annotations

import asyncio
import hashlib
import logging
import re
from dataclasses import asdict
from typing import Any

from jarvis.control.cancel import CancelScope, CancelToken, get_kill_switch
from jarvis.core.protocols import ExecutionContext, ToolResult

from .engine import Evidence, Limits, Procedure, Refused, Verdict, execute
from .store import Store
from .workflow import Marker, WorkflowProcedure, run_workflow

logger = logging.getLogger(__name__)

NAMESPACE = re.compile(
    r"^forge\s+(?:learn|list|inspect|validate|promote|replay|stop)\b",
    re.ASCII | re.IGNORECASE,
)
COMMAND = re.compile(
    r"forge (?:list|stop|learn (?:workflow )?(?:alpha|beta)|(?:inspect|promote) [1-9][0-9]{0,8}"
    r"|(?:validate|replay) [1-9][0-9]{0,8} (?:alpha|beta))[.!]?",
    re.ASCII | re.IGNORECASE,
)


def command(text: str) -> tuple[str, ...] | None:
    text = text.strip()
    if not COMMAND.fullmatch(text):
        return None
    return tuple(text.rstrip(".!").lower().split()[1:])


def render_result(result: ToolResult) -> str:
    if not result.success:
        return f"FORGE stopped ({result.error})."
    if isinstance(result.output, str):
        return result.output
    if isinstance(result.output, list):
        if not result.output:
            return "No FORGE procedures have been saved."
        items = [f"version {row['version']} is {row['state']}" for row in result.output[:10]]
        return "FORGE procedures. " + ". ".join(items) + "."
    version = result.output["version"]
    state = result.output["state"]
    reply = f"FORGE procedure {version} is {state}."
    if "procedure" in result.output:
        spec = result.output["procedure"]
        if spec.get("schema_version") == 2:
            actions = ", ".join(step["action"].replace("_", " ") for step in spec["steps"])
            return reply + f" Verified workflow steps are {actions}."
        return reply + " It sets the blank fixture field and reads back the exact marker."
    return reply + " Verification succeeded."


class ForgeMindTool:
    name = "forge_mind"
    risk_tier = "monitor"
    description = (
        "Run the exact spoken or typed FORGE command. Commands: forge list, forge stop, "
        "forge learn alpha, forge inspect VERSION, forge validate VERSION beta, "
        "forge promote VERSION, forge replay VERSION alpha, forge learn workflow alpha. "
        "The actual user utterance supplies the full command."
    )
    schema: dict[str, Any] = {"type": "object", "properties": {}, "additionalProperties": False}
    suppress_response = True

    def __init__(self, store: Store, open_target: Any, limits: Limits) -> None:
        self._store = store
        self._open_target = open_target
        self._limits = limits
        self._active: CancelToken | None = None

    @classmethod
    def build(cls, config: Any, executor: Any) -> ForgeMindTool:
        from jarvis.core.instance import current_instance
        from jarvis.core.paths import user_data_dir

        from .desktop import fixture_factory

        return cls(
            Store(user_data_dir() / current_instance().data_dir_name / "forge.sqlite"),
            fixture_factory(executor, memory_mb=config.memory_mb),
            Limits(config.attempts, config.run_seconds, config.action_budget),
        )

    async def execute(self, args: dict[str, Any], ctx: ExecutionContext) -> ToolResult:
        words = command(ctx.user_utterance)
        if args or words is None:
            return ToolResult(False, None, "exact_forge_command_required")
        if not (
            ctx.config.get("forge_interactive") is True
            or ctx.config.get("tool_origin") == "realtime"
        ):
            return ToolResult(False, None, "interactive_forge_command_required")
        mode, *params = words
        if mode == "stop":
            if self._active is not None:
                self._active.cancel("user_requested")
                return ToolResult(True, "FORGE stop requested.")
            return ToolResult(True, "No FORGE task is active.")
        if ctx.config.get("chat_read_only") and mode not in {"list", "inspect"}:
            return ToolResult(False, None, "read_only")
        try:
            if mode == "list":
                return ToolResult(True, await asyncio.to_thread(self._store.list))
            version = None if mode == "learn" else int(params[0])
            procedure, state = Procedure(), ""
            if version is not None:
                procedure, state = await asyncio.to_thread(self._store.get, version)
            if mode == "inspect":
                assert version is not None
                return ToolResult(
                    True,
                    {
                        "version": version,
                        "state": state,
                        "procedure": asdict(procedure),
                    },
                )
            if mode == "promote":
                assert version is not None
                # The full command comes from the current user turn, never
                # from tool args, screen text, retrieved notes, or a model.
                await asyncio.to_thread(self._store.promote, version, authorized=True)
                return ToolResult(True, f"FORGE procedure {version} promoted.")
            if mode == "replay" and state != "promoted":
                raise Refused("promoted_version_required")
            if mode == "validate" and state != "draft":
                receipt = await asyncio.to_thread(
                    self._store.receipt,
                    self._run_id(ctx, mode, version, params[-1]),
                )
                if receipt is None:
                    raise Refused("draft_required")
                return await self._result(*receipt)
            if self._active is not None:
                raise Refused("experiment_already_active")
            explore = mode == "learn" and params[0] == "workflow"
            return await self._run(mode, version, procedure, params[-1], ctx, explore=explore)
        except Refused as exc:
            logger.debug("FORGE command refused; returning a controlled result")
            return ToolResult(False, None, str(exc))

    async def _run(
        self,
        mode: str,
        version: int | None,
        procedure: Procedure | WorkflowProcedure,
        marker: str,
        ctx: ExecutionContext,
        *,
        explore: bool,
    ) -> ToolResult:
        async with CancelScope(get_kill_switch(), holder="forge") as token:
            self._active = token
            try:
                return await self._record_run(
                    mode, version, procedure, marker, ctx, token, explore=explore
                )
            finally:
                self._active = None

    @staticmethod
    def _run_id(ctx: ExecutionContext, mode: str, version: int | None, marker: str) -> str:
        return hashlib.sha256(f"{ctx.trace_id}:{mode}:{version}:{marker}".encode()).hexdigest()

    async def _record_run(
        self,
        mode: str,
        version: int | None,
        procedure: Procedure | WorkflowProcedure,
        marker: str,
        ctx: ExecutionContext,
        token: CancelToken,
        *,
        explore: bool,
    ) -> ToolResult:
        run_id = self._run_id(ctx, mode, version, f"workflow:{marker}" if explore else marker)
        existing = await asyncio.to_thread(self._store.claim, run_id, mode, version)
        if existing is not None:
            return await self._result(*existing)

        trace: list[Evidence] = []
        outcome = "interrupted"
        learned = None
        try:
            try:
                rejected = (
                    await asyncio.to_thread(self._store.rejected_actions, Marker(marker))
                    if explore
                    else frozenset()
                )
                async with self._open_target(run_id, ctx) as target:
                    if explore or isinstance(procedure, WorkflowProcedure):
                        learned = await run_workflow(
                            target,
                            Marker(marker),
                            token,
                            trace,
                            self._limits,
                            rejected=rejected,
                            procedure=procedure
                            if isinstance(procedure, WorkflowProcedure)
                            else None,
                        )
                    else:
                        await execute(target, procedure, marker, token, trace, self._limits)
                outcome = str(trace[-1].verdict) if trace else "no_evidence"
                if trace and trace[-1].reason == "already_matched" and mode != "replay":
                    outcome = "no_new_evidence"
            except Refused as exc:
                logger.debug("FORGE run stopped by a target, policy, or cancellation check")
                outcome = str(exc)
            except TimeoutError:
                logger.warning("FORGE run timed out; recording an unresolved outcome")
                outcome = "budget_exceeded"
            if token.is_cancelled():
                outcome = "cancelled"
        finally:
            version, outcome = await asyncio.to_thread(
                self._store.record,
                run_id,
                mode,
                version,
                outcome,
                trace,
                learned,
            )
        return await self._result(version, outcome)

    async def _result(self, version: int | None, outcome: str) -> ToolResult:
        state = None
        if version is not None:
            _, state = await asyncio.to_thread(self._store.get, version)
        success = outcome == Verdict.SUCCESS
        return ToolResult(
            success,
            {"outcome": outcome, "version": version, "state": state},
            None if success else outcome,
        )
