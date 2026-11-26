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

from .browser import EXPECTED, BrowserProcedure, Search, run_browser
from .engine import Evidence, Limits, Procedure, Refused, Verdict, execute
from .store import Store
from .workflow import Goal, Marker, WorkflowProcedure, run_workflow

logger = logging.getLogger(__name__)

NAMESPACE = re.compile(
    r"^forge\s+(?:learn|list|inspect|validate|promote|replay|stop|improve|rollback)\b",
    re.ASCII | re.IGNORECASE,
)
JOB_PARAMETERS = r"(?:engineer|designer) (?:remote|onsite)"
COMMAND = re.compile(
    r"forge (?:list|stop|learn (?:(?:workflow|receipt) )?(?:alpha|beta)"
    rf"|learn jobs {JOB_PARAMETERS}"
    r"|(?:inspect|promote|rollback) [1-9][0-9]{0,8}"
    rf"|(?:validate|replay|improve) [1-9][0-9]{{0,8}} (?:alpha|beta|{JOB_PARAMETERS}))[.!]?",
    re.ASCII | re.IGNORECASE,
)


def command(text: str) -> tuple[str, ...] | None:
    text = text.strip()
    if not COMMAND.fullmatch(text):
        return None
    return tuple(text.rstrip(".!").lower().split()[1:])


def render_result(result: ToolResult) -> str:
    if result.error == "procedure_already_saved":
        return f"FORGE procedure {result.output['version']} already contains these steps."
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
        if spec.get("schema_version") == 4:
            failures = result.output["recent_failures"]
            recent = f" Recent failures: {', '.join(failures)}." if failures else ""
            return (
                reply + f" It searches fixture revision {spec['fixture_version']} by job role "
                "and work location, then verifies the displayed results." + recent
            )
        if spec.get("schema_version") in (2, 3):
            actions = ", ".join(step["action"].replace("_", " ") for step in spec["steps"])
            return reply + f" Verified workflow steps are {actions}."
        return reply + " It sets the blank fixture field and reads back the exact marker."
    if "jobs" in result.output:
        return reply + f" Verified fixture results are {', '.join(result.output['jobs'])}."
    return reply + " Verification succeeded."


class ForgeMindTool:
    name = "forge_mind"
    risk_tier = "monitor"
    description = (
        "Run the exact spoken or typed FORGE command. Commands: forge list, forge stop, "
        "forge learn alpha, forge inspect VERSION, forge validate VERSION beta, "
        "forge promote VERSION, forge replay VERSION alpha, forge learn workflow alpha. "
        "For the local job fixture use forge learn jobs engineer remote, or give a "
        "role and location after a version when validating, replaying, or improving. "
        "Use forge improve VERSION alpha to explore a new draft, or forge rollback VERSION "
        "to restore the previous validated version. The user utterance supplies the command."
    )
    schema: dict[str, Any] = {"type": "object", "properties": {}, "additionalProperties": False}
    suppress_response = True

    def __init__(
        self,
        store: Store,
        open_target: Any,
        limits: Limits,
        *,
        available_goals: frozenset[Goal] = frozenset(Goal),
    ) -> None:
        self._store = store
        self._open_target = open_target
        self._limits = limits
        self._available_goals = available_goals
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
            available_goals=frozenset({Goal.SUBMIT_MARKER}),
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
                details = {}
                if isinstance(procedure, BrowserProcedure):
                    details["recent_failures"] = await asyncio.to_thread(
                        self._store.browser_failures, version
                    )
                return ToolResult(
                    True,
                    {
                        "version": version,
                        "state": state,
                        "procedure": asdict(procedure),
                        "parent_version": await asyncio.to_thread(self._store.parent, version),
                        **details,
                    },
                )
            if mode == "promote":
                assert version is not None
                # The full command comes from the current user turn, never
                # from tool args, screen text, retrieved notes, or a model.
                await asyncio.to_thread(self._store.promote, version, authorized=True)
                return ToolResult(True, f"FORGE procedure {version} promoted.")
            if mode == "rollback":
                assert version is not None
                restored = await asyncio.to_thread(self._store.rollback, version, authorized=True)
                return ToolResult(True, f"FORGE procedure {restored} restored.")
            if isinstance(procedure, BrowserProcedure) or (mode == "learn" and params[0] == "jobs"):
                if len(params) != 3:
                    raise Refused("job_search_parameters_required")
                search = Search.read(*params[1:])
                return await self._run_browser(mode, version, procedure, state, search, ctx)
            if len(params) == 3:
                raise Refused("marker_required")
            explore = None
            if mode == "learn" and params[0] == "workflow":
                explore = Goal.SUBMIT_MARKER
            if mode == "learn" and params[0] == "receipt":
                explore = Goal.ISSUE_RECEIPT
            if mode == "improve":
                if not isinstance(procedure, WorkflowProcedure):
                    raise Refused("workflow_required")
                explore = procedure.goal
            goal = explore or (procedure.goal if isinstance(procedure, WorkflowProcedure) else None)
            marker = params[-1]
            run_id = self._run_id(ctx, mode, version, self._run_marker(marker, explore))
            receipt = await asyncio.to_thread(self._store.receipt, run_id)
            if receipt is not None:
                return await self._result(*receipt)
            if goal is not None and goal not in self._available_goals:
                raise Refused("fixture_unavailable")
            if mode in {"replay", "improve"} and state != "promoted":
                raise Refused("promoted_version_required")
            if mode == "validate" and state != "draft":
                raise Refused("draft_required")
            if self._active is not None:
                raise Refused("experiment_already_active")
            return await self._run(mode, version, procedure, marker, ctx, explore=explore)
        except Refused as exc:
            logger.debug("FORGE command refused; returning a controlled result")
            return ToolResult(False, None, str(exc))

    async def _run_browser(
        self,
        mode: str,
        version: int | None,
        procedure: Procedure | WorkflowProcedure | BrowserProcedure,
        state: str,
        search: Search,
        ctx: ExecutionContext,
    ) -> ToolResult:
        run_id = self._run_id(ctx, mode, version, f"jobs:{search.marker}")
        receipt = await asyncio.to_thread(self._store.receipt, run_id)
        if receipt is not None:
            return await self._result(*receipt, search=search)
        if mode in {"replay", "improve"} and state != "promoted":
            raise Refused("promoted_version_required")
        if mode == "validate" and state != "draft":
            raise Refused("draft_required")
        if self._limits.actions < 3:
            raise Refused("action_budget_exceeded")
        if self._active is not None:
            raise Refused("experiment_already_active")

        stored_mode = "learn" if mode == "improve" else mode
        stored_version = None if mode == "improve" else version
        parent = version if mode == "improve" else None
        saved = procedure if mode in {"validate", "replay"} else None
        assert saved is None or isinstance(saved, BrowserProcedure)
        async with CancelScope(get_kill_switch(), holder="forge") as token:
            self._active = token
            try:
                old = await asyncio.to_thread(
                    self._store.claim, run_id, stored_mode, stored_version
                )
                if old is not None:
                    return await self._result(*old, search=search)
                trace: list[Evidence] = []
                learned = None
                outcome = "interrupted"
                try:
                    async with asyncio.timeout(self._limits.seconds):
                        learned = await run_browser(search, token, trace, saved)
                    outcome = (
                        "no_procedure_change"
                        if parent and learned == procedure
                        else Verdict.SUCCESS
                    )
                except TimeoutError:
                    logger.warning("FORGE browser run exceeded its deadline")
                    outcome = "budget_exceeded"
                except Refused as exc:
                    logger.debug("FORGE browser run refused")
                    outcome = str(exc)
                finally:
                    if token.is_cancelled():
                        outcome = "cancelled"
                    result = await asyncio.to_thread(
                        self._store.record,
                        run_id,
                        stored_mode,
                        stored_version,
                        outcome,
                        trace,
                        learned,
                        parent,
                    )
                return await self._result(*result, search=search)
            finally:
                self._active = None

    async def _run(
        self,
        mode: str,
        version: int | None,
        procedure: Procedure | WorkflowProcedure,
        marker: str,
        ctx: ExecutionContext,
        *,
        explore: Goal | None,
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

    @staticmethod
    def _run_marker(marker: str, explore: Goal | None) -> str:
        # Keep existing v2 discovery deliveries keyed exactly as before.
        if explore == Goal.SUBMIT_MARKER:
            return f"workflow:{marker}"
        if explore == Goal.ISSUE_RECEIPT:
            return f"receipt:{marker}"
        return marker

    async def _record_run(
        self,
        mode: str,
        version: int | None,
        procedure: Procedure | WorkflowProcedure,
        marker: str,
        ctx: ExecutionContext,
        token: CancelToken,
        *,
        explore: Goal | None,
    ) -> ToolResult:
        run_id = self._run_id(ctx, mode, version, self._run_marker(marker, explore))
        parent = version if mode == "improve" else None
        stored_mode = "learn" if parent is not None else mode
        stored_version = None if parent is not None else version
        existing = await asyncio.to_thread(self._store.claim, run_id, stored_mode, stored_version)
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
                            goal=explore or procedure.goal,
                            procedure=None if explore is not None else procedure,
                        )
                    else:
                        await execute(target, procedure, marker, token, trace, self._limits)
                outcome = str(trace[-1].verdict) if trace else "no_evidence"
                if (
                    parent is not None
                    and learned is not None
                    and isinstance(procedure, WorkflowProcedure)
                    and (learned.steps, learned.fixture_version)
                    == (procedure.steps, procedure.fixture_version)
                ):
                    outcome = "no_procedure_change"
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
                stored_mode,
                stored_version,
                outcome,
                trace,
                learned,
                parent,
            )
        return await self._result(version, outcome)

    async def _result(
        self, version: int | None, outcome: str, *, search: Search | None = None
    ) -> ToolResult:
        state = None
        if version is not None:
            _, state = await asyncio.to_thread(self._store.get, version)
        success = outcome == Verdict.SUCCESS
        output: dict[str, Any] = {"outcome": outcome, "version": version, "state": state}
        if success and search is not None:
            output["jobs"] = EXPECTED[search.role, search.location].copy()
        return ToolResult(success, output, None if success else outcome)
