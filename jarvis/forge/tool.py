"""Jarvis tool and command vocabulary for FORGE procedures."""

from __future__ import annotations

import asyncio
import hashlib
import logging
import re
import time
from dataclasses import asdict, replace
from typing import Any

from jarvis.control.cancel import CancelScope, CancelToken, get_kill_switch
from jarvis.core.protocols import ExecutionContext, ToolResult

from .browser import EXPECTED, BrowserProcedure, Search, run_browser
from .engine import Evidence, Limits, Procedure, Refused, State, Verdict, execute
from .live_navigation import LIVE_STEPS, DynamicSearch, LiveSearch, search_origin
from .store import Store
from .workflow import Goal, Marker, WorkflowProcedure, run_workflow

logger = logging.getLogger(__name__)

NAMESPACE = re.compile(
    r"^(?:forge\s+(?:learn|list|inspect|validate|promote|replay|stop|improve|rollback)\b"
    r"|(?:learn to search|use what you learned to search) https?://)",
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
WEB_COMMAND = re.compile(
    r"forge (?:(learn|replay) (web|web-legacy)|(validate|replay) ([1-9][0-9]{0,8}) (web)) "
    r"(\S{1,2048}) ([A-Za-z ]{2,99})",
    re.ASCII | re.IGNORECASE,
)


SEARCH_OBJECTIVE = re.compile(
    r"(learn to search|use what you learned to search) (https?://\S{1,2048}) "
    r'for "([A-Za-z ]{2,99})"[.!]?',
    re.ASCII | re.IGNORECASE,
)


def command(text: str) -> tuple[str, ...] | None:
    text = text.strip()
    objective = SEARCH_OBJECTIVE.fullmatch(text)
    if objective:
        intent, url, query = objective.groups()
        mode = "learn" if intent.lower() == "learn to search" else "replay"
        return mode, "web", url, query
    web = WEB_COMMAND.fullmatch(text)
    if web:
        automatic, kind, mode, version, _, url, query = web.groups()
        if automatic and automatic.lower() == "replay" and kind.lower() != "web":
            return None
        return (
            (automatic.lower(), kind.lower(), url, query)
            if automatic
            else (mode.lower(), version, "web", url, query)
        )
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
        if spec.get("schema_version") in (5, 6):
            actions = ", ".join(step.split(":")[0].replace("_", " ") for step in spec["steps"])
            failures = result.output["recent_failures"]
            recent = f" Recent failures: {', '.join(failures)}." if failures else ""
            return reply + f" Verified public search steps are {actions}." + recent
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
        "In Agent Chat use forge learn web URL QUERY to learn a supported public search, "
        "then forge validate VERSION web URL QUERY with a different-length query. "
        "After promotion, use forge replay web URL QUERY or forge replay VERSION web URL QUERY. "
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
        web_seconds: float = 300,
        web_model_seconds: float = 60,
        web_approval_seconds: float = 75,
        web_replay_foreground: bool = False,
    ) -> None:
        self._store = store
        self._open_target = open_target
        self._limits = limits
        self._available_goals = available_goals
        self._web_seconds = web_seconds
        self._web_model_seconds = web_model_seconds
        self._web_approval_seconds = web_approval_seconds
        self._web_replay_foreground = web_replay_foreground
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
            web_seconds=config.web_run_seconds,
            web_model_seconds=config.web_model_seconds,
            web_approval_seconds=config.web_approval_seconds,
            web_replay_foreground=config.web_replay_foreground,
        )

    async def execute(self, args: dict[str, Any], ctx: ExecutionContext) -> ToolResult:
        mastery_search = ctx.config.get("forge_mastery_search")
        if not args and mastery_search is not None:
            if ctx.config.get("chat_read_only"):
                return ToolResult(False, None, "read_only")
            try:
                return await self._run_live_search(
                    "mastery", None, None, "", *mastery_search, ctx,
                )
            except Refused as exc:
                return ToolResult(False, None, str(exc))
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
            if mode == "replay" and params[0] == "web":
                origin = search_origin(*params[-2:])
                active = await asyncio.to_thread(self._store.promoted_search, origin)
                if active is None:
                    raise Refused("promoted_search_not_found")
                params.insert(0, str(active))
            version = None if mode == "learn" else int(params[0])
            procedure, state = Procedure(), ""
            if version is not None:
                procedure, state = await asyncio.to_thread(self._store.get, version)
            if mode == "inspect":
                assert version is not None
                details = {}
                if isinstance(procedure, (BrowserProcedure, LiveSearch, DynamicSearch)):
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
            web_kind = params[0 if mode == "learn" else 1]
            if web_kind in {"web", "web-legacy"}:
                if mode != "learn" and not isinstance(procedure, (LiveSearch, DynamicSearch)):
                    raise Refused("live_search_procedure_required")
                saved = procedure if isinstance(procedure, (LiveSearch, DynamicSearch)) else None
                result = await self._run_live_search(
                    mode,
                    version,
                    saved,
                    state,
                    *params[-2:],
                    ctx,
                    legacy_learning=web_kind == "web-legacy",
                )
                if (
                    mode == "replay"
                    and isinstance(saved, DynamicSearch)
                    and SEARCH_OBJECTIVE.fullmatch(ctx.user_utterance.strip())
                    and result.error == "search_page_drift"
                ):
                    return await self._run_live_search(
                        "learn", None, None, "", *params[-2:], ctx, parent=version
                    )
                return result
            if isinstance(procedure, (LiveSearch, DynamicSearch)):
                raise Refused("public_search_parameters_required")
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

    async def _run_live_search(
        self,
        mode: str,
        version: int | None,
        procedure: LiveSearch | DynamicSearch | None,
        state: str,
        url: str,
        query: str,
        ctx: ExecutionContext,
        *,
        legacy_learning: bool = False,
        parent: int | None = None,
    ) -> ToolResult:
        from jarvis.society.approvals import Verdict as ApprovalVerdict
        from jarvis.society.approvals import decide
        from jarvis.society.browser.bridge import execute_live
        from jarvis.society.runtime import current_runtime

        if ctx.config.get("forge_interactive") is not True or not str(
            ctx.config.get("approval_ref", "")
        ).startswith("agent-chat:"):
            raise Refused("agent_chat_required")
        origin = search_origin(url, query)
        mastery = mode == "mastery"
        if procedure is not None and procedure.origin != origin:
            raise Refused("browser_origin_mismatch")
        run_id = self._run_id(ctx, mode, version, "web")
        receipt = None if mastery else await asyncio.to_thread(self._store.receipt, run_id)
        if receipt is not None:
            return await self._result(*receipt)
        if mode == "validate" and state != "draft":
            raise Refused("draft_required")
        if mode == "replay" and state != "promoted":
            raise Refused("promoted_version_required")
        if self._limits.actions < (len(procedure.steps) if procedure else len(LIVE_STEPS)):
            raise Refused("action_budget_exceeded")
        if self._active is not None:
            raise Refused("experiment_already_active")
        runtime = current_runtime()
        if runtime is None or runtime.browser._python is not None:
            raise Refused("managed_browser_required")
        legacy = legacy_learning or isinstance(procedure, LiveSearch)
        if legacy:
            browser_options = {
                "public_search": {
                    "origin": origin,
                    "url": url,
                    "query": query,
                    "result_list": procedure.result_list
                    if isinstance(procedure, LiveSearch)
                    else None,
                }
            }
            args = {}
        else:
            expected = procedure.steps if procedure else ()
            browser_options = {
                "dynamic_search": {
                    "origin": origin,
                    "url": url,
                    "query": query,
                    "steps": expected,
                    "deterministic": mode in {"validate", "replay"},
                    "budget": self._limits.actions,
                    "foreground": mode != "replay" or self._web_replay_foreground,
                    "startup_seconds": 75,
                    "cleanup_seconds": 16,
                    "model_seconds": self._web_model_seconds,
                    "approval_seconds": self._web_approval_seconds,
                    "require_rendered_results": mastery,
                }
            }
            action_labels = {"input_text": "input", "verify_results": "done"}
            names = (step.split(":")[0] for step in expected if step != "navigate")
            sequence = ", ".join(action_labels.get(name, name) for name in names)
            args = {
                "task": (
                    f"Navigate to exactly {url} in the existing tab and use the site's public "
                    f"search interface to display results for {query!r}. Enter only that query. "
                    "Use indexed search controls or Enter. Use one action at a time. "
                    "Stop if login, consent, or CAPTCHA is required. "
                    "Finish successfully only when the rendered search results are visible."
                    + (
                        f" After automatic navigation, follow this action order: {sequence}."
                        if sequence
                        else ""
                    )
                ),
                "max_steps": self._limits.actions,
            }
        async with CancelScope(get_kill_switch(), holder="forge") as token:
            self._active = token
            try:
                caller = await runtime.roster.get(runtime.lead_id)
                if caller is None or str(caller.state) != "active":
                    raise Refused("browser_agent_unavailable")
                pick = ctx.config.get("forge_browser_model")
                if not legacy and pick and pick[0]:
                    selected = (pick[0], pick[1] or "")
                    if (caller.provider, caller.model) != selected:
                        caller = replace(caller, provider=selected[0], model=selected[1])
                old = None if mastery else await asyncio.to_thread(
                    self._store.claim, run_id, mode, version,
                )
                if old is not None:
                    return await self._result(*old)
                outcome = "interrupted"
                evidence: list[Evidence] = []
                started = time.monotonic()
                model_calls = 0
                auto_promote = False
                try:
                    async with asyncio.timeout(
                        self._limits.seconds if legacy else self._web_seconds
                    ):
                        async with asyncio.TaskGroup() as group:
                            run = group.create_task(
                                execute_live(
                                    runtime,
                                    caller,
                                    runtime.browser,
                                    args,
                                    ctx,
                                    **browser_options,
                                )
                            )
                            cancel = group.create_task(token.wait_until_cancelled())
                            await asyncio.wait((run, cancel), return_when=asyncio.FIRST_COMPLETED)
                            if token.is_cancelled():
                                run.cancel()
                            cancel.cancel()
                        if token.is_cancelled():
                            raise Refused("cancelled")
                        result = run.result()
                    model_calls = (result.output or {}).get("model_calls", 0)
                    if not result.success:
                        raise Refused(
                            result.error
                            if result.error
                            in {
                                "search_page_drift",
                                "browser_search_stopped",
                                "browser_origin_mismatch",
                                "browser_restricted_page",
                                "browser_target_unavailable",
                                "browser_target_restricted",
                                "browser_input_restricted",
                                "browser_action_restricted",
                                "browser_action_failed",
                                "browser_navigation_blocked",
                                "browser_site_unreachable",
                                "browser_search_model_failed",
                                "browser_search_model_unavailable",
                                "browser_model_timeout",
                                "browser_approval_timeout",
                                "browser_approval_denied",
                                "browser_action_timeout",
                                "browser_verification_timeout",
                                "browser_startup_timeout",
                                "browser_startup_failed",
                                "browser_cleanup_timeout",
                                "browser_cleanup_failed",
                                "managed_browser_required",
                                "search_results_unverified",
                                "action_budget_exceeded",
                            }
                            else "browser_search_unavailable"
                        )
                    raw = result.output
                    if not re.fullmatch(r"[0-9a-f]{32}", raw.get("target_id", "")):
                        raise Refused("missing_browser_evidence")
                    steps = tuple(raw["actions"])
                    observed = (
                        LiveSearch(origin, raw.get("result_list"), steps)
                        if legacy
                        else DynamicSearch(origin, steps)
                    )
                    observed.validate()
                    if procedure is not None and procedure != observed:
                        raise Refused("search_page_drift")
                    procedure = observed
                    # Length is enough to prove that validation exercised different input.
                    # No query, query hash, page path, or page text enters the store.
                    evidence = [
                        Evidence(
                            raw["target_id"],
                            step,
                            f"length:{len(query)}",
                            State.BLANK,
                            State.MATCH,
                            Verdict.SUCCESS,
                            "verified",
                            index,
                        )
                        for index, step in enumerate(observed.steps, 1)
                    ]
                    if mode == "validate" and isinstance(observed, DynamicSearch):
                        current = await runtime.roster.get(runtime.lead_id)
                        auto_promote = bool(
                            current is not None
                            and str(current.state) == "active"
                            and "core:forge:promote"
                            in current.approval_rules.get("always_allow", [])
                            and decide(current, "core:forge", "monitor", verb="promote")
                            is ApprovalVerdict.RUN
                        )
                    outcome = Verdict.SUCCESS
                except TimeoutError:
                    # The receipt below reports the action deadline to the user.
                    outcome = "budget_exceeded" if legacy else "browser_session_timeout"
                except asyncio.CancelledError:
                    outcome = "cancelled"
                    raise
                except Refused as exc:
                    # Persist the bounded refusal reason in the run receipt.
                    outcome = str(exc)
                finally:
                    if token.is_cancelled():
                        outcome = "cancelled"
                    if not legacy:
                        logger.info(
                            "FORGE mode=%s outcome=%s model_calls=%d elapsed_ms=%d",
                            mode,
                            outcome,
                            model_calls,
                            int((time.monotonic() - started) * 1000),
                        )
                    if not mastery:
                        saved = await asyncio.to_thread(
                            self._store.record,
                            run_id,
                            mode,
                            version,
                            outcome,
                            evidence,
                            procedure,
                            parent=parent,
                            auto_promote=auto_promote,
                        )
                if mastery:
                    if outcome != Verdict.SUCCESS:
                        return ToolResult(False, None, str(outcome))
                    return ToolResult(True, raw)
                return await self._result(*saved)
            finally:
                self._active = None

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
