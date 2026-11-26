"""Registered fixture goals and a bounded learner shared by all workflows."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from enum import StrEnum
from typing import Any, Protocol

from .engine import FIXTURE, Evidence, Lease, Limits, Refused, Target, Verdict

MAX_STEPS = 12


class Goal(StrEnum):
    SUBMIT_MARKER = "submit_marker"
    ISSUE_RECEIPT = "issue_receipt"


class Marker(StrEnum):
    ALPHA = "alpha"
    BETA = "beta"


class Phase(StrEnum):
    LOCKED = "locked"
    READY = "ready"
    FILLED = "filled"
    SUBMITTED = "submitted"
    EMPTY = "empty"
    STAGED = "staged"
    REVIEW = "review"
    ISSUED = "issued"


class Action(StrEnum):
    UNLOCK_A = "unlock_a"
    UNLOCK_B = "unlock_b"
    ENTER_MARKER = "enter_marker"
    SUBMIT = "submit"
    STAGE = "stage"
    REVIEW = "review"
    BACK = "back"
    QUICK_ISSUE = "quick_issue"
    ISSUE = "issue"


class ActionOutcome(StrEnum):
    ACCEPTED = "accepted"
    REJECTED = "rejected"
    STALE = "stale"
    UNKNOWN = "unknown"


StateKey = tuple[Phase, Marker | None, tuple[Action, ...], bool]
ActionContext = tuple[str, int, Marker, StateKey]
RejectionKey = tuple[ActionContext, Action]


@dataclass(frozen=True)
class Observation:
    phase: Phase
    revision: int
    marker: Marker | None
    controls: tuple[Action, ...]
    receipt: bool = False

    @classmethod
    def read(cls, raw: dict) -> Observation:
        """Read only the closed fixture vocabulary."""
        try:
            phase = Phase(raw["phase"])
            revision = raw["revision"]
            value = raw["marker"]
            marker = None if value is None else Marker(value)
            controls = raw["controls"]
            receipt = raw.get("receipt", False)
            if type(revision) is not int or not 0 <= revision <= MAX_STEPS:
                raise ValueError("invalid revision")
            if not isinstance(controls, list) or len(controls) > len(Action):
                raise ValueError("invalid controls")
            if type(receipt) is not bool:
                raise ValueError("invalid receipt")
            return cls(
                phase,
                revision,
                marker,
                tuple(action for action in Action if action.value in controls),
                receipt,
            )
        except (KeyError, ValueError, TypeError):
            raise Refused("invalid_observation") from None

    def state_key(self) -> StateKey:
        """Semantic state, independent of a guest's current revision counter."""
        return self.phase, self.marker, self.controls, self.receipt


@dataclass(frozen=True)
class Step:
    action: Action
    before: Phase
    after: Phase


@dataclass(frozen=True)
class GoalSpec:
    fixture: str
    start: Phase
    finish: Phase
    marked: frozenset[Phase]
    transitions: tuple[Step, ...]
    requires_receipt: bool = False

    def verifies_state(
        self, observed: Observation, phase: Phase, revision: int, marker: Marker
    ) -> bool:
        return (
            observed.phase == phase
            and observed.revision == revision
            and observed.marker == (marker if phase in self.marked else None)
            and observed.receipt == (self.requires_receipt and phase == self.finish)
        )


GOALS = {
    Goal.SUBMIT_MARKER: GoalSpec(
        FIXTURE,
        Phase.LOCKED,
        Phase.SUBMITTED,
        frozenset({Phase.FILLED, Phase.SUBMITTED}),
        (
            Step(Action.UNLOCK_A, Phase.LOCKED, Phase.READY),
            Step(Action.UNLOCK_B, Phase.LOCKED, Phase.READY),
            Step(Action.ENTER_MARKER, Phase.READY, Phase.FILLED),
            Step(Action.SUBMIT, Phase.FILLED, Phase.SUBMITTED),
        ),
    ),
    Goal.ISSUE_RECEIPT: GoalSpec(
        "forge-receipt-fixture",
        Phase.EMPTY,
        Phase.ISSUED,
        frozenset({Phase.STAGED, Phase.REVIEW, Phase.ISSUED}),
        (
            Step(Action.STAGE, Phase.EMPTY, Phase.STAGED),
            Step(Action.REVIEW, Phase.STAGED, Phase.REVIEW),
            Step(Action.QUICK_ISSUE, Phase.STAGED, Phase.ISSUED),
            Step(Action.ISSUE, Phase.REVIEW, Phase.ISSUED),
            Step(Action.BACK, Phase.REVIEW, Phase.STAGED),
        ),
        requires_receipt=True,
    ),
}


@dataclass(frozen=True)
class WorkflowProcedure:
    steps: tuple[Step, ...]
    goal: Goal = Goal.SUBMIT_MARKER
    schema_version: int = 3
    fixture_version: int = 1

    def validate(self) -> None:
        if (
            self.goal not in GOALS
            or self.schema_version not in (2, 3)
            or type(self.fixture_version) is not int
            or not 1 <= self.fixture_version <= 999
            or not 1 <= len(self.steps) <= MAX_STEPS
        ):
            raise Refused("unsupported_procedure")
        if self.schema_version == 2 and (
            self.goal != Goal.SUBMIT_MARKER or self.fixture_version != 1
        ):
            raise Refused("unsupported_procedure")
        spec = GOALS[self.goal]
        expected = spec.start
        for step in self.steps:
            if step not in spec.transitions or step.before != expected or expected == spec.finish:
                raise Refused("unsafe_step")
            expected = step.after
        if expected != spec.finish:
            raise Refused("goal_not_verified")

    def verifies(self, trace: list[Evidence]) -> bool:
        self.validate()
        verified = [entry for entry in trace if entry.verdict == Verdict.SUCCESS]
        if len(verified) != len(self.steps) or verified[-1] is not trace[-1]:
            return False
        first = verified[0]
        spec = GOALS[self.goal]
        for index, (step, entry) in enumerate(zip(self.steps, verified, strict=True)):
            before, after = entry.before, entry.after
            if not isinstance(before, Observation) or not isinstance(after, Observation):
                return False
            if not (
                first.target_id
                and entry.target_id == first.target_id
                and entry.marker == first.marker
                and entry.action == step.action
                and entry.reason == "verified"
                and step.action in before.controls
                and entry.fixture == spec.fixture
                and entry.fixture_version == self.fixture_version
                and (index == 0 or before == verified[index - 1].after)
                and spec.verifies_state(before, step.before, index, first.marker)
                and spec.verifies_state(after, step.after, index + 1, first.marker)
            ):
                return False
        return True

    @classmethod
    def read(cls, raw: dict) -> WorkflowProcedure:
        try:
            schema = raw["schema_version"]
            keys = {"schema_version", "goal", "steps"}
            if schema == 3:
                keys.add("fixture_version")
            if set(raw) != keys or type(schema) is not int or schema not in (2, 3):
                raise ValueError("invalid schema")
            if not isinstance(raw["steps"], list) or not 1 <= len(raw["steps"]) <= MAX_STEPS:
                raise ValueError("invalid step count")
            steps = []
            for step in raw["steps"]:
                if not isinstance(step, dict) or set(step) != {"action", "before", "after"}:
                    raise ValueError("invalid step")
                steps.append(
                    Step(Action(step["action"]), Phase(step["before"]), Phase(step["after"]))
                )
            procedure = cls(tuple(steps), Goal(raw["goal"]), schema, raw.get("fixture_version", 1))
            procedure.validate()
            return procedure
        except (ValueError, KeyError, TypeError):
            raise Refused("unsupported_procedure") from None


class WorkflowTarget(Target, Protocol):
    fixture_version: int

    async def observe_workflow(self) -> Observation: ...
    async def apply_workflow(
        self,
        action: Action,
        marker: Marker,
        revision: int,
        fingerprint: tuple[Any, ...],
        token: Any,
    ) -> ActionOutcome: ...


async def run_workflow(
    target: WorkflowTarget,
    marker: Marker,
    token: Any,
    trace: list[Evidence],
    limits: Limits,
    *,
    goal: Goal = Goal.SUBMIT_MARKER,
    procedure: WorkflowProcedure | None = None,
    rejected: frozenset[RejectionKey] = frozenset(),
) -> WorkflowProcedure:
    if not isinstance(marker, Marker) or goal not in GOALS:
        raise Refused("invalid_goal_parameters")
    spec = GOALS[goal]
    if target.kind != spec.fixture or not target.isolated or target.network_enabled:
        raise Refused("target_policy_denied")
    version = target.fixture_version
    if type(version) is not int or not 1 <= version <= 999:
        raise Refused("fixture_version_unavailable")
    if procedure is not None:
        procedure.validate()
        if procedure.goal != goal or procedure.fixture_version != version:
            raise Refused("fixture_version_drift")
    if token.is_cancelled():
        raise Refused("cancelled")

    async with asyncio.timeout(limits.seconds):
        identity = target.identity
        fingerprint = await target.fingerprint()
        if not identity or not fingerprint or fingerprint[0] == "none":
            raise Refused("target_unavailable")
        lease = Lease(identity, fingerprint, frozenset(step.action for step in spec.transitions))
        await lease.check(target)
        current = await target.observe_workflow()
        await lease.check(target)
        if not spec.verifies_state(current, spec.start, 0, marker):
            raise Refused("workflow_not_fresh")

        learned: list[Step] = []
        tried: set[RejectionKey] = set()
        rejections = 0
        for attempt in range(1, limits.actions + 1):
            if token.is_cancelled():
                raise Refused("cancelled")
            context = (spec.fixture, version, marker, current.state_key())
            if procedure is None:
                candidates = (
                    option
                    for option in spec.transitions
                    if option.before == current.phase
                    and option.action in current.controls
                    and (context, option.action) not in tried
                )
                # Prefer untried history. If behavior changed without a version
                # bump, a prior rejection can be reconsidered once within budget.
                step = min(
                    candidates,
                    key=lambda option: (context, option.action) in rejected,
                    default=None,
                )
                if step is None:
                    raise Refused("no_progress")
            else:
                step = procedure.steps[len(learned)]
                if step.before != current.phase or step.action not in current.controls:
                    raise Refused("workflow_drift")
            tried.add((context, step.action))
            await lease.check(target)
            if token.is_cancelled():
                raise Refused("cancelled")

            try:
                receipt = await target.apply_workflow(
                    step.action, marker, current.revision, fingerprint, token
                )
                await lease.check(target)
                after = await target.observe_workflow()
                await lease.check(target)
            except (Exception, asyncio.CancelledError):
                trace.append(
                    Evidence(
                        identity,
                        step.action,
                        marker,
                        current,
                        None,
                        Verdict.UNKNOWN,
                        "interrupted",
                        attempt,
                        spec.fixture,
                        version,
                    )
                )
                raise

            if receipt is ActionOutcome.ACCEPTED and spec.verifies_state(
                after, step.after, current.revision + 1, marker
            ):
                trace.append(
                    Evidence(
                        identity,
                        step.action,
                        marker,
                        current,
                        after,
                        Verdict.SUCCESS,
                        "verified",
                        attempt,
                        spec.fixture,
                        version,
                    )
                )
                learned.append(step)
                if token.is_cancelled():
                    raise Refused("cancelled")
                if after.phase == spec.finish:
                    result = WorkflowProcedure(tuple(learned), goal, fixture_version=version)
                    result.validate()
                    return result
                current = after
                continue

            proven_rejection = receipt is ActionOutcome.REJECTED and after == current
            verdict = Verdict.UNKNOWN if receipt is ActionOutcome.UNKNOWN else Verdict.FAILURE
            if proven_rejection:
                reason = "rejected_unchanged"
            elif receipt is ActionOutcome.UNKNOWN:
                reason = "action_unconfirmed"
            elif receipt is ActionOutcome.STALE:
                reason = "stale_revision"
            else:
                reason = "workflow_drift"
            trace.append(
                Evidence(
                    identity,
                    step.action,
                    marker,
                    current,
                    after,
                    verdict,
                    reason,
                    attempt,
                    spec.fixture,
                    version,
                )
            )
            if not proven_rejection or procedure is not None:
                raise Refused(reason)
            rejections += 1
            if rejections >= limits.attempts:
                raise Refused("retry_budget_exceeded")
        raise Refused("action_budget_exceeded")
