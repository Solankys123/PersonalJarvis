"""Typed workflow learning and replay over the leased FORGE fixture."""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from enum import StrEnum
from typing import Any, Protocol

from .engine import FIXTURE, Evidence, Lease, Limits, Refused, Target, Verdict

logger = logging.getLogger(__name__)


class Goal(StrEnum):
    SUBMIT_MARKER = "submit_marker"


class Marker(StrEnum):
    ALPHA = "alpha"
    BETA = "beta"


class Phase(StrEnum):
    LOCKED = "locked"
    READY = "ready"
    FILLED = "filled"
    SUBMITTED = "submitted"


class Action(StrEnum):
    UNLOCK_A = "unlock_a"
    UNLOCK_B = "unlock_b"
    ENTER_MARKER = "enter_marker"
    SUBMIT = "submit"


class ActionOutcome(StrEnum):
    ACCEPTED = "accepted"
    REJECTED = "rejected"
    STALE = "stale"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class Observation:
    phase: Phase
    revision: int
    marker: Marker | None
    controls: tuple[Action, ...]

    @classmethod
    def read(cls, raw: dict) -> Observation:
        """Read only the closed fixture vocabulary, ignoring labels and free text."""
        try:
            phase = Phase(raw["phase"])
            revision = raw["revision"]
            value = raw["marker"]
            marker = None if value is None else Marker(value)
            controls = raw["controls"]
            if type(revision) is not int or not 0 <= revision <= 3:
                raise ValueError("invalid revision")
            if not isinstance(controls, list) or len(controls) > len(Action):
                raise ValueError("invalid controls")
            return cls(
                phase,
                revision,
                marker,
                tuple(action for action in Action if action.value in controls),
            )
        except (KeyError, ValueError, TypeError):
            raise Refused("invalid_observation") from None


# Allowed effects constrain exploration. The guest decides which controls are
# exposed and which of the two unlock actions can actually make progress.
EFFECTS = {
    Action.UNLOCK_A: (Phase.LOCKED, Phase.READY),
    Action.UNLOCK_B: (Phase.LOCKED, Phase.READY),
    Action.ENTER_MARKER: (Phase.READY, Phase.FILLED),
    Action.SUBMIT: (Phase.FILLED, Phase.SUBMITTED),
}


@dataclass(frozen=True)
class Step:
    action: Action
    before: Phase
    after: Phase

    def validate(self) -> None:
        if EFFECTS.get(self.action) != (self.before, self.after):
            raise Refused("unsafe_step")


@dataclass(frozen=True)
class WorkflowProcedure:
    steps: tuple[Step, ...]
    goal: Goal = Goal.SUBMIT_MARKER
    schema_version: int = 2

    def validate(self) -> None:
        if self.goal != Goal.SUBMIT_MARKER or self.schema_version != 2 or len(self.steps) != 3:
            raise Refused("unsupported_procedure")
        expected = Phase.LOCKED
        for step in self.steps:
            step.validate()
            if step.before != expected:
                raise Refused("invalid_step_chain")
            expected = step.after
        if expected != Phase.SUBMITTED:
            raise Refused("goal_not_verified")

    def verifies(self, trace: list[Evidence]) -> bool:
        self.validate()
        verified = [entry for entry in trace if entry.verdict == Verdict.SUCCESS]
        if len(verified) != len(self.steps) or verified[-1] is not trace[-1]:
            return False
        first = verified[0]
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
                and (before.phase, before.revision, before.marker)
                == (step.before, index, first.marker if step.before == Phase.FILLED else None)
                and (after.phase, after.revision, after.marker)
                == (step.after, index + 1, first.marker if index >= 1 else None)
            ):
                return False
        return True

    @classmethod
    def read(cls, raw: dict) -> WorkflowProcedure:
        try:
            if set(raw) != {"schema_version", "goal", "steps"}:
                raise ValueError("invalid procedure fields")
            if type(raw["schema_version"]) is not int or raw["schema_version"] != 2:
                raise ValueError("invalid schema version")
            if not isinstance(raw["steps"], list) or len(raw["steps"]) != 3:
                raise ValueError("invalid step count")
            steps = tuple(
                Step(Action(step["action"]), Phase(step["before"]), Phase(step["after"]))
                for step in raw["steps"]
                if isinstance(step, dict) and set(step) == {"action", "before", "after"}
            )
            procedure = cls(steps, Goal(raw["goal"]))
            procedure.validate()
            return procedure
        except (ValueError, KeyError, TypeError):
            raise Refused("unsupported_procedure") from None


class WorkflowTarget(Target, Protocol):
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
    procedure: WorkflowProcedure | None = None,
    rejected: frozenset[tuple[Phase, Action]] = frozenset(),
) -> WorkflowProcedure:
    if not isinstance(marker, Marker):
        raise Refused("marker_denied")
    if procedure is not None:
        procedure.validate()
    if target.kind != FIXTURE or not target.isolated or target.network_enabled:
        raise Refused("target_policy_denied")

    if token.is_cancelled():
        raise Refused("cancelled")
    async with asyncio.timeout(limits.seconds):
        identity = target.identity
        fingerprint = await target.fingerprint()
        if not identity or not fingerprint or fingerprint[0] == "none":
            raise Refused("target_unavailable")
        lease = Lease(identity, fingerprint, frozenset(Action))
        await lease.check(target)
        initial = await target.observe_workflow()
        await lease.check(target)
        if (initial.phase, initial.revision, initial.marker) != (Phase.LOCKED, 0, None):
            raise Refused("workflow_not_fresh")
        current = initial
        learned: list[Step] = []
        rejected_now: set[tuple[Phase, Action]] = set()
        for attempt in range(1, limits.actions + 1):
            if token.is_cancelled():
                raise Refused("cancelled")
            if procedure is None:
                candidates = [
                    action
                    for action in current.controls
                    if action in lease.actions
                    and EFFECTS[action][0] == current.phase
                    and (current.phase, action) not in rejected_now
                    and (current.phase, action) not in rejected
                ]
                if not candidates:
                    raise Refused("no_progress")
                action = min(candidates)
                step = Step(action, *EFFECTS[action])
            else:
                step = procedure.steps[len(learned)]
                if step.before != current.phase or step.action not in current.controls:
                    raise Refused("workflow_drift")
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
                    )
                )
                raise

            expected_marker = marker if step.after in (Phase.FILLED, Phase.SUBMITTED) else None
            if receipt is ActionOutcome.ACCEPTED and (
                after.phase,
                after.revision,
                after.marker,
            ) == (step.after, current.revision + 1, expected_marker):
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
                    )
                )
                learned.append(step)
                if token.is_cancelled():
                    raise Refused("cancelled")
                if after.phase == Phase.SUBMITTED:
                    result = WorkflowProcedure(tuple(learned))
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
                Evidence(identity, step.action, marker, current, after, verdict, reason, attempt)
            )
            if not proven_rejection or procedure is not None:
                raise Refused(reason)
            rejected_now.add((current.phase, step.action))
            if len(rejected_now) >= limits.attempts:
                raise Refused("retry_budget_exceeded")
        raise Refused("action_budget_exceeded")
