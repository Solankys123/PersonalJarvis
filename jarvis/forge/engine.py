"""Deterministic learning for an isolated text-entry fixture."""

from __future__ import annotations

import asyncio
import hashlib
import json
from dataclasses import asdict, dataclass
from enum import StrEnum
from typing import TYPE_CHECKING, Any, Protocol

if TYPE_CHECKING:
    from .workflow import Observation

FIXTURE = "forge-text-v1"
MARKERS = {"alpha": "FORGE ALPHA", "beta": "FORGE BETA"}


class Verdict(StrEnum):
    SUCCESS = "success"
    FAILURE = "failure"
    UNKNOWN = "unknown"


class State(StrEnum):
    BLANK = "blank"
    MATCH = "match"
    DRIFT = "drift"
    UNKNOWN = "unknown"


class Refused(RuntimeError):
    """A fail-closed policy decision."""


@dataclass(frozen=True)
class Procedure:
    schema_version: int = 1
    target: str = FIXTURE
    action: str = "type_text"
    parameter: str = "marker"
    precondition: str = "blank"
    postcondition: str = "exact_marker"

    def validate(self) -> None:
        if self != Procedure() or type(self.schema_version) is not int:
            raise Refused("unsupported_procedure")


@dataclass(frozen=True)
class Evidence:
    target_id: str
    action: str
    marker: str
    before: State | Observation
    after: State | Observation | None
    verdict: Verdict
    reason: str
    attempt: int
    fixture: str | None = None
    fixture_version: int | None = None

    def record(self) -> dict[str, Any]:
        data = asdict(self)
        encoded = json.dumps(data, sort_keys=True).encode()
        return {**data, "sha256": hashlib.sha256(encoded).hexdigest()}


class Target(Protocol):
    identity: str
    kind: str
    isolated: bool
    network_enabled: bool

    async def fingerprint(self) -> tuple[Any, ...]: ...
    async def observe(self, marker: str) -> State: ...
    async def apply(self, marker: str, fingerprint: tuple[Any, ...], token: Any) -> bool: ...


@dataclass(frozen=True)
class Lease:
    target_id: str
    fingerprint: tuple[Any, ...]
    actions: frozenset[str] = frozenset({"type_text"})

    async def check(self, target: Target) -> None:
        if target.identity != self.target_id or await target.fingerprint() != self.fingerprint:
            raise Refused("target_changed")


@dataclass(frozen=True)
class Limits:
    attempts: int = 2
    seconds: float = 25
    actions: int = 8

    def __post_init__(self) -> None:
        if (
            not 1 <= self.attempts <= 3
            or not 1 <= self.seconds <= 60
            or not 1 <= self.actions <= 12
        ):
            raise ValueError("invalid experiment limits")


async def execute(
    target: Target,
    procedure: Procedure,
    marker: str,
    token: Any,
    trace: list[Evidence],
    limits: Limits,
) -> None:
    procedure.validate()
    if target.kind != FIXTURE or not target.isolated or target.network_enabled:
        raise Refused("target_policy_denied")
    if marker not in MARKERS:
        raise Refused("marker_denied")

    async with asyncio.timeout(limits.seconds):
        identity = target.identity
        fingerprint = await target.fingerprint()
        if not identity or not fingerprint or fingerprint[0] == "none":
            raise Refused("target_unavailable")
        lease = Lease(identity, fingerprint)
        if procedure.action not in lease.actions:
            raise Refused("capability_denied")
        for attempt in range(1, limits.attempts + 1):
            if token.is_cancelled():
                raise Refused("cancelled")
            await lease.check(target)
            before = await target.observe(marker)
            await lease.check(target)
            if before != State.BLANK:
                verdict = Verdict.SUCCESS if before == State.MATCH else Verdict.FAILURE
                if before == State.UNKNOWN:
                    verdict = Verdict.UNKNOWN
                trace.append(
                    Evidence(
                        identity,
                        procedure.action,
                        marker,
                        before,
                        before,
                        verdict,
                        "already_matched" if before == State.MATCH else "precondition_failed",
                        attempt,
                    )
                )
                return
            if token.is_cancelled():
                raise Refused("cancelled")
            try:
                accepted = await target.apply(marker, fingerprint, token)
                await lease.check(target)
                after = await target.observe(marker)
                await lease.check(target)
            except (Exception, asyncio.CancelledError):
                trace.append(
                    Evidence(
                        identity,
                        procedure.action,
                        marker,
                        before,
                        State.UNKNOWN,
                        Verdict.UNKNOWN,
                        "interrupted",
                        attempt,
                    )
                )
                raise
            if after == State.MATCH:
                verdict, reason = Verdict.SUCCESS, "verified"
            elif after == State.UNKNOWN:
                verdict, reason = Verdict.UNKNOWN, "verification_unavailable"
            else:
                verdict = Verdict.FAILURE
                reason = "postcondition_failed" if accepted else "action_failed"
            trace.append(
                Evidence(
                    identity, procedure.action, marker, before, after, verdict, reason, attempt
                )
            )
            # Only a rejected action with proof of the unchanged blank state
            # may retry. Partial input and uncertain outcomes require a reset.
            if accepted or after != State.BLANK:
                return
