"""The same learner drives two fixtures, keeps scoped evidence, and versions skills."""

from __future__ import annotations

import asyncio
import json
import sqlite3
from contextlib import asynccontextmanager

import pytest

from jarvis.agent_screen.protocol import ForegroundInfo
from jarvis.forge.desktop import WindowsFixture
from jarvis.forge.engine import Evidence, Limits, Verdict
from jarvis.forge.store import Store
from jarvis.forge.tool import ForgeMindTool
from jarvis.forge.workflow import GOALS, Action, Goal, Marker, Observation, Phase
from tests.forge.test_forge import FakeScreen, context, executor


class ReceiptScreen(FakeScreen):
    """A separate, fake-only device whose terminal proof includes an issued receipt."""

    def __init__(self, controls, rejected=(), *, issue_receipt=True):
        super().__init__()
        self.phase = "empty"
        self.marker = None
        self.revision = 0
        self.receipt = False
        self.controls = controls
        self.rejected = set(rejected)
        self.issue_receipt = issue_receipt
        self.calls = []

    def foreground(self):
        return ForegroundInfo(True, app=GOALS[Goal.ISSUE_RECEIPT].fixture, handle=self.handle)

    def forge_observation(self):
        return {
            "phase": self.phase,
            "revision": self.revision,
            "marker": self.marker,
            "receipt": self.receipt,
            "controls": self.controls.get(self.phase, []),
            "label": "untrusted content",
        }

    def forge_action(self, params):
        action = params["action"]
        self.calls.append(action)
        assert action in self.controls[self.phase]
        if params["revision"] != self.revision or params["handle"] != str(self.handle):
            return "stale"
        if action in self.rejected:
            return "rejected"
        if action == "stage":
            self.marker = params["marker"]
        self.phase = {
            "stage": "staged",
            "review": "review",
            "back": "staged",
            "issue": "issued",
            "quick_issue": "issued",
        }[action]
        self.revision += 1
        self.receipt = self.phase == "issued" and self.issue_receipt
        return "accepted"


class ReceiptFixture(WindowsFixture):
    kind = GOALS[Goal.ISSUE_RECEIPT].fixture


REVIEWED = {"empty": ["stage"], "staged": ["review", "quick_issue"], "review": ["issue", "back"]}
QUICK = {"empty": ["stage"], "staged": ["quick_issue"]}


@pytest.fixture
def adaptive(tmp_path):
    store = Store(tmp_path / "forge.sqlite")
    screens = []
    boundary = executor()

    def build(*, controls=REVIEWED, rejected=(), version=1, receipt=True, limits=None):
        @asynccontextmanager
        async def guest(run_id, ctx):
            screen = ReceiptScreen(controls, rejected, issue_receipt=receipt)
            target = ReceiptFixture(screen, boundary, ctx)
            target.fixture_version = version
            screens.append(screen)
            yield target

        return ForgeMindTool(Store(store.path), guest, limits or Limits())

    return store, screens, build


async def invoke(tool, command):
    return await tool.execute({}, context(f"forge {command}"))


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("controls", "rejected", "actions", "steps"),
    [
        (QUICK, (), ["stage", "quick_issue"], 2),
        (REVIEWED, (), ["stage", "review", "issue"], 3),
        (REVIEWED, ("issue",), ["stage", "review", "issue", "back", "quick_issue"], 4),
    ],
)
async def test_learns_variable_length_paths_and_replays_evidence(
    adaptive, controls, rejected, actions, steps
):
    store, screens, build = adaptive
    tool = build(controls=controls, rejected=rejected)
    result = await invoke(tool, "learn receipt alpha")
    assert result.success
    assert screens[-1].calls == actions
    procedure, state = store.get(1)
    assert state == "draft" and procedure.goal == Goal.ISSUE_RECEIPT
    assert len(procedure.steps) == steps
    assert (await invoke(tool, "validate 1 beta")).success
    assert (await invoke(tool, "promote 1")).success
    assert (await invoke(tool, "replay 1 beta")).success
    assert screens[-1].receipt and screens[-1].marker == "beta"
    assert len(screens[-1].calls) == steps


@pytest.mark.asyncio
async def test_terminal_phase_without_receipt_is_independently_refused(adaptive):
    store, _, build = adaptive
    result = await invoke(build(controls=QUICK, receipt=False), "learn receipt alpha")
    assert result.error == "workflow_drift"
    assert store.list() == []


@pytest.mark.asyncio
async def test_failure_memory_is_state_fixture_version_and_marker_scoped(adaptive):
    store, screens, build = adaptive
    # A different fixture's rejection may share an action identifier but is not evidence here.
    state = Observation(Phase.STAGED, 1, Marker.ALPHA, (Action.REVIEW, Action.QUICK_ISSUE))
    foreign = Evidence(
        "other",
        Action.REVIEW,
        "alpha",
        state,
        state,
        Verdict.FAILURE,
        "rejected_unchanged",
        1,
        "other-fixture",
        1,
    )
    store.record("foreign", "learn", None, "no_progress", [foreign])
    assert (await invoke(build(rejected=("review",)), "learn receipt alpha")).success
    assert screens[-1].calls == ["stage", "review", "quick_issue"]
    assert (await invoke(build(), "learn receipt alpha")).success
    assert screens[-1].calls == ["stage", "quick_issue"]
    assert (await invoke(build(version=2), "learn receipt alpha")).success
    assert screens[-1].calls == ["stage", "review", "issue"]
    assert (await invoke(build(), "learn receipt beta")).success
    assert screens[-1].calls == ["stage", "review", "issue"]
    review_only = {**REVIEWED, "staged": ["review"]}
    assert (await invoke(build(controls=review_only), "learn receipt alpha")).success
    assert screens[-1].calls == ["stage", "review", "issue"]


@pytest.mark.asyncio
async def test_drift_improvement_fresh_validation_promotion_and_rollback(adaptive):
    store, screens, build = adaptive
    original = build()
    assert (await invoke(original, "learn receipt alpha")).success
    assert (await invoke(original, "validate 1 beta")).success
    assert (await invoke(original, "promote 1")).success
    assert (await invoke(original, "improve 1 alpha")).error == "no_procedure_change"

    changed = build(controls=QUICK, version=2)
    assert (await invoke(changed, "replay 1 beta")).error == "fixture_version_drift"
    assert screens[-1].calls == []
    improved = await invoke(changed, "improve 1 alpha")
    assert improved.output["version"] == 2
    assert store.parent(2) == 1
    assert not (await invoke(changed, "promote 2")).success
    assert store.get(1)[1] == "promoted"  # Failed promotion rolled back the transaction.
    assert not (await invoke(changed, "validate 2 alpha")).success
    assert (await invoke(changed, "validate 2 beta")).success
    assert (await invoke(changed, "promote 2")).success
    assert store.get(1)[1] == "validated"
    assert not (await invoke(original, "promote 1")).success
    assert (await invoke(changed, "replay 2 beta")).success
    assert (await invoke(changed, "rollback 2")).success
    assert store.get(2)[1] == "validated"
    assert not (await invoke(changed, "replay 2 beta")).success
    assert (await invoke(original, "replay 1 beta")).success
    assert not (await invoke(changed, "rollback 2")).success


@pytest.mark.asyncio
async def test_interrupted_delivery_after_guest_side_effect_never_reopens(tmp_path):
    store = Store(tmp_path / "forge.sqlite")
    applied = asyncio.Event()
    opens = []
    screen = ReceiptScreen(QUICK)

    class LostReceipt(ReceiptFixture):
        async def apply_workflow(self, *args, **kwargs):
            await super().apply_workflow(*args, **kwargs)
            applied.set()
            await asyncio.Event().wait()

    @asynccontextmanager
    async def guest(run_id, ctx):
        opens.append(run_id)
        yield LostReceipt(screen, executor(), ctx)

    tool = ForgeMindTool(store, guest, Limits())
    request = context("forge learn receipt alpha")
    task = asyncio.create_task(tool.execute({}, request))
    await asyncio.wait_for(applied.wait(), 2)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    result = await tool.execute({}, request)
    assert result.error == "interrupted"
    assert screen.calls == ["stage"] and len(opens) == 1
    assert store.list() == []


@pytest.mark.asyncio
async def test_recovery_obeys_global_action_budget(adaptive):
    store, screens, build = adaptive
    tool = build(rejected=("issue",), limits=Limits(actions=4))
    result = await invoke(tool, "learn receipt alpha")
    assert result.error == "action_budget_exceeded"
    assert screens[-1].calls == ["stage", "review", "issue", "back"]
    assert store.list() == []


def test_migrates_old_database_and_reads_v2_procedure(tmp_path):
    path = tmp_path / "forge.sqlite"
    legacy = {
        "schema_version": 2,
        "goal": "submit_marker",
        "steps": [
            {"action": "unlock_b", "before": "locked", "after": "ready"},
            {"action": "enter_marker", "before": "ready", "after": "filled"},
            {"action": "submit", "before": "filled", "after": "submitted"},
        ],
    }
    store = Store(path)
    # Recreate the on-disk shape of the old store while preserving its procedure.
    with store.connection() as db:
        db.execute("INSERT INTO procedures VALUES (1, ?, 'promoted')", (json.dumps(legacy),))
        db.executescript("DROP TABLE skill_lineage; PRAGMA user_version=1;")
    procedure, state = Store(path).get(1)
    assert procedure.schema_version == 2 and procedure.fixture_version == 1
    assert state == "promoted" and store.parent(1) is None
    with sqlite3.connect(path) as db:
        assert db.execute("PRAGMA user_version").fetchone() == (2,)


@pytest.mark.asyncio
async def test_unavailable_goal_cannot_start_another_guest(tmp_path):
    @asynccontextmanager
    async def guest(run_id, ctx):
        pytest.fail("unavailable goals must not launch a guest")
        yield

    tool = ForgeMindTool(
        Store(tmp_path / "forge.sqlite"),
        guest,
        Limits(),
        available_goals=frozenset({Goal.SUBMIT_MARKER}),
    )
    assert (await invoke(tool, "learn receipt alpha")).error == "fixture_unavailable"
