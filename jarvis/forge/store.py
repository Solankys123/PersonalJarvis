"""Versioned SQLite persistence for procedures and bounded evidence."""

from __future__ import annotations

import json
import sqlite3
from contextlib import closing, contextmanager
from dataclasses import asdict
from pathlib import Path

from .browser import BrowserProcedure, BrowserStep
from .engine import Evidence, Procedure, Refused, State, Verdict
from .live_navigation import LiveSearch
from .workflow import Action, Marker, Observation, RejectionKey, WorkflowProcedure

SCHEMA = """
BEGIN IMMEDIATE;
CREATE TABLE IF NOT EXISTS procedures (
    version INTEGER PRIMARY KEY,
    spec TEXT NOT NULL,
    state TEXT NOT NULL CHECK (state IN ('draft', 'validated', 'promoted'))
);
CREATE TABLE IF NOT EXISTS runs (
    id TEXT PRIMARY KEY,
    version INTEGER REFERENCES procedures(version),
    mode TEXT NOT NULL CHECK (mode IN ('learn', 'validate', 'replay')),
    outcome TEXT NOT NULL,
    evidence TEXT NOT NULL
);
"""

LINEAGE_SCHEMA = """
CREATE TABLE IF NOT EXISTS skill_lineage (
    version INTEGER PRIMARY KEY REFERENCES procedures(version),
    parent INTEGER NOT NULL REFERENCES procedures(version)
);
PRAGMA user_version = 2;
COMMIT;
"""

SCHEMA += LINEAGE_SCHEMA
MIGRATE_V1 = "BEGIN IMMEDIATE;\n" + LINEAGE_SCHEMA


class Store:
    def __init__(self, path: Path) -> None:
        self.path = path

    @contextmanager
    def connection(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with closing(sqlite3.connect(self.path)) as db:
            db.execute("PRAGMA foreign_keys = ON")
            version = db.execute("PRAGMA user_version").fetchone()[0]
            if version == 0:
                db.executescript(SCHEMA)
            elif version == 1:
                db.executescript(MIGRATE_V1)
            elif version != 2:
                raise Refused("unsupported_database_version")
            with db:
                yield db

    def list(self) -> list[dict]:
        with self.connection() as db:
            return [
                dict(version=v, state=s)
                for v, s in db.execute(
                    "SELECT version, state FROM procedures ORDER BY version DESC LIMIT 50"
                )
            ]

    def get(
        self, version: int
    ) -> tuple[Procedure | WorkflowProcedure | BrowserProcedure | LiveSearch, str]:
        with self.connection() as db:
            row = db.execute(
                "SELECT spec, state FROM procedures WHERE version=?",
                (version,),
            ).fetchone()
        if row is None:
            raise Refused("unknown_version")
        try:
            raw = json.loads(row[0])
            if not isinstance(raw, dict):
                raise ValueError("invalid procedure")
            if type(raw.get("schema_version")) is int and raw["schema_version"] == 5:
                raw["steps"] = tuple(raw["steps"])
                procedure = LiveSearch(**raw)
            elif type(raw.get("schema_version")) is int and raw["schema_version"] == 4:
                raw["steps"] = tuple(map(BrowserStep, raw["steps"]))
                procedure = BrowserProcedure(**raw)
            elif raw.get("schema_version") in (2, 3):
                procedure = WorkflowProcedure.read(raw)
            else:
                procedure = Procedure(**raw)
            procedure.validate()
        except (KeyError, ValueError, TypeError):
            raise Refused("unsupported_procedure") from None
        return procedure, row[1]

    def receipt(self, run_id: str) -> tuple[int | None, str] | None:
        with self.connection() as db:
            return self._receipt(db, run_id)

    @staticmethod
    def _receipt(db: sqlite3.Connection, run_id: str) -> tuple[int | None, str] | None:
        row = db.execute(
            "SELECT version, outcome, mode, evidence FROM runs WHERE id=?",
            (run_id,),
        ).fetchone()
        if row is None:
            return None
        version, outcome, mode, evidence = row
        if mode == "validate" and outcome == Verdict.SUCCESS:
            # Older receipts recorded action success even for a repeated marker.
            original = Store._learning_evidence(db, version)
            steps = json.loads(evidence)
            state = db.execute(
                "SELECT state FROM procedures WHERE version=?", (version,)
            ).fetchone()
            if (
                not steps
                or not original
                or not state
                or state[0] == "draft"
                or steps[-1]["marker"] == original["marker"]
                or steps[-1]["target_id"] == original["target_id"]
            ):
                outcome = "independent_validation_required"
        return version, outcome

    @staticmethod
    def _learning_evidence(db: sqlite3.Connection, version: int | None) -> dict | None:
        row = db.execute(
            "SELECT evidence FROM runs WHERE version=? AND mode='learn' AND outcome='success'",
            (version,),
        ).fetchone()
        return json.loads(row[0])[-1] if row else None

    def claim(self, run_id: str, mode: str, version: int | None) -> tuple[int | None, str] | None:
        """Reserve a delivery before starting its guest, or return the old receipt."""
        with self.connection() as db:
            inserted = db.execute(
                "INSERT OR IGNORE INTO runs VALUES (?, ?, ?, 'interrupted', '[]')",
                (run_id, version, mode),
            ).rowcount
            if inserted:
                return None
            return self._receipt(db, run_id)

    def rejected_actions(self, marker: Marker) -> frozenset[RejectionKey]:
        """Reuse only scoped unchanged-state rejections from recent explorations."""
        with self.connection() as db:
            rows = db.execute(
                "SELECT evidence FROM runs WHERE mode='learn' ORDER BY rowid DESC LIMIT 50"
            ).fetchall()
        return frozenset(
            (
                (
                    step["fixture"],
                    step["fixture_version"],
                    marker,
                    Observation.read(step["before"]).state_key(),
                ),
                Action(step["action"]),
            )
            for (raw,) in rows
            for step in json.loads(raw)
            if step["reason"] == "rejected_unchanged"
            and step["marker"] == marker
            and step.get("fixture")
            and step.get("fixture_version") is not None
        )

    def parent(self, version: int) -> int | None:
        with self.connection() as db:
            row = db.execute(
                "SELECT parent FROM skill_lineage WHERE version=?", (version,)
            ).fetchone()
            return row[0] if row else None

    def browser_failures(self, version: int) -> list[str]:
        with self.connection() as db:
            return [
                outcome
                for (outcome,) in db.execute(
                    "SELECT outcome FROM runs WHERE version=? AND mode IN ('validate', 'replay') "
                    "AND outcome != 'success' ORDER BY rowid DESC LIMIT 5",
                    (version,),
                )
            ]

    def record(
        self,
        run_id: str,
        mode: str,
        version: int | None,
        outcome: str,
        evidence: list[Evidence],
        procedure: WorkflowProcedure | BrowserProcedure | LiveSearch | None = None,
        parent: int | None = None,
    ) -> tuple[int | None, str]:
        final = evidence[-1] if evidence else None
        verified = (
            outcome == Verdict.SUCCESS
            and final is not None
            and final.verdict == Verdict.SUCCESS
            and final.before == State.BLANK
            and final.after == State.MATCH
            and final.reason == "verified"
            and bool(final.target_id)
        )
        if outcome == Verdict.SUCCESS and procedure is not None:
            verified = procedure.verifies(evidence)
            if not verified:
                outcome = "goal_not_verified"
        with self.connection() as db:
            if isinstance(procedure, (BrowserProcedure, LiveSearch)) and verified:
                db.execute("BEGIN IMMEDIATE")
            if parent is not None and verified:
                if not db.in_transaction:
                    db.execute("BEGIN IMMEDIATE")
                row = db.execute(
                    "SELECT state FROM procedures WHERE version=?", (parent,)
                ).fetchone()
                if row != ("promoted",):
                    verified, outcome = False, "active_parent_required"
            if mode == "learn" and verified:
                spec = json.dumps(asdict(procedure if procedure is not None else Procedure()))
                duplicate = None
                if isinstance(procedure, (BrowserProcedure, LiveSearch)):
                    duplicate = db.execute(
                        "SELECT p.version FROM procedures p "
                        "LEFT JOIN skill_lineage l USING (version) "
                        "WHERE p.spec=? AND l.parent IS ? ORDER BY p.version LIMIT 1",
                        (spec, parent),
                    ).fetchone()
                if duplicate:
                    version, outcome = duplicate[0], "procedure_already_saved"
                else:
                    row = db.execute(
                        "INSERT INTO procedures(spec,state) VALUES (?, 'draft')", (spec,)
                    )
                    version = row.lastrowid
                    if parent is not None:
                        db.execute("INSERT INTO skill_lineage VALUES (?, ?)", (version, parent))
            if mode == "validate" and verified:
                original = self._learning_evidence(db, version)
                changed = 0
                if (
                    original
                    and original["target_id"] != final.target_id
                    and original["marker"] != final.marker
                ):
                    changed = db.execute(
                        "UPDATE procedures SET state='validated' WHERE version=? AND state='draft'",
                        (version,),
                    ).rowcount
                if changed != 1:
                    outcome = "independent_validation_required"
            db.execute(
                "INSERT INTO runs VALUES (?, ?, ?, ?, ?) "
                "ON CONFLICT(id) DO UPDATE SET version=excluded.version, "
                "outcome=excluded.outcome, evidence=excluded.evidence",
                (
                    run_id,
                    version,
                    mode,
                    outcome,
                    json.dumps([step.record() for step in evidence]),
                ),
            )
        return version, outcome

    def promote(self, version: int, *, authorized: bool) -> None:
        if not authorized:
            raise Refused("promotion_requires_authorization")
        with self.connection() as db:
            db.execute("BEGIN IMMEDIATE")
            row = db.execute(
                "SELECT parent FROM skill_lineage WHERE version=?", (version,)
            ).fetchone()
            if row:
                changed = db.execute(
                    "UPDATE procedures SET state='validated' WHERE version=? AND state='promoted'",
                    row,
                ).rowcount
                if changed != 1:
                    raise Refused("active_parent_required")
            else:
                active_child = db.execute(
                    """WITH RECURSIVE descendants(version) AS (
                        SELECT version FROM skill_lineage WHERE parent=?
                        UNION
                        SELECT s.version FROM skill_lineage s
                        JOIN descendants d ON s.parent=d.version
                    )
                    SELECT 1 FROM descendants d JOIN procedures p USING (version)
                    WHERE p.state='promoted' LIMIT 1""",
                    (version,),
                ).fetchone()
                if active_child:
                    raise Refused("active_descendant_exists")
            changed = db.execute(
                "UPDATE procedures SET state='promoted' WHERE version=? AND state='validated'",
                (version,),
            ).rowcount
            if changed != 1:
                raise Refused("validated_draft_required")

    def rollback(self, version: int, *, authorized: bool) -> int:
        if not authorized:
            raise Refused("rollback_requires_authorization")
        with self.connection() as db:
            db.execute("BEGIN IMMEDIATE")
            row = db.execute(
                "SELECT parent FROM skill_lineage WHERE version=?", (version,)
            ).fetchone()
            if row is None:
                raise Refused("no_prior_version")
            demoted = db.execute(
                "UPDATE procedures SET state='validated' WHERE version=? AND state='promoted'",
                (version,),
            ).rowcount
            restored = db.execute(
                "UPDATE procedures SET state='promoted' WHERE version=? AND state='validated'",
                row,
            ).rowcount
            if demoted != 1 or restored != 1:
                raise Refused("rollback_version_unavailable")
            return row[0]
