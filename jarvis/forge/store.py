"""Versioned SQLite persistence for procedures and bounded evidence."""

from __future__ import annotations

import json
import sqlite3
from contextlib import closing, contextmanager
from dataclasses import asdict
from pathlib import Path

from .engine import Evidence, Procedure, Refused, State, Verdict
from .workflow import Action, Marker, Phase, WorkflowProcedure

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
PRAGMA user_version = 1;
COMMIT;
"""


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
            elif version != 1:
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

    def get(self, version: int) -> tuple[Procedure | WorkflowProcedure, str]:
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
            procedure = (
                WorkflowProcedure.read(raw) if raw.get("schema_version") == 2 else Procedure(**raw)
            )
            procedure.validate()
        except (ValueError, TypeError):
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
            "SELECT evidence FROM runs WHERE version=? AND mode='learn'",
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

    def rejected_actions(self, marker: Marker) -> frozenset[tuple[Phase, Action]]:
        """Reuse only unchanged-state rejections from recent fixture explorations."""
        with self.connection() as db:
            rows = db.execute(
                "SELECT evidence FROM runs WHERE mode='learn' ORDER BY rowid DESC LIMIT 50"
            ).fetchall()
        return frozenset(
            (Phase(step["before"]["phase"]), Action(step["action"]))
            for (raw,) in rows
            for step in json.loads(raw)
            if step["reason"] == "rejected_unchanged" and step["marker"] == marker
        )

    def record(
        self,
        run_id: str,
        mode: str,
        version: int | None,
        outcome: str,
        evidence: list[Evidence],
        procedure: WorkflowProcedure | None = None,
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
            if mode == "learn" and verified:
                row = db.execute(
                    "INSERT INTO procedures(spec,state) VALUES (?, 'draft')",
                    (json.dumps(asdict(procedure if procedure is not None else Procedure())),),
                )
                version = row.lastrowid
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
            changed = db.execute(
                "UPDATE procedures SET state='promoted' WHERE version=? AND state='validated'",
                (version,),
            ).rowcount
            if changed != 1:
                raise Refused("validated_draft_required")
