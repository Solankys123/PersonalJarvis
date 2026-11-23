"""Versioned SQLite persistence for procedures and bounded evidence."""

from __future__ import annotations

import json
import sqlite3
from contextlib import closing, contextmanager
from dataclasses import asdict
from pathlib import Path

from .engine import Evidence, Procedure, Refused, State, Verdict

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

    def get(self, version: int) -> tuple[Procedure, str]:
        with self.connection() as db:
            row = db.execute(
                "SELECT spec, state FROM procedures WHERE version=?",
                (version,),
            ).fetchone()
        if row is None:
            raise Refused("unknown_version")
        procedure = Procedure(**json.loads(row[0]))
        procedure.validate()
        return procedure, row[1]

    def claim(self, run_id: str, mode: str, version: int | None) -> tuple[int | None, str] | None:
        """Reserve a delivery before starting its guest, or return the old receipt."""
        with self.connection() as db:
            inserted = db.execute(
                "INSERT OR IGNORE INTO runs VALUES (?, ?, ?, 'interrupted', '[]')",
                (run_id, version, mode),
            ).rowcount
            if inserted:
                return None
            return db.execute(
                "SELECT version, outcome FROM runs WHERE id=?",
                (run_id,),
            ).fetchone()

    def record(
        self, run_id: str, mode: str, version: int | None, outcome: str, evidence: list[Evidence]
    ) -> int | None:
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
        with self.connection() as db:
            if mode == "learn" and verified:
                row = db.execute(
                    "INSERT INTO procedures(spec,state) VALUES (?, 'draft')",
                    (json.dumps(asdict(Procedure())),),
                )
                version = row.lastrowid
            if mode == "validate" and verified:
                learned = db.execute(
                    "SELECT evidence FROM runs WHERE version=? AND mode='learn'",
                    (version,),
                ).fetchone()
                if learned:
                    original = json.loads(learned[0])[-1]
                    if (
                        original["target_id"] != evidence[-1].target_id
                        and original["marker"] != evidence[-1].marker
                    ):
                        db.execute(
                            "UPDATE procedures SET state='validated' "
                            "WHERE version=? AND state='draft'",
                            (version,),
                        )
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
        return version

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
