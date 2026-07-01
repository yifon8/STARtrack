"""
Skill: SQLite persistence layer (opt-in alternative)
Owner: P3
Depends on: progression.py (record shape) + history/*.jsonl files (seed data).

Replaces flat .jsonl file reads/writes with SQLite, behind the same
interface progression.py already exposes, so no downstream Skill
(assessment.py, scoresheet.py, eval.py, ui.py) needs to change.

Functions:
    init_db()           -> creates the sessions table and imports any
                            existing history/*.jsonl files (idempotent,
                            safe to call on every app start).
    save_session_db()   -> same contract as progression.save_session().
    load_history_db()   -> same contract as progression._load_history(),
                            renamed to a public name since this is now a
                            first-class read path other Skills may call
                            directly instead of just progression.py.

Schema:
    CREATE TABLE sessions (
        question_id            TEXT NOT NULL,
        user_id                 TEXT NOT NULL,
        attempt_number          INTEGER NOT NULL,
        date                     TEXT,
        source_file              TEXT,
        transcription_method     TEXT,
        transcript                TEXT NOT NULL,
        scores                    TEXT NOT NULL,   -- JSON-encoded dict
        strengths                 TEXT NOT NULL,   -- JSON-encoded list
        gaps                      TEXT NOT NULL,   -- JSON-encoded list
        one_specific_improvement  TEXT NOT NULL,
        PRIMARY KEY (question_id, user_id, attempt_number)
    )

The composite primary key mirrors the de-duplication behavior the .jsonl
version implements manually (save_session() rewrites the file with the
old record for that key dropped before appending the new one) -- SQLite
gives us that for free via INSERT ... ON CONFLICT DO UPDATE.
"""

from __future__ import annotations

import json
import sqlite3
from datetime import date
from pathlib import Path
import os

DIMENSIONS = [
    "star_structure",
    "specificity",
    "relevance",
    "confidence_language",
    "conciseness",
]

_REQUIRED_SESSION_FIELDS = [
    "question_id",
    "user_id",
    "attempt_number",
    "scores",
    "strengths",
    "gaps",
    "one_specific_improvement",
    "transcript",
]

_DEFAULT_DB_PATH = "history/sessions.db"

_CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS sessions (
    question_id               TEXT NOT NULL,
    user_id                    TEXT NOT NULL,
    attempt_number              INTEGER NOT NULL,
    date                         TEXT,
    source_file                  TEXT,
    transcription_method         TEXT,
    transcript                    TEXT NOT NULL,
    scores                        TEXT NOT NULL,
    strengths                     TEXT NOT NULL,
    gaps                          TEXT NOT NULL,
    one_specific_improvement      TEXT NOT NULL,
    PRIMARY KEY (question_id, user_id, attempt_number)
)
"""

_UPSERT_SQL = """
INSERT INTO sessions (
    question_id, user_id, attempt_number, date, source_file,
    transcription_method, transcript, scores, strengths, gaps,
    one_specific_improvement
) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
ON CONFLICT (question_id, user_id, attempt_number) DO UPDATE SET
    date = excluded.date,
    source_file = excluded.source_file,
    transcription_method = excluded.transcription_method,
    transcript = excluded.transcript,
    scores = excluded.scores,
    strengths = excluded.strengths,
    gaps = excluded.gaps,
    one_specific_improvement = excluded.one_specific_improvement
"""

_SELECT_BY_USER_AND_QUESTION_SQL = """
SELECT question_id, user_id, attempt_number, date, source_file,
       transcription_method, transcript, scores, strengths, gaps,
       one_specific_improvement
FROM sessions
WHERE user_id = ? AND question_id = ?
ORDER BY attempt_number ASC
"""


# ------------------------------------------------------------------
# Connection helper
# ------------------------------------------------------------------

def _get_connection(db_path: str = _DEFAULT_DB_PATH) -> sqlite3.Connection:
    """Open a SQLite connection, ensuring the parent directory exists.

    Foreign-key-style integrity isn't needed here (single table), but we
    still set a busy_timeout so concurrent UI submissions from different
    Gradio tabs don't crash with "database is locked" under light load.
    """
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path, timeout=10.0)
    conn.execute("PRAGMA busy_timeout = 10000")
    return conn


# ------------------------------------------------------------------
# Row <-> dict conversion
# ------------------------------------------------------------------

def _row_to_record(row: tuple) -> dict:
    """Convert a raw sessions table row back into the same dict shape
    progression._load_history() has always returned, so callers can't
    tell the difference."""
    (
        question_id, user_id, attempt_number, rec_date, source_file,
        transcription_method, transcript, scores_json, strengths_json,
        gaps_json, one_specific_improvement,
    ) = row

    return {
        "question_id": question_id,
        "user_id": user_id,
        "attempt_number": attempt_number,
        "date": rec_date,
        "source_file": source_file,
        "transcription_method": transcription_method,
        "transcript": transcript,
        "scores": json.loads(scores_json),
        "strengths": json.loads(strengths_json),
        "gaps": json.loads(gaps_json),
        "one_specific_improvement": one_specific_improvement,
    }


def _record_to_row(record: dict) -> tuple:
    """Convert a session record dict (the same shape save_session() builds)
    into a tuple ready for the UPSERT statement."""
    return (
        record["question_id"],
        record["user_id"],
        record["attempt_number"],
        record.get("date") or date.today().isoformat(),
        record.get("source_file"),
        record.get("transcription_method") or "text_upload",
        record["transcript"],
        json.dumps(record["scores"]),
        json.dumps(record["strengths"]),
        json.dumps(record["gaps"]),
        record["one_specific_improvement"],
    )


# ------------------------------------------------------------------
# init_db: create table + one-time seed from history/*.jsonl
# ------------------------------------------------------------------

def init_db(db_path: str = _DEFAULT_DB_PATH, history_dir: str = "history") -> dict:
    """Create the sessions table if needed, and import any existing
    history/*.jsonl files as seed data -- but ONLY on first run.

    Importantly, this does NOT re-import on every call. Once the sessions
    table exists, init_db() is a pure no-op (besides the table-existence
    check): it will never overwrite live DB rows with stale .jsonl content.
    The .jsonl files are seed data for migrating a fresh install, not a
    source of truth that re-syncs on every app restart -- save_session_db()
    is the source of truth from that point on.

    Safe to call on every app startup regardless: CREATE TABLE IF NOT
    EXISTS makes table creation idempotent, and the seed-only-once guard
    means calling this repeatedly never reverts in-DB edits.

    Args:
        db_path: Path to the SQLite database file. Created if missing.
        history_dir: Directory to scan for legacy {user_id}.jsonl files.
                      Only read if the sessions table doesn't exist yet.

    Returns:
        dict with keys:
            db_path (str), tables_created (bool),
            files_imported (list[str]), records_imported (int).
            files_imported / records_imported are always empty/0 when
            tables_created is False, since seeding is skipped.
    """
    conn = _get_connection(db_path)
    try:
        cursor = conn.cursor()

        cursor.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='sessions'"
        )
        table_already_existed = cursor.fetchone() is not None

        cursor.execute(_CREATE_TABLE_SQL)
        conn.commit()

        files_imported: list[str] = []
        records_imported = 0

        # Only seed from .jsonl on a genuinely fresh database. If the table
        # already existed, save_session_db() is the source of truth and we
        # must not let stale .jsonl content clobber newer in-DB edits.
        if not table_already_existed:
            history_path = Path(history_dir)
            if history_path.exists():
                for jsonl_file in sorted(history_path.glob("*.jsonl")):
                    imported_here = _import_jsonl_file(cursor, jsonl_file)
                    if imported_here:
                        files_imported.append(jsonl_file.name)
                        records_imported += imported_here

            conn.commit()

        return {
            "db_path": db_path,
            "tables_created": not table_already_existed,
            "files_imported": files_imported,
            "records_imported": records_imported,
        }
    finally:
        conn.close()


def _import_jsonl_file(cursor: sqlite3.Cursor, jsonl_file: Path) -> int:
    """Read one {user_id}.jsonl file and upsert every valid record into
    the sessions table. Skips lines that don't parse or are missing
    required fields, rather than aborting the whole import.

    Returns the number of records successfully imported from this file.
    """
    imported = 0
    with jsonl_file.open("r") as f:
        for line_number, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue

            missing = [
                field for field in _REQUIRED_SESSION_FIELDS
                if record.get(field) is None
            ]
            if missing:
                continue

            cursor.execute(_UPSERT_SQL, _record_to_row(record))
            imported += 1

    return imported


# ------------------------------------------------------------------
# save_session_db: same contract as progression.save_session()
# ------------------------------------------------------------------

def save_session_db(
    session: dict,
    db_path: str = _DEFAULT_DB_PATH,
    history_dir: str = "history",
) -> bool:
    """Upsert a validated session record into the sessions table.

    Same validation and same return contract as progression.save_session():
    raises ValueError on malformed input, returns True/False for write
    success/failure rather than raising on storage-layer errors.

    Re-saving the same (question_id, user_id, attempt_number) overwrites
    the prior record -- identical de-duplication behavior to the .jsonl
    version, but enforced by the table's PRIMARY KEY instead of a manual
    read-filter-rewrite cycle.

    Args:
        session: A complete session dict conforming to schema/session.json.
                 Must pass validate_session() before calling this function.
        db_path: Path to the SQLite database file.
        history_dir: Unused by this function directly, but accepted so
                     call sites can swap save_session -> save_session_db
                     without changing their argument list.

    Returns:
        True if the record was written successfully, False otherwise.

    Raises:
        ValueError: if session is missing required fields, missing a
                    rubric dimension, or overall_score doesn't match the
                    sum of the five dimensions.
    """
    missing = [
        field for field in _REQUIRED_SESSION_FIELDS if session.get(field) is None
    ]
    if missing:
        raise ValueError(f"save_session_db: session missing required fields: {missing}")

    scores = session["scores"]
    missing_dims = [dim for dim in DIMENSIONS if dim not in scores]
    if missing_dims:
        raise ValueError(f"save_session_db: scores missing dimensions: {missing_dims}")
    if "overall_score" not in scores:
        raise ValueError("save_session_db: scores missing 'overall_score'")

    expected_overall = sum(scores[dim] for dim in DIMENSIONS)
    if scores["overall_score"] != expected_overall:
        raise ValueError(
            f"save_session_db: overall_score ({scores['overall_score']}) != "
            f"sum of dimensions ({expected_overall}). "
            "Run validate_session() before save_session_db()."
        )

    record = {
        "question_id": session["question_id"],
        "user_id": session["user_id"],
        "attempt_number": session["attempt_number"],
        "date": session.get("date") or date.today().isoformat(),
        "source_file": session.get("source_file"),
        "transcription_method": session.get("transcription_method") or "text_upload",
        "transcript": session["transcript"],
        "scores": scores,
        "strengths": session["strengths"],
        "gaps": session["gaps"],
        "one_specific_improvement": session["one_specific_improvement"],
    }

    try:
        conn = _get_connection(db_path)
        try:
            conn.execute(_UPSERT_SQL, _record_to_row(record))
            conn.commit()
            return True
        finally:
            conn.close()
    except sqlite3.Error:
        return False


# ------------------------------------------------------------------
# load_history_db: same contract as progression._load_history()
# ------------------------------------------------------------------

def load_history_db(
    user_id: str,
    question_id: str = "question_1",
    db_path: str = _DEFAULT_DB_PATH,
) -> list[dict]:
    """Load and sort all session records for a user, filtered by question_id.

    Identical interface and return shape to progression._load_history(),
    so any Skill currently doing:

        from skills.progression import _load_history
        records = _load_history(user_id, question_id, "history")

    can be swapped to:

        from skills.persistence import load_history_db
        records = load_history_db(user_id, question_id)

    with no other code changes required.

    Mirrors .jsonl behavior: if the DB file doesn't exist, returns [] without
    creating any files or directories (pure read-only, no side effects).

    Args:
        user_id: Progression track identifier (e.g. user_c).
        question_id: Filters records to a single question set.
        db_path: Path to the SQLite database file.

    Returns:
        list[dict], sorted by attempt_number ascending. Empty list if the
        DB file doesn't exist or the table hasn't been created yet, or the
        user has no records for this question_id (mirrors the .jsonl version's
        "file not found -> []" behavior rather than raising or creating files).
    """
    # Check if DB file exists before trying to connect; sqlite3.connect() creates
    # the file, but we want to match .jsonl behavior (no filesystem side effects).
    if not Path(db_path).exists():
        return []

    try:
        conn = _get_connection(db_path)
    except sqlite3.Error:
        return []

    try:
        cursor = conn.cursor()
        cursor.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='sessions'"
        )
        if cursor.fetchone() is None:
            return []

        cursor.execute(_SELECT_BY_USER_AND_QUESTION_SQL, (user_id, question_id))
        rows = cursor.fetchall()
        return [_row_to_record(row) for row in rows]
    finally:
        conn.close()


# ------------------------------------------------------------------
# delete_user_history: supports ui.py's reset_progress()
#
# Not part of the original spec's three named functions, but needed
# because reset_progress() used to just unlink history/{user_id}.jsonl --
# there's no equivalent filesystem-level reset for a SQLite table, so this
# is the SQLite-native replacement for that one piece of behavior.
# ------------------------------------------------------------------

def delete_user_history(
    user_id: str,
    db_path: str = _DEFAULT_DB_PATH,
) -> int:
    """Delete all rows for a given user_id across every question_id.

    Mirrors what unlinking history/{user_id}.jsonl used to do: a full
    reset of that user's progression track, regardless of question_id,
    so reset_progress() in ui.py keeps its existing "clear everything for
    this user" behavior after the SQLite swap.

    Like load_history_db, this is a read-style operation that should not
    create side effects: if the DB file doesn't exist, returns 0 without
    creating the file or directories.

    Args:
        user_id: Progression track identifier to clear (e.g. user_c).
        db_path: Path to the SQLite database file.

    Returns:
        Number of rows deleted. 0 if the DB file doesn't exist, or if the
        table doesn't exist or the user had no rows (mirrors the old
        "file not found" -> nothing-to-delete case rather than raising).
    """
    # Check if DB file exists before trying to connect; sqlite3.connect() creates
    # the file, but we want no filesystem side effects (matching .jsonl behavior).
    if not Path(db_path).exists():
        return 0

    try:
        conn = _get_connection(db_path)
    except sqlite3.Error:
        return 0

    try:
        cursor = conn.cursor()
        cursor.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='sessions'"
        )
        if cursor.fetchone() is None:
            return 0

        cursor.execute("DELETE FROM sessions WHERE user_id = ?", (user_id,))
        deleted = cursor.rowcount
        conn.commit()
        return deleted
    finally:
        conn.close()