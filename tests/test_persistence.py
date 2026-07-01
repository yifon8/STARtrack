"""
tests/test_persistence.py — verify the SQLite DB is created and behaves correctly.

Run with:
    pytest tests/test_persistence.py -v

Uses a temp directory for every test — never touches history/sessions.db.
"""

import json
import sqlite3
from pathlib import Path

import pytest

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from skills.persistence import (
    init_db,
    save_session_db,
    load_history_db,
    delete_user_history,
)


# ------------------------------------------------------------------
# Fixtures
# ------------------------------------------------------------------

@pytest.fixture
def db(tmp_path):
    """Fresh DB path and history dir in a temp directory per test."""
    return {
        "db_path":     str(tmp_path / "sessions.db"),
        "history_dir": str(tmp_path / "history"),
    }


@pytest.fixture
def seeded_jsonl(db):
    """Write two .jsonl records to the temp history dir, return the db dict."""
    hdir = Path(db["history_dir"])
    hdir.mkdir()
    records = [
        _make_session("user_a", 1, overall=5),
        _make_session("user_a", 2, overall=12),
    ]
    with (hdir / "user_a.jsonl").open("w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")
    return db


def _make_session(user_id, attempt_number, overall=10):
    dim = overall // 5
    return {
        "question_id":             "question_1",
        "user_id":                  user_id,
        "attempt_number":           attempt_number,
        "date":                      "2026-06-01",
        "source_file":               None,
        "transcription_method":      "text_upload",
        "transcript":                f"Transcript for attempt {attempt_number}.",
        "scores": {
            "star_structure": dim, "specificity": dim, "relevance": dim,
            "confidence_language": dim, "conciseness": dim,
            "overall_score": dim * 5,
        },
        "strengths":                ["Good attempt."],
        "gaps":                     ["Needs work."],
        "one_specific_improvement": "Be more specific.",
    }


# ------------------------------------------------------------------
# 1. DB file is created on disk
# ------------------------------------------------------------------

def test_db_file_is_created_on_disk(db):
    assert not Path(db["db_path"]).exists(), "should not exist before init_db"
    init_db(**db)
    assert Path(db["db_path"]).exists(), "sessions.db should exist after init_db"


# ------------------------------------------------------------------
# 2. sessions table has the right columns
# ------------------------------------------------------------------

def test_sessions_table_has_correct_schema(db):
    init_db(**db)
    conn = sqlite3.connect(db["db_path"])
    cursor = conn.execute("PRAGMA table_info(sessions)")
    columns = {row[1] for row in cursor.fetchall()}
    conn.close()

    expected = {
        "question_id", "user_id", "attempt_number",
        "date", "source_file", "transcription_method",
        "transcript", "scores", "strengths", "gaps",
        "one_specific_improvement",
    }
    assert expected == columns


# ------------------------------------------------------------------
# 3. Composite PRIMARY KEY is enforced
# ------------------------------------------------------------------

def test_composite_primary_key_is_enforced(db):
    init_db(**db)
    conn = sqlite3.connect(db["db_path"])
    pk_cols = set()
    for row in conn.execute("PRAGMA index_list(sessions)"):
        index_name = row[1]
        if "pk" in index_name.lower() or row[2]:   # row[2] = unique flag
            for col_row in conn.execute(f"PRAGMA index_info({index_name})"):
                pk_cols.add(col_row[2])
    conn.close()
    assert {"question_id", "user_id", "attempt_number"} == pk_cols


# ------------------------------------------------------------------
# 4. init_db seeds from .jsonl on first run
# ------------------------------------------------------------------

def test_init_db_seeds_from_jsonl_on_first_run(seeded_jsonl):
    result = init_db(**seeded_jsonl)
    assert result["tables_created"] is True
    assert result["records_imported"] == 2
    assert "user_a.jsonl" in result["files_imported"]

    records = load_history_db("user_a", db_path=seeded_jsonl["db_path"])
    assert len(records) == 2
    assert records[0]["attempt_number"] == 1
    assert records[1]["attempt_number"] == 2


# ------------------------------------------------------------------
# 5. init_db is idempotent — second call skips seeding
# ------------------------------------------------------------------

def test_init_db_does_not_reseed_on_second_call(seeded_jsonl):
    init_db(**seeded_jsonl)                          # first call — seeds

    # Modify attempt 1 through the live API
    updated = _make_session("user_a", 1, overall=15)
    save_session_db(updated, db_path=seeded_jsonl["db_path"])

    result2 = init_db(**seeded_jsonl)                # second call — no-op
    assert result2["tables_created"] is False
    assert result2["records_imported"] == 0

    # Live edit should still be intact, not reverted by re-seeding
    records = load_history_db("user_a", db_path=seeded_jsonl["db_path"])
    assert records[0]["scores"]["overall_score"] == 15


# ------------------------------------------------------------------
# 6. save_session_db writes and load_history_db reads back correctly
# ------------------------------------------------------------------

def test_save_and_load_roundtrip(db):
    init_db(**db)
    session = _make_session("user_b", 1, overall=10)
    ok = save_session_db(session, db_path=db["db_path"])
    assert ok is True

    records = load_history_db("user_b", db_path=db["db_path"])
    assert len(records) == 1
    r = records[0]
    assert r["user_id"] == "user_b"
    assert r["attempt_number"] == 1
    assert r["scores"]["overall_score"] == 10
    assert isinstance(r["scores"], dict)    # deserialized from JSON, not a string
    assert isinstance(r["strengths"], list)


# ------------------------------------------------------------------
# 7. Upsert — saving same (question_id, user_id, attempt_number) overwrites
# ------------------------------------------------------------------

def test_upsert_overwrites_not_duplicates(db):
    init_db(**db)
    save_session_db(_make_session("user_b", 1, overall=5),  db_path=db["db_path"])
    save_session_db(_make_session("user_b", 1, overall=15), db_path=db["db_path"])

    records = load_history_db("user_b", db_path=db["db_path"])
    assert len(records) == 1                              # no duplicate
    assert records[0]["scores"]["overall_score"] == 15   # latest value wins


# ------------------------------------------------------------------
# 8. load_history_db returns [] for unknown user — never raises
# ------------------------------------------------------------------

def test_load_returns_empty_for_unknown_user(db):
    init_db(**db)
    assert load_history_db("ghost_user", db_path=db["db_path"]) == []


# ------------------------------------------------------------------
# 9. load_history_db returns [] if DB doesn't exist yet — never raises
# ------------------------------------------------------------------

def test_load_returns_empty_before_init(db):
    assert load_history_db("user_a", db_path=db["db_path"]) == []


# ------------------------------------------------------------------
# 10. delete_user_history removes all rows for that user
# ------------------------------------------------------------------

def test_delete_user_history(db):
    init_db(**db)
    save_session_db(_make_session("user_c", 1), db_path=db["db_path"])
    save_session_db(_make_session("user_c", 2), db_path=db["db_path"])

    deleted = delete_user_history("user_c", db_path=db["db_path"])
    assert deleted == 2
    assert load_history_db("user_c", db_path=db["db_path"]) == []


# ------------------------------------------------------------------
# 11. delete_user_history returns 0 for user with no rows — never raises
# ------------------------------------------------------------------

def test_delete_user_with_no_rows_returns_zero(db):
    init_db(**db)
    assert delete_user_history("nobody", db_path=db["db_path"]) == 0