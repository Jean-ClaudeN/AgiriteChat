"""
feedback.py — SQLite logging for queries, responses, and farmer feedback.

Why: You can't improve what you don't measure. Every interaction is logged
with the full trace (question, retrieved docs, final response, thumbs
up/down). This becomes your evaluation dataset for iterating on the KB
and the agent.
"""

import json
import sqlite3
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Any

DB_PATH = Path(__file__).parent / "agiritechat_feedback.db"


def _conn():
    c = sqlite3.connect(DB_PATH)
    c.execute("PRAGMA journal_mode = WAL")
    return c


def init_db() -> None:
    with _conn() as c:
        c.execute("""
            CREATE TABLE IF NOT EXISTS interactions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                session_id TEXT,
                user_question TEXT NOT NULL,
                crop_hint TEXT,
                category_hint TEXT,
                classified_crop TEXT,
                top_score REAL,
                matches_json TEXT,
                response_json TEXT,
                trace_json TEXT,
                image_source TEXT,
                needs_escalation INTEGER,
                feedback INTEGER DEFAULT NULL
            )
        """)


def log_interaction(session_id: str, state: Dict[str, Any]) -> int:
    init_db()
    with _conn() as c:
        cur = c.execute(
            """INSERT INTO interactions
               (timestamp, session_id, user_question, crop_hint, category_hint,
                classified_crop, top_score, matches_json, response_json,
                trace_json, image_source, needs_escalation)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                datetime.utcnow().isoformat(),
                session_id,
                state.get("user_question", ""),
                state.get("crop_hint", ""),
                state.get("category_hint", ""),
                state.get("classified_crop", ""),
                float(state.get("top_score", 0.0)),
                json.dumps(state.get("matches", [])),
                json.dumps(state.get("response", {})),
                json.dumps(state.get("trace", [])),
                state.get("image_source", "none"),
                1 if state.get("needs_escalation") else 0,
            ),
        )
        return cur.lastrowid


def record_feedback(interaction_id: int, thumbs: int) -> None:
    """thumbs: 1 for up, -1 for down."""
    with _conn() as c:
        c.execute(
            "UPDATE interactions SET feedback = ? WHERE id = ?",
            (thumbs, interaction_id),
        )


def recent_stats() -> Dict[str, Any]:
    init_db()
    with _conn() as c:
        total = c.execute("SELECT COUNT(*) FROM interactions").fetchone()[0]
        ups = c.execute("SELECT COUNT(*) FROM interactions WHERE feedback = 1").fetchone()[0]
        downs = c.execute("SELECT COUNT(*) FROM interactions WHERE feedback = -1").fetchone()[0]
        escalations = c.execute("SELECT COUNT(*) FROM interactions WHERE needs_escalation = 1").fetchone()[0]
        return {
            "total": total,
            "thumbs_up": ups,
            "thumbs_down": downs,
            "escalations": escalations,
        }
