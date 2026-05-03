"""User profile storage and management.

Single source of truth for user state: interests, LLM-generated persona
summary, rolling conversation history, and sent-article dedupe set. All
other modules consume User objects via get_user_profile().
"""
from __future__ import annotations

import json
import logging
import os
import sqlite3
import time
import uuid
from dataclasses import asdict, dataclass, field
from typing import Iterable, List, Optional

from config import (
    CONVERSATION_HISTORY_TURNS,
    DB_PATH,
)
from core.llm import active_model, complete, llm_configured  # re-exported

log = logging.getLogger(__name__)

__all__ = ["llm_configured"]  # keep callers' `from core.user_profile import llm_configured` working

_SCHEMA = """
CREATE TABLE IF NOT EXISTS users (
    user_id            TEXT PRIMARY KEY,
    phone              TEXT UNIQUE NOT NULL,
    name               TEXT NOT NULL DEFAULT '',
    interests_json     TEXT NOT NULL DEFAULT '[]',
    persona_summary    TEXT NOT NULL DEFAULT '',
    onboarding_state   TEXT NOT NULL DEFAULT 'NEEDS_ONBOARDING',
    frequency          TEXT NOT NULL DEFAULT 'morning_9am',
    custom_push_hour   INTEGER,  -- 0..23 UTC; primary fire time (or starting hour for every_4h/8h)
    custom_push_minute INTEGER,  -- 0..59 UTC; primary fire minute
    custom_push_hour_2 INTEGER,  -- 0..23 UTC; second fire time (twice_daily only)
    custom_push_minute_2 INTEGER, -- 0..59 UTC; second fire minute (twice_daily only)
    last_pushed_at     REAL,
    created_at         REAL NOT NULL,
    updated_at         REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS messages (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id    TEXT NOT NULL,
    role       TEXT NOT NULL,       -- 'user' or 'assistant'
    content    TEXT NOT NULL,
    created_at REAL NOT NULL,
    FOREIGN KEY(user_id) REFERENCES users(user_id)
);
CREATE INDEX IF NOT EXISTS idx_messages_user_time
    ON messages(user_id, created_at DESC);

CREATE TABLE IF NOT EXISTS sent_articles (
    user_id    TEXT NOT NULL,
    article_id TEXT NOT NULL,
    sent_at    REAL NOT NULL,
    PRIMARY KEY(user_id, article_id)
);
"""


FREQUENCY_CHOICES: dict[str, str] = {
    "every_4h":     "Every 4 hours",
    "every_8h":     "Every 8 hours",
    "morning_9am":  "Morning briefing",
    "evening_6pm":  "Evening wrap-up",
    "twice_daily":  "Morning + evening",
    "custom_daily": "Custom time",
}


@dataclass
class User:
    user_id: str
    phone: str
    name: str = ""
    interests: List[str] = field(default_factory=list)
    persona_summary: str = ""
    onboarding_state: str = "NEEDS_ONBOARDING"
    frequency: str = "morning_9am"
    custom_push_hour: Optional[int] = None
    custom_push_minute: Optional[int] = None
    custom_push_hour_2: Optional[int] = None
    custom_push_minute_2: Optional[int] = None
    last_pushed_at: Optional[float] = None
    conversation_history: List[dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


def _connect() -> sqlite3.Connection:
    os.makedirs(os.path.dirname(DB_PATH) or ".", exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def init_db() -> None:
    with _connect() as conn:
        conn.executescript(_SCHEMA)
        _migrate(conn)


def _migrate(conn: sqlite3.Connection) -> None:
    """Idempotent column adds for DBs created before a column existed."""
    existing = {r["name"] for r in conn.execute("PRAGMA table_info(users)").fetchall()}
    for col, ddl in (
        ("frequency",          "ALTER TABLE users ADD COLUMN frequency TEXT NOT NULL DEFAULT 'morning_9am'"),
        ("last_pushed_at",     "ALTER TABLE users ADD COLUMN last_pushed_at REAL"),
        ("custom_push_hour",     "ALTER TABLE users ADD COLUMN custom_push_hour INTEGER"),
        ("custom_push_minute",   "ALTER TABLE users ADD COLUMN custom_push_minute INTEGER"),
        ("custom_push_hour_2",   "ALTER TABLE users ADD COLUMN custom_push_hour_2 INTEGER"),
        ("custom_push_minute_2", "ALTER TABLE users ADD COLUMN custom_push_minute_2 INTEGER"),
        ("name",                 "ALTER TABLE users ADD COLUMN name TEXT NOT NULL DEFAULT ''"),
    ):
        if col not in existing:
            conn.execute(ddl)


def _row_to_user(row: sqlite3.Row, history: List[dict]) -> User:
    return User(
        user_id=row["user_id"],
        phone=row["phone"],
        name=row["name"] if "name" in row.keys() else "",
        interests=json.loads(row["interests_json"]),
        persona_summary=row["persona_summary"],
        onboarding_state=row["onboarding_state"],
        frequency=row["frequency"] if "frequency" in row.keys() else "morning_9am",
        custom_push_hour=row["custom_push_hour"] if "custom_push_hour" in row.keys() else None,
        custom_push_minute=row["custom_push_minute"] if "custom_push_minute" in row.keys() else None,
        custom_push_hour_2=row["custom_push_hour_2"] if "custom_push_hour_2" in row.keys() else None,
        custom_push_minute_2=row["custom_push_minute_2"] if "custom_push_minute_2" in row.keys() else None,
        last_pushed_at=row["last_pushed_at"] if "last_pushed_at" in row.keys() else None,
        conversation_history=history,
    )


def _fetch_history(conn: sqlite3.Connection, user_id: str) -> List[dict]:
    rows = conn.execute(
        "SELECT role, content, created_at FROM messages "
        "WHERE user_id = ? ORDER BY created_at DESC LIMIT ?",
        (user_id, CONVERSATION_HISTORY_TURNS * 2),
    ).fetchall()
    return [dict(r) for r in reversed(rows)]


def get_user_profile(user_id: str) -> Optional[User]:
    with _connect() as conn:
        row = conn.execute(
            "SELECT * FROM users WHERE user_id = ?", (user_id,)
        ).fetchone()
        if row is None:
            return None
        return _row_to_user(row, _fetch_history(conn, user_id))


def get_user_by_phone(phone: str) -> Optional[User]:
    with _connect() as conn:
        row = conn.execute(
            "SELECT * FROM users WHERE phone = ?", (phone,)
        ).fetchone()
        if row is None:
            return None
        return _row_to_user(row, _fetch_history(conn, row["user_id"]))


def get_or_create_user(phone: str) -> User:
    existing = get_user_by_phone(phone)
    if existing is not None:
        return existing
    now = time.time()
    user_id = f"u_{uuid.uuid4().hex[:12]}"
    with _connect() as conn:
        conn.execute(
            "INSERT INTO users (user_id, phone, created_at, updated_at) "
            "VALUES (?, ?, ?, ?)",
            (user_id, phone, now, now),
        )
    return User(user_id=user_id, phone=phone)


def list_users() -> List[User]:
    with _connect() as conn:
        rows = conn.execute("SELECT * FROM users").fetchall()
        return [_row_to_user(r, _fetch_history(conn, r["user_id"])) for r in rows]


def set_name(user_id: str, name: str) -> None:
    with _connect() as conn:
        conn.execute(
            "UPDATE users SET name = ?, updated_at = ? WHERE user_id = ?",
            ((name or "").strip(), time.time(), user_id),
        )


def set_interests(user_id: str, interests: Iterable[str]) -> None:
    cleaned = [i.strip() for i in interests if i and i.strip()]
    with _connect() as conn:
        conn.execute(
            "UPDATE users SET interests_json = ?, updated_at = ? WHERE user_id = ?",
            (json.dumps(cleaned), time.time(), user_id),
        )


def set_persona_summary(user_id: str, summary: str) -> None:
    with _connect() as conn:
        conn.execute(
            "UPDATE users SET persona_summary = ?, updated_at = ? WHERE user_id = ?",
            (summary, time.time(), user_id),
        )


def set_onboarding_state(user_id: str, state: str) -> None:
    with _connect() as conn:
        conn.execute(
            "UPDATE users SET onboarding_state = ?, updated_at = ? WHERE user_id = ?",
            (state, time.time(), user_id),
        )


def _validate_hm(h, m, label: str) -> tuple[int, int]:
    if h is None or m is None:
        raise ValueError(f"{label} requires hour and minute")
    h, m = int(h), int(m)
    if not (0 <= h <= 23):
        raise ValueError(f"{label} hour must be 0..23")
    if not (0 <= m <= 59):
        raise ValueError(f"{label} minute must be 0..59")
    return h, m


def set_frequency(
    user_id: str,
    frequency: str,
    custom_push_hour: Optional[int] = None,
    custom_push_minute: Optional[int] = None,
    custom_push_hour_2: Optional[int] = None,
    custom_push_minute_2: Optional[int] = None,
) -> None:
    if frequency not in FREQUENCY_CHOICES:
        raise ValueError(f"unknown frequency: {frequency}")

    h2 = m2 = None
    if frequency == "twice_daily":
        custom_push_hour, custom_push_minute = _validate_hm(
            custom_push_hour, custom_push_minute, "twice_daily morning")
        h2, m2 = _validate_hm(
            custom_push_hour_2, custom_push_minute_2, "twice_daily evening")
    elif frequency in ("every_4h", "every_8h", "morning_9am", "evening_6pm", "custom_daily"):
        custom_push_hour, custom_push_minute = _validate_hm(
            custom_push_hour, custom_push_minute, frequency)

    with _connect() as conn:
        conn.execute(
            "UPDATE users SET frequency = ?, custom_push_hour = ?, custom_push_minute = ?, "
            "custom_push_hour_2 = ?, custom_push_minute_2 = ?, updated_at = ? WHERE user_id = ?",
            (frequency, custom_push_hour, custom_push_minute, h2, m2, time.time(), user_id),
        )


def mark_pushed(user_id: str, when: Optional[float] = None) -> None:
    with _connect() as conn:
        conn.execute(
            "UPDATE users SET last_pushed_at = ?, updated_at = ? WHERE user_id = ?",
            (when or time.time(), time.time(), user_id),
        )


_MIN_SECONDS_BETWEEN_PUSHES = 3 * 60 * 60  # debounce against cron retries
_MINUTE_SLACK = 1  # tolerate ±1 minute of cron drift

# Defaults for users created before per-frequency custom times existed; mirror
# the old _FREQUENCY_HOURS so they keep firing at the same UTC time.
_DEFAULT_HOUR = {
    "every_4h": 8, "every_8h": 0,
    "morning_9am": 9, "evening_6pm": 18,
    "twice_daily": 9,
}
_DEFAULT_HOUR_2 = {"twice_daily": 18}


def _target_minutes_for(user: User) -> set[int]:
    """Return the set of UTC minute-of-day values the user should be pushed at.

    One minute-of-day = hour * 60 + minute, so a single scalar per target.
    Each frequency reads its time(s) from the user's stored custom values,
    falling back to legacy defaults when those columns are NULL.
    """
    h, m   = user.custom_push_hour, user.custom_push_minute
    h2, m2 = user.custom_push_hour_2, user.custom_push_minute_2
    f = user.frequency

    if f == "custom_daily":
        if h is None or m is None:
            return set()
        return {int(h) * 60 + int(m)}

    if f in ("morning_9am", "evening_6pm"):
        h_eff = int(h) if h is not None else _DEFAULT_HOUR[f]
        m_eff = int(m) if m is not None else 0
        return {h_eff * 60 + m_eff}

    if f == "twice_daily":
        morn = (int(h)  if h  is not None else _DEFAULT_HOUR[f])   * 60 + (int(m)  if m  is not None else 0)
        eve  = (int(h2) if h2 is not None else _DEFAULT_HOUR_2[f]) * 60 + (int(m2) if m2 is not None else 0)
        return {morn, eve}

    if f == "every_4h":
        start = (int(h) if h is not None else _DEFAULT_HOUR[f]) * 60 + (int(m) if m is not None else 0)
        return {(start + i * 240) % 1440 for i in range(4)}

    if f == "every_8h":
        start = (int(h) if h is not None else _DEFAULT_HOUR[f]) * 60 + (int(m) if m is not None else 0)
        return {(start + i * 480) % 1440 for i in range(3)}

    return {9 * 60}


def is_push_due(user: User, now: float) -> bool:
    """Return True when "now" is within _MINUTE_SLACK of one of the user's
    scheduled minute-of-day targets AND the debounce window has elapsed."""
    if user.onboarding_state != "DONE":
        return False
    targets = _target_minutes_for(user)
    if not targets:
        return False
    import datetime
    n = datetime.datetime.utcfromtimestamp(now)
    now_min = n.hour * 60 + n.minute
    within_slack = any(
        abs(now_min - t) <= _MINUTE_SLACK or abs(now_min - t) >= (24 * 60 - _MINUTE_SLACK)
        for t in targets
    )
    if not within_slack:
        return False
    if user.last_pushed_at is None:
        return True
    return (now - user.last_pushed_at) >= _MIN_SECONDS_BETWEEN_PUSHES


def delete_user(user_id: str) -> bool:
    """Remove a user and all their associated rows. Returns True if something
    was deleted. Does NOT touch the user's Chroma collection — callers should
    use rag_engine.drop_user_collection(user_id) for that."""
    with _connect() as conn:
        conn.execute("DELETE FROM messages WHERE user_id = ?", (user_id,))
        conn.execute("DELETE FROM sent_articles WHERE user_id = ?", (user_id,))
        cur = conn.execute("DELETE FROM users WHERE user_id = ?", (user_id,))
        return cur.rowcount > 0


def append_message(user_id: str, role: str, content: str) -> None:
    if role not in {"user", "assistant"}:
        raise ValueError(f"invalid role: {role}")
    with _connect() as conn:
        conn.execute(
            "INSERT INTO messages (user_id, role, content, created_at) "
            "VALUES (?, ?, ?, ?)",
            (user_id, role, content, time.time()),
        )


def mark_articles_sent(user_id: str, article_ids: Iterable[str]) -> None:
    now = time.time()
    rows = [(user_id, aid, now) for aid in article_ids]
    if not rows:
        return
    with _connect() as conn:
        conn.executemany(
            "INSERT OR IGNORE INTO sent_articles (user_id, article_id, sent_at) "
            "VALUES (?, ?, ?)",
            rows,
        )


def clear_sent_articles(user_id: str) -> None:
    with _connect() as conn:
        conn.execute("DELETE FROM sent_articles WHERE user_id = ?", (user_id,))


def already_sent_ids(user_id: str) -> set[str]:
    with _connect() as conn:
        rows = conn.execute(
            "SELECT article_id FROM sent_articles WHERE user_id = ?", (user_id,)
        ).fetchall()
        return {r["article_id"] for r in rows}


def latest_digest_article_ids(user_id: str) -> List[str]:
    """Article ids from this user's most recent digest — the rows whose
    sent_at matches MAX(sent_at) for this user. All articles in a single
    digest share one timestamp because mark_articles_sent stamps them
    together. Order matches insertion (which matches the digest display
    order)."""
    with _connect() as conn:
        rows = conn.execute(
            "SELECT article_id FROM sent_articles "
            "WHERE user_id = ? AND sent_at = ("
            "  SELECT MAX(sent_at) FROM sent_articles WHERE user_id = ?"
            ") ORDER BY rowid",
            (user_id, user_id),
        ).fetchall()
    return [r["article_id"] for r in rows]


_PERSONA_PROMPT = """\
You are building an embedding-friendly interest profile for a news recommendation system.

Given a user's stated interests, write a 2-3 sentence dense summary capturing:
- specific topics and subtopics they care about
- the angle or framing they prefer (policy? technical? consumer?)
- topics they would likely NOT want (if inferrable)

Interests: {interests}

Output only the summary paragraph, no preamble."""


def generate_persona_summary(interests: List[str]) -> str:
    """Ask the LLM to expand raw interests into a dense semantic summary.

    The summary is what we embed and what the ranker compares articles against,
    so it needs to be richer than a keyword list.
    """
    if not interests:
        return ""
    fallback = "User interested in: " + ", ".join(interests)
    if not llm_configured():
        return fallback
    prompt = _PERSONA_PROMPT.format(interests=", ".join(interests))
    try:
        summary = complete(prompt, max_tokens=300, purpose="chat").strip()
        from core.conv_log import log_event
        log_event(
            "llm_call",
            purpose="persona_summary", model=active_model("chat"),
            prompt=prompt, response=summary,
        )
        return summary
    except Exception as e:
        log.warning("persona summary LLM call failed (%s); using keyword fallback", e)
        return fallback
