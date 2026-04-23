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

from groq import Groq

from config import (
    GROQ_API_KEY,
    CONVERSATION_HISTORY_TURNS,
    DB_PATH,
    LLM_MODEL,
)

log = logging.getLogger(__name__)


def llm_configured() -> bool:
    """True only when the API key looks real — not empty and not the
    placeholder value from .env.example."""
    if not GROQ_API_KEY:
        return False
    if GROQ_API_KEY.endswith("...") or GROQ_API_KEY == "gsk_...":
        return False
    return True

_SCHEMA = """
CREATE TABLE IF NOT EXISTS users (
    user_id            TEXT PRIMARY KEY,
    phone              TEXT UNIQUE NOT NULL,
    interests_json     TEXT NOT NULL DEFAULT '[]',
    persona_summary    TEXT NOT NULL DEFAULT '',
    onboarding_state   TEXT NOT NULL DEFAULT 'NEEDS_ONBOARDING',
    frequency          TEXT NOT NULL DEFAULT 'morning_9am',
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

CREATE TABLE IF NOT EXISTS indexed_articles (
    user_id    TEXT NOT NULL,
    article_id TEXT NOT NULL,
    indexed_at REAL NOT NULL,
    PRIMARY KEY(user_id, article_id)
);
"""


FREQUENCY_CHOICES: dict[str, str] = {
    "every_4h":     "Every 4 hours (08:00, 12:00, 16:00, 20:00 UTC)",
    "every_8h":     "Every 8 hours (00:00, 08:00, 16:00 UTC)",
    "morning_9am":  "Once a day — 09:00 UTC",
    "evening_6pm":  "Once a day — 18:00 UTC",
    "twice_daily":  "Twice a day — 09:00 and 18:00 UTC",
}


@dataclass
class User:
    user_id: str
    phone: str
    interests: List[str] = field(default_factory=list)
    persona_summary: str = ""
    onboarding_state: str = "NEEDS_ONBOARDING"
    frequency: str = "morning_9am"
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
        ("frequency",      "ALTER TABLE users ADD COLUMN frequency TEXT NOT NULL DEFAULT 'morning_9am'"),
        ("last_pushed_at", "ALTER TABLE users ADD COLUMN last_pushed_at REAL"),
    ):
        if col not in existing:
            conn.execute(ddl)


def _row_to_user(row: sqlite3.Row, history: List[dict]) -> User:
    return User(
        user_id=row["user_id"],
        phone=row["phone"],
        interests=json.loads(row["interests_json"]),
        persona_summary=row["persona_summary"],
        onboarding_state=row["onboarding_state"],
        frequency=row["frequency"] if "frequency" in row.keys() else "morning_9am",
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


def set_frequency(user_id: str, frequency: str) -> None:
    if frequency not in FREQUENCY_CHOICES:
        raise ValueError(f"unknown frequency: {frequency}")
    with _connect() as conn:
        conn.execute(
            "UPDATE users SET frequency = ?, updated_at = ? WHERE user_id = ?",
            (frequency, time.time(), user_id),
        )


def mark_pushed(user_id: str, when: Optional[float] = None) -> None:
    with _connect() as conn:
        conn.execute(
            "UPDATE users SET last_pushed_at = ?, updated_at = ? WHERE user_id = ?",
            (when or time.time(), time.time(), user_id),
        )


_FREQUENCY_HOURS: dict[str, set[int]] = {
    "every_4h":    {8, 12, 16, 20},
    "every_8h":    {0, 8, 16},
    "morning_9am": {9},
    "evening_6pm": {18},
    "twice_daily": {9, 18},
}

_MIN_SECONDS_BETWEEN_PUSHES = 3 * 60 * 60  # debounce against cron retries


def is_push_due(user: User, now: float) -> bool:
    """Return True if the user's frequency window matches the current UTC
    hour AND it's been long enough since the last push."""
    if user.onboarding_state != "DONE":
        return False
    hours = _FREQUENCY_HOURS.get(user.frequency, {9})
    import datetime
    now_hour = datetime.datetime.utcfromtimestamp(now).hour
    if now_hour not in hours:
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
        conn.execute("DELETE FROM indexed_articles WHERE user_id = ?", (user_id,))
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


def already_sent_ids(user_id: str) -> set[str]:
    with _connect() as conn:
        rows = conn.execute(
            "SELECT article_id FROM sent_articles WHERE user_id = ?", (user_id,)
        ).fetchall()
        return {r["article_id"] for r in rows}


_INDEXED_ARTICLE_TTL = 7 * 24 * 60 * 60  # 7 days


def mark_articles_indexed(user_id: str, article_ids: Iterable[str]) -> None:
    now = time.time()
    rows = [(user_id, aid, now) for aid in article_ids]
    if not rows:
        return
    with _connect() as conn:
        conn.executemany(
            "INSERT OR IGNORE INTO indexed_articles (user_id, article_id, indexed_at) "
            "VALUES (?, ?, ?)",
            rows,
        )


def already_indexed_ids(user_id: str, max_age_seconds: float = _INDEXED_ARTICLE_TTL) -> set[str]:
    cutoff = time.time() - max_age_seconds
    with _connect() as conn:
        rows = conn.execute(
            "SELECT article_id FROM indexed_articles WHERE user_id = ? AND indexed_at >= ?",
            (user_id, cutoff),
        ).fetchall()
        return {r["article_id"] for r in rows}


def purge_old_indexed_articles(user_id: str, max_age_seconds: float = _INDEXED_ARTICLE_TTL) -> int:
    """Remove indexed_articles rows older than max_age_seconds. Returns rows deleted."""
    cutoff = time.time() - max_age_seconds
    with _connect() as conn:
        cur = conn.execute(
            "DELETE FROM indexed_articles WHERE user_id = ? AND indexed_at < ?",
            (user_id, cutoff),
        )
        return cur.rowcount


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
    try:
        client = Groq(api_key=GROQ_API_KEY)
        resp = client.chat.completions.create(
            model=LLM_MODEL,
            max_tokens=300,
            messages=[{
                "role": "user",
                "content": _PERSONA_PROMPT.format(interests=", ".join(interests)),
            }],
        )
        return resp.choices[0].message.content.strip()
    except Exception as e:
        log.warning("persona summary LLM call failed (%s); using keyword fallback", e)
        return fallback
