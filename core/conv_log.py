"""Append-only conversation log.

Two outputs per event:
  1. ``data/logs/conversations.jsonl`` — every event as one JSON line, suitable
     for machine analysis or `jq`-style filtering.
  2. ``data/logs/<phone>.log`` — human-readable per-user transcript that
     includes the full prompts sent to the LLM and the raw responses.

Event kinds:
  - inbound_message    user texted us (text)
  - outbound_message   we texted user (text, purpose)
  - llm_call           we called the LLM (purpose, model, prompt, response)
  - proactive_digest   the proactive digest was prepared (articles, text)

All callers pass `phone` when they have one; events without a phone go only
to the master JSONL.
"""
from __future__ import annotations

import json
import logging
import os
import re
import time
from typing import Any, Optional

from config import DB_PATH

log = logging.getLogger(__name__)

_LOG_DIR = os.path.join(os.path.dirname(DB_PATH) or ".", "logs")
_MASTER_LOG = os.path.join(_LOG_DIR, "conversations.jsonl")


def _safe_name(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9_+.-]", "_", s) or "unknown"


def _ensure_dir() -> None:
    os.makedirs(_LOG_DIR, exist_ok=True)


def per_user_path(phone: str) -> str:
    return os.path.join(_LOG_DIR, f"{_safe_name(phone)}.log")


def log_event(
    kind: str,
    *,
    user_id: Optional[str] = None,
    phone: Optional[str] = None,
    **payload: Any,
) -> None:
    """Append one event to the master JSONL and (if phone given) the user log."""
    try:
        _ensure_dir()
        ts = time.time()
        entry = {"ts": ts, "kind": kind, "user_id": user_id, "phone": phone, **payload}

        with open(_MASTER_LOG, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

        if phone:
            ts_str = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(ts))
            with open(per_user_path(phone), "a", encoding="utf-8") as f:
                f.write(_format_readable(ts_str, kind, payload))
    except Exception:
        # Logging must never crash the request path.
        log.exception("conv_log.log_event failed")


def _hr() -> str:
    return "─" * 78 + "\n"


def _format_readable(ts: str, kind: str, p: dict) -> str:
    if kind == "inbound_message":
        return f"\n[{ts}] USER → {p.get('text','').strip()}\n"
    if kind == "outbound_message":
        purpose = p.get("purpose", "reply")
        return f"\n[{ts}] AGENT ({purpose}) → {p.get('text','').strip()}\n"
    if kind == "llm_call":
        purpose = p.get("purpose", "?")
        model = p.get("model", "?")
        prompt = p.get("prompt", "")
        response = p.get("response", "")
        out = []
        out.append(_hr())
        out.append(f"[{ts}] LLM CALL · purpose={purpose} · model={model}\n")
        out.append("--- PROMPT ---\n")
        out.append(prompt.rstrip() + "\n")
        out.append("--- RESPONSE ---\n")
        out.append(response.rstrip() + "\n")
        out.append(_hr())
        return "".join(out)
    if kind == "proactive_digest":
        articles = p.get("articles", []) or []
        out = [f"\n[{ts}] DIGEST PREPARED · {len(articles)} article(s)\n"]
        for a in articles:
            out.append(f"  - {a.get('title','(no title)')} ({a.get('source','?')})\n")
        return "".join(out)
    # Unknown kind: dump the payload as a single line.
    return f"\n[{ts}] {kind.upper()} {json.dumps(p, ensure_ascii=False)}\n"


def read_user_log(phone: str) -> str:
    path = per_user_path(phone)
    if not os.path.exists(path):
        return ""
    with open(path, "r", encoding="utf-8") as f:
        return f.read()
