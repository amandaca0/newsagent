"""Signup web UI — serves the React SPA and a JSON signup endpoint.

The SPA (gateway/static/signup.html) handles all rendering and step
navigation client-side. This module only:
  - GET /         -> returns the SPA HTML
  - POST /signup  -> accepts JSON, creates/updates the user, returns JSON

Auth is intentionally not implemented — single-tester setup only.
"""
from __future__ import annotations

import logging
import os
import re

from flask import Blueprint, jsonify, request, send_from_directory

from core.user_profile import (
    FREQUENCY_CHOICES,
    generate_persona_summary,
    get_or_create_user,
    get_user_profile,
    set_frequency,
    set_interests,
    set_onboarding_state,
    set_persona_summary,
)

log = logging.getLogger(__name__)

bp = Blueprint("web", __name__)

_STATIC_DIR = os.path.join(os.path.dirname(__file__), "static")
_PHONE_RE = re.compile(r"\D+")


def normalize_phone(raw: str) -> str | None:
    """Coerce user-entered phone to E.164. Returns None if unparseable."""
    if not raw:
        return None
    had_plus = raw.strip().startswith("+")
    digits = _PHONE_RE.sub("", raw)
    if not digits:
        return None
    if had_plus:
        return f"+{digits}" if 7 <= len(digits) <= 15 else None
    if len(digits) == 10:
        return f"+1{digits}"
    if len(digits) == 11 and digits.startswith("1"):
        return f"+{digits}"
    return None


def _parse_topics(raw) -> list[str]:
    if raw is None:
        return []
    if isinstance(raw, list):
        return [str(t).strip() for t in raw if str(t).strip()]
    parts = re.split(r"[,\n;]", str(raw))
    return [p.strip() for p in parts if p.strip()]


@bp.get("/")
def signup_page():
    return send_from_directory(_STATIC_DIR, "signup.html")


@bp.post("/signup")
def signup_submit():
    data = request.get_json(silent=True)
    if data is None:
        data = request.form  # fallback for plain form posts

    phone = normalize_phone(data.get("phone", ""))
    topics = _parse_topics(data.get("topics", ""))
    frequency = data.get("frequency", "morning_9am")

    errors = []
    if phone is None:
        errors.append("That phone number didn't look valid.")
    if not topics:
        errors.append("Add at least one topic.")
    if frequency not in FREQUENCY_CHOICES:
        errors.append("Pick a valid frequency.")

    if errors:
        return jsonify(ok=False, error=" ".join(errors)), 400

    user = get_or_create_user(phone)
    set_interests(user.user_id, topics)
    set_persona_summary(user.user_id, generate_persona_summary(topics))
    set_frequency(user.user_id, frequency)
    set_onboarding_state(user.user_id, "DONE")

    saved = get_user_profile(user.user_id)
    payload = saved.to_dict()
    # conversation_history is verbose and not useful to the UI
    payload.pop("conversation_history", None)
    payload["frequency_label"] = FREQUENCY_CHOICES.get(frequency, frequency)
    return jsonify(ok=True, user=payload)
