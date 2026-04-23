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
    User,
    delete_user,
    generate_persona_summary,
    get_or_create_user,
    get_user_profile,
    list_users,
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


@bp.get("/users")
def users_page():
    return send_from_directory(_STATIC_DIR, "users.html")


def _coerce_int_range(raw, lo: int, hi: int) -> int | None:
    try:
        v = int(raw)
    except (TypeError, ValueError):
        return None
    return v if lo <= v <= hi else None


def _frequency_label(frequency: str, custom_hour: int | None, custom_minute: int | None) -> str:
    if frequency == "custom_daily" and custom_hour is not None and custom_minute is not None:
        return f"Once a day — {custom_hour:02d}:{custom_minute:02d} UTC"
    return FREQUENCY_CHOICES.get(frequency, frequency)


def _user_payload(u: User) -> dict:
    d = u.to_dict()
    d.pop("conversation_history", None)
    d["frequency_label"] = _frequency_label(u.frequency, u.custom_push_hour, u.custom_push_minute)
    return d


def _compose_welcome(u: User) -> str:
    topics = ", ".join(u.interests) if u.interests else "(no topics set)"
    schedule = _frequency_label(u.frequency, u.custom_push_hour, u.custom_push_minute)
    return (
        "Welcome to NewsAgent!\n\n"
        f"I'll send you personalized news digests covering: {topics}\n\n"
        f"Schedule: {schedule}\n\n"
        "Reply to any digest with a follow-up question and I'll answer using "
        "the articles I've sent you."
    )


def _send_welcome(u: User) -> None:
    """Fire off a welcome iMessage. Failures are logged, not raised, so that
    a flaky BlueBubbles server doesn't break the signup flow."""
    try:
        from gateway.bluebubbles import send_bluebubbles
        send_bluebubbles(u.phone, _compose_welcome(u))
    except Exception:
        log.exception("welcome message failed for %s", u.phone)


@bp.post("/signup")
def signup_submit():
    data = request.get_json(silent=True)
    if data is None:
        data = request.form  # fallback for plain form posts

    phone = normalize_phone(data.get("phone", ""))
    topics = _parse_topics(data.get("topics", ""))
    frequency = data.get("frequency", "morning_9am")
    custom_hour = None
    custom_minute = None
    if frequency == "custom_daily":
        custom_hour = _coerce_int_range(data.get("custom_hour"), 0, 23)
        custom_minute = _coerce_int_range(data.get("custom_minute"), 0, 59)

    errors = []
    if phone is None:
        errors.append("That phone number didn't look valid.")
    if not topics:
        errors.append("Add at least one topic.")
    if frequency not in FREQUENCY_CHOICES:
        errors.append("Pick a valid frequency.")
    if frequency == "custom_daily" and (custom_hour is None or custom_minute is None):
        errors.append("Pick a valid custom time (HH:MM).")

    if errors:
        return jsonify(ok=False, error=" ".join(errors)), 400

    user = get_or_create_user(phone)
    set_interests(user.user_id, topics)
    set_persona_summary(user.user_id, generate_persona_summary(topics))
    set_frequency(
        user.user_id, frequency,
        custom_push_hour=custom_hour,
        custom_push_minute=custom_minute,
    )
    set_onboarding_state(user.user_id, "DONE")

    saved = get_user_profile(user.user_id)
    _send_welcome(saved)
    return jsonify(ok=True, user=_user_payload(saved))


# ---------- admin API ----------

@bp.get("/api/users")
def api_list_users():
    users = list_users()
    return jsonify(users=[_user_payload(u) for u in users])


@bp.patch("/api/users/<user_id>")
def api_update_user(user_id: str):
    data = request.get_json(silent=True) or {}
    existing = get_user_profile(user_id)
    if existing is None:
        return jsonify(ok=False, error="user not found"), 404

    errors: list[str] = []

    # topics — regenerate persona summary when they change
    if "topics" in data:
        topics = _parse_topics(data["topics"])
        if not topics:
            errors.append("Topics cannot be empty.")
        else:
            set_interests(user_id, topics)
            set_persona_summary(user_id, generate_persona_summary(topics))

    # frequency (and custom time if applicable)
    if "frequency" in data:
        frequency = data["frequency"]
        if frequency not in FREQUENCY_CHOICES:
            errors.append("Pick a valid frequency.")
        else:
            custom_hour = None
            custom_minute = None
            if frequency == "custom_daily":
                custom_hour = _coerce_int_range(data.get("custom_hour"), 0, 23)
                custom_minute = _coerce_int_range(data.get("custom_minute"), 0, 59)
                if custom_hour is None or custom_minute is None:
                    errors.append("Pick a valid custom time (HH:MM).")
            if not errors:
                set_frequency(
                    user_id, frequency,
                    custom_push_hour=custom_hour,
                    custom_push_minute=custom_minute,
                )

    if errors:
        return jsonify(ok=False, error=" ".join(errors)), 400

    updated = get_user_profile(user_id)
    return jsonify(ok=True, user=_user_payload(updated))


@bp.delete("/api/users/<user_id>")
def api_delete_user(user_id: str):
    from core.rag_engine import drop_user_collection
    existing = get_user_profile(user_id)
    if existing is None:
        return jsonify(ok=False, error="user not found"), 404
    delete_user(user_id)
    drop_user_collection(user_id)
    return jsonify(ok=True)
