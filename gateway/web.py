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
    set_name,
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


_FREQUENCIES_NEEDING_TIME = {
    "every_4h", "every_8h", "morning_9am", "evening_6pm", "custom_daily", "twice_daily",
}


def _parse_schedule_times(data, frequency: str):
    """Pull custom_hour/minute (and _2 for twice_daily) out of the request body.
    Returns a 4-tuple, with values None when not applicable / not provided."""
    h = m = h2 = m2 = None
    if frequency in _FREQUENCIES_NEEDING_TIME:
        h = _coerce_int_range(data.get("custom_hour"), 0, 23)
        m = _coerce_int_range(data.get("custom_minute"), 0, 59)
    if frequency == "twice_daily":
        h2 = _coerce_int_range(data.get("custom_hour_2"), 0, 23)
        m2 = _coerce_int_range(data.get("custom_minute_2"), 0, 59)
    return h, m, h2, m2


def _schedule_time_errors(frequency, h, m, h2, m2) -> list[str]:
    errs: list[str] = []
    if frequency in _FREQUENCIES_NEEDING_TIME and (h is None or m is None):
        errs.append("Pick a valid time (HH:MM) for this schedule.")
    if frequency == "twice_daily" and (h2 is None or m2 is None):
        errs.append("Pick a valid second time (HH:MM) for the twice-daily schedule.")
    return errs


def _frequency_label(frequency: str) -> str:
    """Human-readable name only — frontend renders any time numbers in local time."""
    return FREQUENCY_CHOICES.get(frequency, frequency)


def _user_payload(u: User) -> dict:
    d = u.to_dict()
    d.pop("conversation_history", None)
    d["frequency_label"] = _frequency_label(u.frequency)
    return d


def _compose_welcome(u: User) -> str:
    topics = ", ".join(u.interests) if u.interests else "(no topics set)"
    schedule = _frequency_label(u.frequency)
    greeting = f"Welcome to NewsAgent, {u.name}!" if u.name else "Welcome to NewsAgent!"
    return (
        f"{greeting}\n\n"
        f"I'll send you personalized news digests covering: {topics}\n\n"
        f"Schedule: {schedule}\n\n"
        "Reply to any digest with a follow-up question and I'll answer using "
        "the articles I've sent you."
    )


def _send_welcome(u: User) -> None:
    """Fire off a welcome iMessage. Failures are logged, not raised, so that
    a flaky BlueBubbles server doesn't break the signup flow."""
    body = _compose_welcome(u)
    try:
        from gateway.bluebubbles import send_bluebubbles
        send_bluebubbles(u.phone, body)
        from core.conv_log import log_event
        log_event(
            "outbound_message",
            user_id=u.user_id, phone=u.phone,
            text=body, purpose="welcome",
        )
    except Exception:
        log.exception("welcome message failed for %s", u.phone)


@bp.post("/signup")
def signup_submit():
    data = request.get_json(silent=True)
    if data is None:
        data = request.form  # fallback for plain form posts

    phone = normalize_phone(data.get("phone", ""))
    name = (data.get("name") or "").strip()
    topics = _parse_topics(data.get("topics", ""))
    frequency = data.get("frequency", "morning_9am")
    custom_hour, custom_minute, custom_hour_2, custom_minute_2 = _parse_schedule_times(data, frequency)

    errors = []
    if phone is None:
        errors.append("That phone number didn't look valid.")
    if not name:
        errors.append("Tell us your name.")
    if len(topics) < 5:
        errors.append("Add at least 5 topics.")
    if frequency not in FREQUENCY_CHOICES:
        errors.append("Pick a valid frequency.")
    errors.extend(_schedule_time_errors(frequency, custom_hour, custom_minute,
                                        custom_hour_2, custom_minute_2))

    if errors:
        return jsonify(ok=False, error=" ".join(errors)), 400

    user = get_or_create_user(phone)
    set_name(user.user_id, name)
    set_interests(user.user_id, topics)
    set_persona_summary(user.user_id, generate_persona_summary(topics))
    set_frequency(
        user.user_id, frequency,
        custom_push_hour=custom_hour,
        custom_push_minute=custom_minute,
        custom_push_hour_2=custom_hour_2,
        custom_push_minute_2=custom_minute_2,
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

    if "name" in data:
        name = (data.get("name") or "").strip()
        if not name:
            errors.append("Name cannot be empty.")
        else:
            set_name(user_id, name)

    # topics — regenerate persona summary when they change
    if "topics" in data:
        topics = _parse_topics(data["topics"])
        if len(topics) < 5:
            errors.append("Add at least 5 topics.")
        else:
            set_interests(user_id, topics)
            set_persona_summary(user_id, generate_persona_summary(topics))

    # frequency (and custom time(s) if applicable)
    if "frequency" in data:
        frequency = data["frequency"]
        if frequency not in FREQUENCY_CHOICES:
            errors.append("Pick a valid frequency.")
        else:
            h, m, h2, m2 = _parse_schedule_times(data, frequency)
            errors.extend(_schedule_time_errors(frequency, h, m, h2, m2))
            if not errors:
                set_frequency(
                    user_id, frequency,
                    custom_push_hour=h,
                    custom_push_minute=m,
                    custom_push_hour_2=h2,
                    custom_push_minute_2=m2,
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
