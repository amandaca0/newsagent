"""Signup web UI.

Single-page form: phone + topics + frequency. Posts back to /signup which
creates (or updates) the user, sets interests + persona summary, and marks
onboarding DONE so the scheduler starts delivering.

Auth is intentionally not implemented — single-tester setup only.
"""
from __future__ import annotations

import logging
import re

from flask import Blueprint, render_template, request

from core.user_profile import (
    FREQUENCY_CHOICES,
    generate_persona_summary,
    get_or_create_user,
    set_frequency,
    set_interests,
    set_onboarding_state,
    set_persona_summary,
)

log = logging.getLogger(__name__)

bp = Blueprint("web", __name__)


_PHONE_RE = re.compile(r"\D+")


def normalize_phone(raw: str) -> str | None:
    """Coerce user-entered phone to E.164. Returns None if unparseable.

    Simple rules suitable for a demo:
      - strip everything non-digit
      - if it starts with +, keep as-is after non-digit strip (then re-add +)
      - 10 digits -> assume US, prepend +1
      - 11 digits starting with 1 -> prepend +
      - otherwise, reject
    """
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


def _parse_topics(raw: str) -> list[str]:
    if not raw:
        return []
    parts = re.split(r"[,\n;]", raw)
    return [p.strip() for p in parts if p.strip()]


@bp.get("/")
def signup_form():
    return render_template("signup.html", frequency_choices=FREQUENCY_CHOICES, form={}, error=None)


@bp.post("/signup")
def signup_submit():
    form = request.form
    phone = normalize_phone(form.get("phone", ""))
    topics = _parse_topics(form.get("topics", ""))
    frequency = form.get("frequency", "morning_9am")

    errors = []
    if phone is None:
        errors.append("That phone number didn't look valid.")
    if not topics:
        errors.append("Add at least one topic.")
    if frequency not in FREQUENCY_CHOICES:
        errors.append("Pick a valid frequency.")

    if errors:
        return render_template(
            "signup.html",
            frequency_choices=FREQUENCY_CHOICES,
            form=form,
            error=" ".join(errors),
        ), 400

    user = get_or_create_user(phone)
    set_interests(user.user_id, topics)
    set_persona_summary(user.user_id, generate_persona_summary(topics))
    set_frequency(user.user_id, frequency)
    set_onboarding_state(user.user_id, "DONE")

    # re-read to show the canonical stored state
    from core.user_profile import get_user_profile
    saved = get_user_profile(user.user_id)
    return render_template(
        "success.html",
        user=saved,
        frequency_label=FREQUENCY_CHOICES[frequency],
    )
