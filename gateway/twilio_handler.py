"""Twilio SMS gateway.

Exposes a Flask app with one webhook (`POST /sms`) that Twilio hits on each
inbound SMS. Outbound sends go through `send_sms()`, which is also used by
the scheduler for the proactive digest.
"""
from __future__ import annotations

import logging
from urllib.parse import urljoin

from flask import Flask, Response, abort, request
from twilio.request_validator import RequestValidator
from twilio.rest import Client
from twilio.twiml.messaging_response import MessagingResponse

from agent.graph import run_inbound
from config import (
    PUBLIC_URL,
    TWILIO_ACCOUNT_SID,
    TWILIO_AUTH_TOKEN,
    TWILIO_FROM_NUMBER,
)
from core.article_fetcher import init_cache
from core.user_profile import get_or_create_user, init_db
from gateway.web import bp as web_bp

log = logging.getLogger(__name__)

# SMS has a 1600-char hard limit; Twilio segments at 160 GSM-7 / 70 UCS-2.
# We truncate at 1500 to leave safety margin for the final segment.
_MAX_SMS_CHARS = 1500


app = Flask(__name__)
app.register_blueprint(web_bp)


def _validate_twilio(req) -> bool:
    """Verify the request came from Twilio using the HMAC signature."""
    if not TWILIO_AUTH_TOKEN:
        log.warning("TWILIO_AUTH_TOKEN not set — skipping signature validation")
        return True
    validator = RequestValidator(TWILIO_AUTH_TOKEN)
    signature = req.headers.get("X-Twilio-Signature", "")
    url = urljoin(PUBLIC_URL, req.path)
    return validator.validate(url, req.form.to_dict(), signature)


@app.post("/sms")
def sms_webhook() -> Response:
    if not _validate_twilio(request):
        abort(403)

    from_number = request.form.get("From", "").strip()
    body = request.form.get("Body", "").strip()
    if not from_number:
        abort(400, "missing From")

    user = get_or_create_user(from_number)
    reply = run_inbound(user.user_id, body)

    twiml = MessagingResponse()
    if reply:
        twiml.message(_truncate(reply))
    return Response(str(twiml), mimetype="application/xml")


@app.get("/healthz")
def healthz() -> Response:
    return Response("ok", mimetype="text/plain")


def _truncate(text: str) -> str:
    if len(text) <= _MAX_SMS_CHARS:
        return text
    return text[: _MAX_SMS_CHARS - 3] + "..."


_client: Client | None = None


def _get_client() -> Client:
    global _client
    if _client is None:
        if not (TWILIO_ACCOUNT_SID and TWILIO_AUTH_TOKEN):
            raise RuntimeError("Twilio credentials not configured")
        _client = Client(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)
    return _client


def send_sms(to_number: str, body: str) -> str:
    """Send an outbound SMS. Returns the Twilio message SID."""
    if not body.strip():
        return ""
    client = _get_client()
    msg = client.messages.create(
        to=to_number,
        from_=TWILIO_FROM_NUMBER,
        body=_truncate(body),
    )
    return msg.sid


def main() -> None:
    import os
    logging.basicConfig(level=logging.INFO)
    init_db()
    init_cache()
    port = int(os.getenv("PORT", "5000"))
    log.info("serving on http://localhost:%d (signup at /, Twilio webhook at /sms)", port)
    app.run(host="0.0.0.0", port=port)


if __name__ == "__main__":
    main()
