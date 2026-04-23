"""BlueBubbles bridge — two-way iMessage over the local BlueBubbles REST API.

- Outbound: `POST /api/v1/message/text` on the BlueBubbles server.
- Inbound:  BlueBubbles server POSTs to our `/bluebubbles/webhook` on each
            new message; we run the agent and reply back through the same API.

Setup on the Mac running BlueBubbles:
  1. Install the BlueBubbles server app (https://bluebubbles.app) and grant
     Full Disk Access when prompted.
  2. Set a server password (Settings → API); put it in BLUEBUBBLES_PASSWORD.
  3. Note the server URL (default http://localhost:1234); put it in
     BLUEBUBBLES_SERVER_URL if non-default.
  4. Add a webhook in Settings → API → Webhooks pointing at
     http://localhost:5000/bluebubbles/webhook and enable the `new-message`
     event.
"""
from __future__ import annotations

import json
import logging
import urllib.error
import urllib.parse
import urllib.request
import uuid
from typing import Optional

from flask import Blueprint, jsonify, request

from agent.graph import run_inbound
from config import BLUEBUBBLES_PASSWORD, BLUEBUBBLES_SERVER_URL
from core.user_profile import get_or_create_user

log = logging.getLogger(__name__)

bp = Blueprint("bluebubbles", __name__, url_prefix="/bluebubbles")


def _chat_guid_for(phone: str) -> str:
    # 1:1 iMessage chat guids look like "iMessage;-;+15551234567"
    return f"iMessage;-;{phone}"


def send_bluebubbles(to: str, body: str, chat_guid: Optional[str] = None) -> str:
    """Send a message via the local BlueBubbles server. Returns the tempGuid."""
    if not body.strip():
        return ""
    if not BLUEBUBBLES_PASSWORD:
        raise RuntimeError("BLUEBUBBLES_PASSWORD not configured")

    temp_guid = str(uuid.uuid4())
    url = (
        f"{BLUEBUBBLES_SERVER_URL.rstrip('/')}/api/v1/message/text"
        f"?password={urllib.parse.quote(BLUEBUBBLES_PASSWORD)}"
    )
    payload = {
        "chatGuid": chat_guid or _chat_guid_for(to),
        "tempGuid": temp_guid,
        "message": body,
        "method": "apple-script",
    }

    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            resp.read()
    except urllib.error.HTTPError as e:
        raise RuntimeError(
            f"BlueBubbles HTTP {e.code}: {e.read().decode(errors='replace')}"
        ) from e
    except urllib.error.URLError as e:
        raise RuntimeError(
            f"BlueBubbles server unreachable at {BLUEBUBBLES_SERVER_URL}: {e}"
        ) from e

    log.info("sent iMessage to %s via BlueBubbles (tempGuid=%s)", to, temp_guid)
    return temp_guid


def _extract_phone(data: dict) -> Optional[str]:
    """The sender's address may land under handle, handles[], or chats[].participants[]."""
    handle = data.get("handle")
    if isinstance(handle, dict) and handle.get("address"):
        return handle["address"]
    handles = data.get("handles")
    if isinstance(handles, list) and handles and isinstance(handles[0], dict):
        addr = handles[0].get("address")
        if addr:
            return addr
    chats = data.get("chats")
    if isinstance(chats, list) and chats and isinstance(chats[0], dict):
        participants = chats[0].get("participants") or []
        if participants and isinstance(participants[0], dict):
            addr = participants[0].get("address")
            if addr:
                return addr
    return None


def _extract_chat_guid(data: dict) -> Optional[str]:
    chats = data.get("chats")
    if isinstance(chats, list) and chats and isinstance(chats[0], dict):
        return chats[0].get("guid")
    return None


@bp.post("/webhook")
def bluebubbles_webhook():
    """Handle a BlueBubbles new-message event and reply through the agent.

    We only process `new-message` with `isFromMe=false`. Everything else
    (typing indicators, message updates, our own outbound echoes) is ack'd
    with 200 and skipped.
    """
    event = request.get_json(silent=True) or {}
    event_type = event.get("type", "")
    data = event.get("data") or {}

    if event_type != "new-message":
        return jsonify(ok=True, skipped=event_type or "empty")
    if data.get("isFromMe"):
        return jsonify(ok=True, skipped="self")

    text = (data.get("text") or "").strip()
    phone = _extract_phone(data)
    chat_guid = _extract_chat_guid(data)
    if not (phone and text):
        log.warning("BlueBubbles webhook missing phone or text: %s", event)
        return jsonify(ok=True, skipped="missing-fields")

    try:
        user = get_or_create_user(phone)
        reply = run_inbound(user.user_id, text)
    except Exception:
        log.exception("inbound handler failed for %s", phone)
        return jsonify(ok=True, error="handler-failed")

    if reply:
        try:
            send_bluebubbles(phone, reply, chat_guid=chat_guid)
        except Exception:
            log.exception("send_bluebubbles reply failed for %s", phone)

    return jsonify(ok=True)
