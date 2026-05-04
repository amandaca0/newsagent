"""One-command entry points for local development.

    python main.py init             # initialize DB + cache schema
    python main.py serve            # run Flask app (signup UI + BlueBubbles webhook)
    python main.py scheduler        # run the hourly proactive-push scheduler
    python main.py push-once        # one scheduler tick now (respects frequency)
    python main.py push-all         # force-push every onboarded user (demos)
    python main.py push <phone>     # force-push to one user by phone
    python main.py cli <phone>      # local mirror of BlueBubbles flow: digest + REPL
    python main.py list             # list all users
    python main.py delete <phone>   # delete a user and all their data
    python main.py logs <phone>     # print the readable conversation log
"""
from __future__ import annotations

import argparse
import logging
import sys

from agent.graph import run_inbound, run_proactive_push
from core.article_fetcher import init_cache
from core.conv_log import log_event
from core.rag_engine import drop_user_collection
from core.user_profile import (
    clear_sent_articles,
    delete_user,
    get_user_by_phone,
    init_db,
    list_users,
    mark_pushed,
)
from gateway.web import normalize_phone


def _init() -> None:
    init_db()
    init_cache()
    print("initialized SQLite + article cache schemas")


def _serve() -> None:
    from gateway.app import main as serve_main
    serve_main()


def _scheduler() -> None:
    from scheduler import main as sched_main
    sys.argv = [sys.argv[0]]
    sched_main()


def _push_once() -> None:
    from scheduler import push_due_users
    init_db()
    init_cache()
    push_due_users()


def _push_all() -> None:
    from scheduler import push_all_users
    init_db()
    init_cache()
    push_all_users()


def _push_user(phone_raw: str) -> None:
    from scheduler import push_one_phone
    init_db()
    init_cache()
    phone = normalize_phone(phone_raw) or phone_raw
    push_one_phone(phone)


def _list() -> None:
    init_db()
    users = list_users()
    if not users:
        print("no users.")
        return
    for u in users:
        topics = ", ".join(u.interests) or "(none)"
        print(f"{u.phone}\t{u.user_id}\t{u.onboarding_state}\t{u.frequency}\t{topics}")


def _delete(phone_raw: str, yes: bool) -> None:
    init_db()
    phone = normalize_phone(phone_raw) or phone_raw
    user = get_user_by_phone(phone)
    if user is None:
        print(f"no user found for {phone}")
        return
    if not yes:
        resp = input(f"delete {user.phone} ({user.user_id}) and all their data? [y/N] ")
        if resp.strip().lower() not in {"y", "yes"}:
            print("aborted.")
            return
    deleted = delete_user(user.user_id)
    drop_user_collection(user.user_id)
    print(f"deleted={deleted} user_id={user.user_id}")


def _logs(phone_raw: str) -> None:
    from core.conv_log import per_user_path, read_user_log
    phone = normalize_phone(phone_raw) or phone_raw
    text = read_user_log(phone)
    if not text:
        print(f"no log file for {phone} (looked for {per_user_path(phone)})")
        return
    print(text)


def _cli(phone_raw: str) -> None:
    """Local mirror of the BlueBubbles flow — no iMessage required.

    Looks up an already-registered, onboarded user, force-pushes a digest to
    the terminal (mirroring scheduler.push_one_phone), then drops into a REPL
    where each line is sent through run_inbound and the reply is printed.
    Eval metrics fire automatically through the graph hooks when EVAL_MODE=1.
    """
    init_db()
    init_cache()
    phone = normalize_phone(phone_raw) or phone_raw
    user = get_user_by_phone(phone)
    if user is None:
        print(f"no user found for {phone}. Sign up at the web UI first.")
        return
    if user.onboarding_state != "DONE":
        print(f"user {phone} is not onboarded yet (state={user.onboarding_state}).")
        return

    print(f"talking as {user.user_id} ({user.phone}). Ctrl-D to exit.\n")

    # --- digest push (mirror push_one_phone, but print instead of BlueBubbles) ---
    clear_sent_articles(user.user_id)
    try:
        reply, articles = run_proactive_push(user.user_id, force_refresh=True)
    except Exception:
        logging.getLogger(__name__).exception("proactive push failed")
        reply, articles = "", []

    if reply and articles:
        log_event(
            "proactive_digest",
            user_id=user.user_id, phone=user.phone,
            articles=[{"title": a.title, "source": a.source, "url": a.url} for a in articles],
            text=reply,
        )
        log_event(
            "outbound_message",
            user_id=user.user_id, phone=user.phone,
            text=reply, purpose="digest",
        )
        mark_pushed(user.user_id)
        print("assistant:")
        print(reply)
        print()
    else:
        print("(no new articles to send right now — entering REPL anyway)\n")

    # --- follow-up REPL (mirror bluebubbles_webhook inbound path) ---
    # Re-fetch the user so conversation_history reflects the digest we just sent.
    user = get_user_by_phone(phone)
    while True:
        try:
            text = input("you: ")
        except (EOFError, KeyboardInterrupt):
            print()
            return
        text = text.strip()
        if not text:
            continue
        log_event("inbound_message", user_id=user.user_id, phone=user.phone, text=text)
        try:
            reply = run_inbound(user.user_id, text)
        except Exception:
            logging.getLogger(__name__).exception("inbound handler failed")
            print("assistant: (handler error — see logs)\n")
            continue
        if reply:
            log_event(
                "outbound_message",
                user_id=user.user_id, phone=user.phone,
                text=reply, purpose="rag_reply",
            )
            print(f"assistant: {reply}\n")


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    parser = argparse.ArgumentParser(description="NewsAgent local entry point")
    sub = parser.add_subparsers(dest="cmd", required=True)

    sub.add_parser("init")
    sub.add_parser("serve")
    sub.add_parser("scheduler")
    sub.add_parser("push-once")
    sub.add_parser("push-all")
    p_push = sub.add_parser("push")
    p_push.add_argument("phone", help="phone number of the user to push to")
    sub.add_parser("list")
    p_cli = sub.add_parser("cli")
    p_cli.add_argument("phone", help="phone number in E.164 format, e.g. +15551234567")
    p_del = sub.add_parser("delete")
    p_del.add_argument("phone", help="phone number of the user to delete")
    p_del.add_argument("-y", "--yes", action="store_true", help="skip confirmation prompt")
    p_logs = sub.add_parser("logs")
    p_logs.add_argument("phone", help="phone number of the user")

    args = parser.parse_args()
    dispatch = {
        "init": _init,
        "serve": _serve,
        "scheduler": _scheduler,
        "push-once": _push_once,
        "push-all": _push_all,
        "push": lambda: _push_user(args.phone),
        "list": _list,
        "cli": lambda: _cli(args.phone),
        "delete": lambda: _delete(args.phone, args.yes),
        "logs": lambda: _logs(args.phone),
    }
    dispatch[args.cmd]()


if __name__ == "__main__":
    main()
