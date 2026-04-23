"""One-command entry points for local development.

    python main.py init             # initialize DB + cache schema
    python main.py serve            # run Flask app (signup UI + Twilio webhook)
    python main.py scheduler        # run the hourly proactive-push scheduler
    python main.py push-once        # one scheduler tick now (respects frequency)
    python main.py push-all         # force-push every onboarded user (demos)
    python main.py cli <phone>      # interactive REPL acting as a user
    python main.py list             # list all users
    python main.py delete <phone>   # delete a user and all their data
"""
from __future__ import annotations

import argparse
import logging
import sys

from agent.graph import run_inbound
from core.article_fetcher import init_cache
from core.rag_engine import drop_user_collection
from core.user_profile import (
    delete_user,
    get_or_create_user,
    get_user_by_phone,
    init_db,
    list_users,
)
from gateway.web import normalize_phone


def _init() -> None:
    init_db()
    init_cache()
    print("initialized SQLite + article cache schemas")


def _serve() -> None:
    from gateway.twilio_handler import main as serve_main
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


def _cli(phone: str) -> None:
    init_db()
    init_cache()
    user = get_or_create_user(phone)
    print(f"talking as {user.user_id} ({user.phone}). Ctrl-D to exit.")
    # nudge onboarding if needed
    if user.onboarding_state == "NEEDS_ONBOARDING":
        print("assistant:", run_inbound(user.user_id, ""))
    while True:
        try:
            text = input("you: ")
        except (EOFError, KeyboardInterrupt):
            print()
            return
        if not text.strip():
            continue
        print("assistant:", run_inbound(user.user_id, text))


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    parser = argparse.ArgumentParser(description="NewsAgent local entry point")
    sub = parser.add_subparsers(dest="cmd", required=True)

    sub.add_parser("init")
    sub.add_parser("serve")
    sub.add_parser("scheduler")
    sub.add_parser("push-once")
    sub.add_parser("push-all")
    sub.add_parser("list")
    p_cli = sub.add_parser("cli")
    p_cli.add_argument("phone", help="phone number in E.164 format, e.g. +15551234567")
    p_del = sub.add_parser("delete")
    p_del.add_argument("phone", help="phone number of the user to delete")
    p_del.add_argument("-y", "--yes", action="store_true", help="skip confirmation prompt")

    args = parser.parse_args()
    dispatch = {
        "init": _init,
        "serve": _serve,
        "scheduler": _scheduler,
        "push-once": _push_once,
        "push-all": _push_all,
        "list": _list,
        "cli": lambda: _cli(args.phone),
        "delete": lambda: _delete(args.phone, args.yes),
    }
    dispatch[args.cmd]()


if __name__ == "__main__":
    main()
