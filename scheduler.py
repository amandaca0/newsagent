"""APScheduler process — hourly tick that pushes the right users.

Every hour on the hour, we walk the user table and push to anyone whose
`frequency` window includes the current UTC hour AND who hasn't been pushed
recently (debounce at 3h, see core.user_profile.is_push_due).

    python scheduler.py            # run the hourly scheduler
    python scheduler.py --once      # one tick now — respects frequency
    python scheduler.py --force-all # push every onboarded user now (for demos)
"""
from __future__ import annotations

import argparse
import logging
import time

from apscheduler.schedulers.blocking import BlockingScheduler

from agent.graph import run_proactive_push
from core.article_fetcher import init_cache
from core.user_profile import (
    clear_sent_articles,
    init_db,
    is_push_due,
    list_users,
    mark_pushed,
)
from gateway.bluebubbles import send_bluebubbles

log = logging.getLogger(__name__)


def _push_one(user, force_refresh: bool = False) -> bool:
    try:
        reply, articles = run_proactive_push(user.user_id, force_refresh=force_refresh)
    except Exception:
        log.exception("proactive push failed for %s", user.user_id)
        return False
    if not reply or not articles:
        log.info("no new articles for %s; skipping send", user.user_id)
        return False
    try:
        msg_id = send_bluebubbles(user.phone, reply)
        log.info("pushed %d articles to %s (msg_id=%s)", len(articles), user.phone, msg_id)
    except Exception:
        log.exception("iMessage send failed for %s", user.phone)
        return False
    mark_pushed(user.user_id)
    return True


def push_due_users() -> None:
    now = time.time()
    users = list_users()
    due = [u for u in users if is_push_due(u, now)]
    log.info("hourly tick: %d users, %d due", len(users), len(due))
    for user in due:
        _push_one(user)


def push_all_users() -> None:
    """Force-push to every onboarded user — ignores frequency, debounce, and sent history."""
    users = [u for u in list_users() if u.onboarding_state == "DONE"]
    log.info("force-push to %d onboarded users", len(users))
    for user in users:
        clear_sent_articles(user.user_id)
        _push_one(user, force_refresh=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--once", action="store_true",
                        help="run one tick now (respects frequency)")
    parser.add_argument("--force-all", action="store_true",
                        help="push every onboarded user now (ignores frequency)")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    init_db()
    init_cache()

    if args.force_all:
        push_all_users()
        return
    if args.once:
        push_due_users()
        return

    sched = BlockingScheduler(timezone="UTC")
    sched.add_job(push_due_users, "cron", minute="*",
                  id="minute_tick", max_instances=1, coalesce=True)
    log.info("scheduler started; tick every minute UTC")
    sched.start()


if __name__ == "__main__":
    main()
