# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Common commands

All entry points go through `main.py`:

```bash
python main.py init                # create / migrate SQLite schemas + article cache
python main.py serve               # Flask app on :5000 (signup UI, admin, webhooks)
python main.py scheduler           # blocking APScheduler — ticks every minute
python main.py push-once           # one scheduler tick now (respects per-user frequency)
python main.py push-all            # force-push to every onboarded user (ignores debounce)
python main.py cli <phone>         # interactive REPL acting as that user (no iMessage needed)
python main.py list                # tab-separated dump of all users
python main.py delete <phone>      # remove user's SQLite rows + Chroma collection
python main.py logs <phone>        # print readable per-user transcript with full prompts
```

Evaluation harness:

```bash
python -m eval.evaluation --all --out eval_results.json
python -m eval.evaluation --retrieval    # hit-rate@k on gold QA pairs (works without API key)
python -m eval.evaluation --relevance    # LLM-as-Judge with chain-of-thought
python -m eval.evaluation --baseline     # TF-IDF vs. LLM-ranked comparison
```

There is no test runner, lint configuration, or build step. To smoke-check changes, run `python3 -m compileall -q core/ agent/ gateway/ eval/ scheduler.py main.py`.

After schema changes to `core/user_profile.py`, callers must re-run `python main.py init` — `init_db()` triggers the idempotent `_migrate()` block that runs `ALTER TABLE ... ADD COLUMN` for any new columns.

## Architecture

### Two flows, one agent

Two separate LangGraph pipelines live in `agent/graph.py`, both sharing the same nodes/state:

- **`run_inbound(user_id, text)`** — `load_user → route_inbound → (onboarding | followup) → END`. Driven by the BlueBubbles webhook when a real user texts in. Routing is a single check on `user.onboarding_state`.
- **`run_proactive_push(user_id, force_refresh=False)`** — `load_user → fetch → format → END`. Driven by `scheduler.py` on its minute-level cron tick.

The followup path always reaches `core/rag_engine.handle_followup`, which is the single place the chat-side LLM is called. The proactive path only formats and indexes; it never calls the chat LLM directly.

### Scheduler model

`scheduler.py` runs APScheduler with `cron(minute="*")` — fires every minute UTC. On each tick it walks every user and calls `core.user_profile.is_push_due(user, now)` which checks two things:

1. The current minute-of-day matches one of the user's targets (within ±1 minute slack to tolerate cron drift).
2. `last_pushed_at` is at least `_MIN_SECONDS_BETWEEN_PUSHES` (3h) old.

Targets live in `_FREQUENCY_HOURS` for fixed schedules; for `frequency = "custom_daily"` the target is `(custom_push_hour, custom_push_minute)`. After a successful send, `mark_pushed()` updates `last_pushed_at` so the same minute window doesn't re-fire.

When adding a new schedule type, update **all four**: `FREQUENCY_CHOICES` (label), `_FREQUENCY_HOURS` (minute targets) or the custom branch in `_target_minutes_for`, the React SPA `FREQ_OPTIONS` in both `signup.html` and `users.html`, and the `set_frequency` validator.

### Storage

Three storage layers with different lifecycles:

- **SQLite** (`data/newsagent.db`) — `users`, `messages` (rolling conversation history), `sent_articles` (per-user dedupe), `articles` (global article cache).
- **ChromaDB** (`data/chroma/`) — one collection per user named `user_<user_id>`, holding paragraph-chunked embeddings of the articles that have been pushed to that user. Used only by the RAG retrieval step. Created lazily; deleted by `drop_user_collection()`.
- **JSONL + per-user transcript** (`data/logs/`) — `core/conv_log.py` writes every inbound, outbound, and LLM call with the full prompt and response. Master log is `conversations.jsonl`; readable per-user log is `<phone>.log`.

### Three LLM call sites

When tracing prompt changes or token costs, these are the only places we call Claude:

1. `core/rag_engine.py` — followup answer (`_ANSWER_PROMPT`). The reply users see when they text in.
2. `core/article_fetcher.py` — article ranker (`_RANK_PROMPT`). Scores TF-IDF-prefiltered candidates against the persona summary; called once per push cycle.
3. `core/user_profile.py` — persona summary (`_PERSONA_PROMPT`). Called on signup and on every topic edit; the output is what the article ranker compares articles against.

All three are gated by `llm_configured()` which treats both empty and placeholder `sk-ant-...` keys as "not configured" so the system degrades gracefully when no key is set.

### Messaging gateway

The Flask app in `gateway/twilio_handler.py` registers two blueprints:

- `gateway.web.bp` — serves the React SPAs (`/`, `/users`) and the JSON admin API (`/signup`, `/api/users[/<id>]`).
- `gateway.bluebubbles.bp` — the inbound webhook at `/bluebubbles/webhook`. Filters non-`new-message` events and `isFromMe=true`, looks up the sender by phone (exact match against `users.phone`), runs `run_inbound`, and POSTs the reply back through the BlueBubbles REST API.

Inbound iMessage is **only** processed for already-signed-up phone numbers — strangers are silently ignored. To change this, edit the `get_user_by_phone(phone)` lookup in `bluebubbles_webhook`. The signup page normalizes phones to E.164 (`+15551234567`); BlueBubbles also emits E.164, so an exact string compare is sufficient in practice.

`gateway/twilio_handler.py` itself remains a working alternative SMS backend but no scheduler code calls it currently.

### React SPAs

`gateway/static/signup.html` and `gateway/static/users.html` are self-contained: React 18 UMD + Babel Standalone + Tailwind, all from CDN, no build step. JSX is parsed in-browser. When editing them, remember Tailwind classes (`text-slate-900`, etc.) are interpreted by the runtime CDN script — there's no `tailwind.config.js`.

The custom-time picker computes UTC at submit by `new Date().setHours(h, m); .getUTCHours()/.getUTCMinutes()`, then sends `custom_hour` + `custom_minute` to the backend. The `users.html` edit modal does the inverse with `utcToLocalTime()` to pre-fill the time input.

## Operational notes

- NewsAPI is fetched once per cache cycle (6h) globally, not per user, to stay inside the free-tier quota.
- `MAX_ARTICLES_PER_PUSH` (config) bounds digest size; `TFIDF_PREFILTER_K` (article_fetcher) bounds how many candidates the LLM ranker scores.
- All schedule arithmetic is UTC. The React SPA shows the user's IANA timezone next to the time input but stores UTC.
