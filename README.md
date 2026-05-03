# NewsAgent

Proactive iMessage news agent. Onboards users through a web signup, pushes
personalized digests on a per-user schedule, and answers follow-up questions
over the articles it sent using RAG. iMessage I/O runs locally through a
BlueBubbles server; a Twilio SMS backend is included as an alternative.

## Architecture

```
   ┌────────────────────────┐        ┌────────────────────────┐
   │ React signup / admin   │        │ scheduler.py           │
   │ SPA (CDN, no build)    │        │ APScheduler, 1-min UTC │
   └─────────┬──────────────┘        └────────┬───────────────┘
             │ /signup, /api/users            │ run_proactive_push
             ▼                                 ▼
   ┌──────────────────────────────────────────────────┐
   │ Flask app (gateway/twilio_handler.py)            │
   │   • gateway/web.bp        — SPA + admin JSON     │
   │   • gateway/bluebubbles.bp — /bluebubbles/webhook│
   └─────────┬────────────────────────┬───────────────┘
             │ inbound iMessage       │ outbound digest
             ▼                        ▼
        BlueBubbles server (local Mac, REST + webhooks)
                            │
                            ▼
   ┌──────────────────────────────────────────────────┐
   │ agent/graph.py — LangGraph state machine         │
   │   inbound:    load_user → route → onboarding     │
   │                                  ↘ followup      │
   │   proactive:  load_user → fetch → format         │
   └─────┬─────────────┬──────────────────┬───────────┘
         │             │                  │
   ┌─────▼──────┐  ┌───▼──────────┐  ┌────▼──────────────┐
   │user_profile│  │article_fetch │  │ rag_engine        │
   │ SQLite:    │  │ NewsAPI →    │  │ Chroma (per-user) │
   │ users,     │  │ SQLite cache │  │ + sentence-       │
   │ messages,  │  │ TF-IDF → LLM │  │ transformers +    │
   │ sent_art., │  │ rank + diver │  │ MMR + Llama 3.3   │
   │ articles   │  │ sify         │  │ via Groq          │
   └────────────┘  └──────────────┘  └───────────────────┘
```

**LLM**: Llama 3.3 70B served by Groq (`LLM_MODEL`, default
`llama-3.3-70b-versatile`). Three call sites: persona summarization, article
ranking, and RAG answering. All gated by `llm_configured()` for graceful
degradation when no API key is set.

## Quickstart

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # fill in API keys

python main.py init                 # create SQLite + article-cache schemas
python main.py serve                # start Flask app (signup UI + BlueBubbles webhook)
python main.py scheduler            # run APScheduler daemon (1-min UTC tick)
python main.py push-once            # one scheduler tick now (respects per-user frequency)
python main.py push-all             # force-push to every onboarded user
python main.py push <phone>         # force-push to one user
python main.py cli <phone>          # chat as a user in the terminal (no iMessage needed)
python main.py list                 # list all users
python main.py logs <phone>         # print readable per-user transcript
python main.py delete <phone>       # remove user's SQLite rows + Chroma collection
```

## Evaluation

Set `EVAL_MODE=1` to enable inline evaluation. When active, every digest send
and follow-up reply is automatically scored and appended to
`data/eval_metrics.jsonl` (one JSON record per event, two event types):

| Event | What's measured |
|---|---|
| `digest` | Per-article cosine similarity to user persona; best-matching interest topic + similarity |
| `response` | Question–answer cosine similarity; LLM-as-Judge scores for `human_readability`, `conciseness`, `accuracy` (0–1) |

The LLM-as-Judge step requires an Anthropic API key (`ANTHROPIC_API_KEY`);
Groq is intentionally not used as the judge. If no Anthropic key is
configured, judge scores are recorded as `null` and all other metrics still
run.

The evaluator is fully wrapped in `try/except` — a crash never affects a real
user send.

## Layout

- `core/user_profile.py` — SQLite-backed user, message, and sent-articles tables; persona summarization; per-user frequency / debounce logic
- `core/article_fetcher.py` — NewsAPI fetch (7 categories), 7-day SQLite cache, TF-IDF prefilter, LLM rerank, near-duplicate diversity filter
- `core/rag_engine.py` — paragraph chunking, ChromaDB per-user collections, two-tier candidate selection, MMR retrieval, sentinel-token abstention
- `agent/graph.py` — LangGraph inbound + proactive pipelines (shared nodes); `topic_diverse_articles` enforces per-interest coverage
- `scheduler.py` — APScheduler 1-minute cron; `is_push_due` evaluates target-minute match within ±1 min slack and 3 h debounce
- `gateway/bluebubbles.py` — BlueBubbles outbound sender + `/bluebubbles/webhook`
- `gateway/web.py` — React SPA, signup + admin JSON API
- `gateway/twilio_handler.py` — Flask app entry point; Twilio SMS backend (alternate transport)
- `core/eval_runtime.py` — inline evaluation hooks (digest relevance + LLM-as-Judge response scoring); toggled by `EVAL_MODE=1`, writes to `data/eval_metrics.jsonl`

## BlueBubbles (two-way iMessage, free, local-only)

An alternative to Twilio for demos. Runs entirely on your Mac.

1. Install the server app: https://bluebubbles.app → download "Server"
2. Open it, sign into iMessage, grant **Full Disk Access** + **Accessibility**
   + **Automation** when macOS prompts.
3. Settings → API → set a password; copy into `BLUEBUBBLES_PASSWORD` in `.env`.
4. Settings → API → Webhooks → add `http://localhost:5000/bluebubbles/webhook`
   and enable the `new-message` event.
5. Start the Flask app (`python main.py serve`) and text your Mac's iMessage
   address from your phone — the agent will reply in the same thread.

Outbound (digest push) goes through the same server:
`scheduler.py` calls `send_bluebubbles(user.phone, body)`.

## Notes

- NewsAPI is fetched at most once every 2 h globally (not per user) and articles persist for 7 days, keeping the system inside the NewsAPI free-tier quota.
- `MAX_ARTICLES_PER_PUSH` in config bounds digest size; `TFIDF_PREFILTER_K` (in `article_fetcher.py`) bounds how many candidates the LLM ranker scores.
- Sent-article IDs are tracked per user so users never receive the same story twice.
- Schedule frequencies supported: `every_4h`, `every_8h`, `morning_9am`, `evening_6pm`, `twice_daily`, and `custom_daily` (user-specified UTC hour+minute). All schedule arithmetic is UTC; the React SPA converts the local-time picker to UTC at submit.
- Schema migrations are idempotent — re-run `python main.py init` after any schema change in `core/user_profile.py`.
