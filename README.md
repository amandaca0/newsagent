# NewsAgent

Proactive SMS news agent. Onboards users over text, pushes a personalized
daily digest, and answers follow-up questions over the articles it sent using
RAG.

## Architecture

```
          ┌──────────────────┐
          │ Twilio (SMS)     │
          └───────┬──────────┘
                  │ webhook
          ┌───────▼──────────┐          ┌─────────────────┐
          │ gateway/         │          │ scheduler.py    │
          │ twilio_handler   │          │ (APScheduler)   │
          └───────┬──────────┘          └───────┬─────────┘
                  │                             │
                  │      ┌──────────────────────┘
                  ▼      ▼
             ┌──────────────────────┐
             │ agent/graph.py       │  LangGraph state machine
             │ run_inbound / run_   │
             │ proactive_push       │
             └──┬─────────┬─────┬───┘
                │         │     │
     ┌──────────▼──┐ ┌────▼──┐ ┌▼─────────────┐
     │ user_profile│ │article│ │ rag_engine   │
     │ (SQLite)    │ │fetcher│ │ (ChromaDB +  │
     │             │ │NewsAPI│ │  Claude)     │
     │             │ │ + RSS │ │              │
     └─────────────┘ └───────┘ └──────────────┘
```

## Quickstart

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # fill in API keys

python main.py init                 # create SQLite + Chroma schemas
python main.py cli +15551234567     # chat as a user in the terminal
python main.py push-once            # trigger one proactive-push cycle
python main.py serve                # start Twilio webhook (port 5000)
python main.py scheduler            # run APScheduler daemon
```

## Evaluation

```bash
python -m eval.evaluation --all --out eval_results.json
python -m eval.evaluation --retrieval     # hit-rate@k on gold QA pairs
python -m eval.evaluation --relevance     # LLM-as-Judge on pushed articles
python -m eval.evaluation --baseline      # TF-IDF vs. LLM-ranked comparison
```

## Layout

- `core/user_profile.py` — SQLite-backed user, message, and sent-articles tables
- `core/article_fetcher.py` — NewsAPI + RSS fetch, cache, TF-IDF and LLM ranking
- `core/rag_engine.py` — paragraph chunking, ChromaDB, MMR retrieval, answer gen
- `agent/graph.py` — LangGraph inbound + proactive pipelines
- `gateway/bluebubbles.py` — BlueBubbles outbound sender + `/bluebubbles/webhook`
- `gateway/twilio_handler.py` — Flask webhook with Twilio signature validation
- `eval/evaluation.py` — relevance / retrieval / baseline metrics
- `data/personas.json` — 5 differentiated synthetic personas with gold QA pairs

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

- NewsAPI is fetched once per cache cycle (6h) globally, not per user, to stay
  inside the free-tier quota.
- `MAX_ARTICLES_PER_PUSH` in config bounds the daily digest size.
- Sent-article IDs are tracked per user so users never receive the same story
  twice.
- The RAG answer prompt caps replies near the SMS character budget.
