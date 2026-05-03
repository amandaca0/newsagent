"""Inline evaluation hooks fired on every live digest send and follow-up reply.

Toggled by config.EVAL_MODE. When off, eval_enabled() returns False and the
hooks in agent/graph.py short-circuit before doing any work.

Records are appended to config.EVAL_LOG_PATH as JSONL — one event per line.
Two event_type values: "digest" and "response".

All public functions are wrapped in try/except so a broken evaluator can never
take down a real user-facing send.
"""
from __future__ import annotations

import json
import logging
import os
import re
import time
from typing import List, Optional

import numpy as np

from config import EVAL_LOG_PATH, EVAL_MODE
from core.article_fetcher import Article
from core.llm import complete, llm_configured
from core.rag_engine import _get_embedder
from core.user_profile import User

log = logging.getLogger(__name__)


def eval_enabled() -> bool:
    return EVAL_MODE


# ---------- digest evaluation ----------

def evaluate_digest(user: User, articles: List[Article]) -> Optional[dict]:
    """Score each article in a freshly-sent digest:
      - cosine similarity to the user's persona summary
      - best-matching interest topic + cosine similarity to that topic
    Appends a single record to the eval log and returns it."""
    try:
        if not articles:
            return None
        embedder = _get_embedder()

        article_texts = [f"{a.title} {a.summary}" for a in articles]
        article_vecs = embedder.encode(article_texts, normalize_embeddings=True)

        persona_text = user.persona_summary or ""
        if persona_text:
            persona_vec = embedder.encode([persona_text], normalize_embeddings=True)
            persona_sims = (article_vecs @ persona_vec.T).ravel()
        else:
            persona_sims = np.zeros(len(articles))

        interests = list(user.interests or [])
        if interests:
            interest_vecs = embedder.encode(interests, normalize_embeddings=True)
            # (n_articles, n_interests) similarity matrix
            interest_sims = article_vecs @ interest_vecs.T
            best_idx = interest_sims.argmax(axis=1)
            best_sims = interest_sims.max(axis=1)
        else:
            best_idx = [None] * len(articles)
            best_sims = [0.0] * len(articles)

        per_article = []
        for i, art in enumerate(articles):
            best_interest = interests[best_idx[i]] if interests else None
            per_article.append({
                "article_id": art.article_id,
                "title": art.title,
                "source": art.source,
                "persona_similarity": round(float(persona_sims[i]), 4),
                "best_interest": best_interest,
                "best_interest_similarity": round(float(best_sims[i]), 4),
            })

        payload = {
            "event_type": "digest",
            "user_id": user.user_id,
            "phone": user.phone,
            "persona_summary": persona_text,
            "interests": interests,
            "articles": per_article,
        }
        _record(payload)
        return payload
    except Exception:
        log.exception("evaluate_digest failed")
        return None


# ---------- response evaluation ----------

def evaluate_response(user: User, question: str, answer: str) -> Optional[dict]:
    """Score a follow-up reply:
      - cosine similarity of answer to question
      - LLM-as-Judge scores on human_readability, conciseness, accuracy
    Appends a record and returns it."""
    try:
        embedder = _get_embedder()
        q_text = (question or "").strip()
        a_text = (answer or "").strip()

        if q_text and a_text:
            qa_vecs = embedder.encode([q_text, a_text], normalize_embeddings=True)
            qa_sim = float(qa_vecs[0] @ qa_vecs[1])
        else:
            qa_sim = 0.0

        judge_scores = _judge(q_text, a_text) if (q_text and a_text) else {
            "human_readability": None, "conciseness": None, "accuracy": None,
        }

        payload = {
            "event_type": "response",
            "user_id": user.user_id,
            "phone": user.phone,
            "question": q_text,
            "answer": a_text,
            "qa_similarity": round(qa_sim, 4),
            "judge": judge_scores,
        }
        _record(payload)
        return payload
    except Exception:
        log.exception("evaluate_response failed")
        return None


# ---------- LLM judge ----------

_JUDGE_PROMPT = """You are evaluating an AI assistant's answer to a user's question about news articles.

Question: {question}
Answer: {answer}

Score the answer on three dimensions, each from 0.0 to 1.0:
1. human_readability - clarity, structure, ease of reading
2. conciseness       - avoids unnecessary verbosity
3. accuracy          - does it actually answer the question that was asked?

Respond with ONLY a JSON object, no prose:
{{"human_readability": 0.x, "conciseness": 0.x, "accuracy": 0.x}}"""


_NULL_SCORES = {"human_readability": None, "conciseness": None, "accuracy": None}


def _judge(question: str, answer: str) -> dict:
    if not llm_configured():
        return dict(_NULL_SCORES)
    try:
        prompt = _JUDGE_PROMPT.format(question=question, answer=answer)
        raw = complete(prompt, max_tokens=200, purpose="judge").strip()
        return _parse_judge_json(raw)
    except Exception:
        log.exception("judge call failed")
        return dict(_NULL_SCORES)


def _parse_judge_json(raw: str) -> dict:
    match = re.search(r"\{.*\}", raw, re.DOTALL)
    if not match:
        return dict(_NULL_SCORES)
    try:
        parsed = json.loads(match.group(0))
    except json.JSONDecodeError:
        return dict(_NULL_SCORES)
    out = {}
    for key in ("human_readability", "conciseness", "accuracy"):
        val = parsed.get(key)
        if isinstance(val, (int, float)):
            out[key] = round(max(0.0, min(1.0, float(val))), 3)
        else:
            out[key] = None
    return out


# ---------- file append ----------

def _record(payload: dict) -> None:
    payload = {"timestamp": time.time(), **payload}
    parent = os.path.dirname(EVAL_LOG_PATH)
    if parent:
        os.makedirs(parent, exist_ok=True)
    block = json.dumps(payload, ensure_ascii=False, indent=2)
    with open(EVAL_LOG_PATH, "a", encoding="utf-8") as f:
        f.write(block + "\n\n")
