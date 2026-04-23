"""Offline evaluation harness.

Three metrics are produced against the synthetic personas in data/personas.json:

  1. Relevance (LLM-as-Judge, 1-5 with rationale) — scores articles the
     pipeline pushed.
  2. Retrieval hit-rate@k — does retrieval return the ground-truth article
     for each labeled QA pair?
  3. Baseline comparison — relevance of TF-IDF-only pipeline vs. full
     LLM-ranked pipeline, on the same candidate pool per persona.

Usage:
    python -m eval.evaluation --all
    python -m eval.evaluation --relevance
    python -m eval.evaluation --retrieval
    python -m eval.evaluation --baseline
"""
from __future__ import annotations

import argparse
import json
import logging
import statistics
import time
import uuid
from pathlib import Path
from typing import List

from anthropic import Anthropic

from config import ANTHROPIC_API_KEY, JUDGE_MODEL
from core.article_fetcher import (
    Article,
    fetch_articles,
    llm_rank,
    tfidf_rank,
)
from core.rag_engine import index_articles, retrieve
from core.user_profile import User, generate_persona_summary

log = logging.getLogger(__name__)

PERSONAS_PATH = Path(__file__).resolve().parent.parent / "data" / "personas.json"

_JUDGE_PROMPT = """\
You are scoring whether a news article is relevant to a user's stated interests.

User persona:
{persona}

Article:
Title: {title}
Source: {source}
Summary: {summary}

Think step by step about what this user cares about and whether this article matches. Then output ONLY a single JSON object:
{{"rationale": "<one-to-two sentence chain of thought>", "score": <integer 1-5>}}

Scoring rubric:
5 = must-read for this exact user
4 = clearly relevant
3 = tangentially relevant
2 = weakly related
1 = irrelevant
"""


def _load_personas() -> List[dict]:
    with PERSONAS_PATH.open() as f:
        return json.load(f)


def _build_ephemeral_user(persona: dict) -> User:
    """Spin up an in-memory user object for evaluation.

    We intentionally don't persist these — we don't want eval runs to
    pollute the production user table.
    """
    interests = persona["interests"]
    summary = persona.get("persona_summary") or generate_persona_summary(interests)
    return User(
        user_id=f"eval_{persona['id']}_{uuid.uuid4().hex[:6]}",
        phone=f"+1555{persona['id']:07d}",
        interests=interests,
        persona_summary=summary,
        onboarding_state="DONE",
    )


def _judge_relevance(client: Anthropic, persona: str, article: Article) -> dict:
    prompt = _JUDGE_PROMPT.format(
        persona=persona,
        title=article.title,
        source=article.source,
        summary=(article.summary or article.content)[:600],
    )
    resp = client.messages.create(
        model=JUDGE_MODEL,
        max_tokens=400,
        messages=[{"role": "user", "content": prompt}],
    )
    text = resp.content[0].text
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end == -1:
        return {"score": 0, "rationale": f"unparsable: {text[:120]}"}
    try:
        return json.loads(text[start : end + 1])
    except Exception as e:
        return {"score": 0, "rationale": f"parse error: {e}"}


# ---------- metric 1: relevance ----------

def eval_relevance(top_k: int = 5) -> dict:
    if not ANTHROPIC_API_KEY:
        return {"metric": "relevance_llm_judge", "skipped": "no ANTHROPIC_API_KEY"}
    client = Anthropic(api_key=ANTHROPIC_API_KEY)
    personas = _load_personas()
    articles = fetch_articles()

    per_persona = {}
    all_scores: List[int] = []
    for persona in personas:
        user = _build_ephemeral_user(persona)
        ranked = llm_rank(articles, user, top_k=top_k)
        scored = []
        for art in ranked:
            j = _judge_relevance(client, user.persona_summary, art)
            scored.append({
                "article_id": art.article_id,
                "title": art.title,
                "rank_score": art.score,
                "judge_score": j.get("score", 0),
                "rationale": j.get("rationale", ""),
            })
            all_scores.append(int(j.get("score", 0)))
        per_persona[persona["id"]] = {
            "name": persona["name"],
            "articles": scored,
            "mean_score": statistics.mean([s["judge_score"] for s in scored]) if scored else 0,
        }

    return {
        "metric": "relevance_llm_judge",
        "top_k": top_k,
        "overall_mean": statistics.mean(all_scores) if all_scores else 0,
        "per_persona": per_persona,
    }


# ---------- metric 2: retrieval hit-rate ----------

def eval_retrieval(k: int = 4) -> dict:
    personas = _load_personas()
    hits, total = 0, 0
    per_persona = {}

    for persona in personas:
        qa_pairs = persona.get("qa_pairs", [])
        if not qa_pairs:
            continue
        user = _build_ephemeral_user(persona)

        corpus: List[Article] = []
        gold_ids = set()
        for qa in qa_pairs:
            aid = f"eval_gold_{uuid.uuid4().hex[:10]}"
            corpus.append(Article(
                article_id=aid,
                title=qa["article_title"],
                source=qa.get("source", "EvalSource"),
                url=qa.get("url", f"https://example.test/{aid}"),
                published_at=None,
                summary=qa["article_content"][:300],
                content=qa["article_content"],
            ))
            qa["_gold_article_id"] = aid
            gold_ids.add(aid)

        # add distractors so retrieval isn't trivially correct
        for distractor in fetch_articles()[:10]:
            corpus.append(distractor)

        index_articles(user.user_id, corpus)

        persona_hits, persona_total = 0, 0
        details = []
        for qa in qa_pairs:
            chunks = retrieve(user.user_id, qa["question"], k=k)
            retrieved_ids = {c.article_id for c in chunks}
            hit = qa["_gold_article_id"] in retrieved_ids
            persona_hits += int(hit)
            persona_total += 1
            details.append({
                "question": qa["question"],
                "hit": hit,
                "retrieved": [c.title for c in chunks],
            })
        per_persona[persona["id"]] = {
            "name": persona["name"],
            "hit_rate": persona_hits / persona_total if persona_total else 0,
            "details": details,
        }
        hits += persona_hits
        total += persona_total

    return {
        "metric": "retrieval_hit_rate_at_k",
        "k": k,
        "overall_hit_rate": hits / total if total else 0,
        "per_persona": per_persona,
    }


# ---------- metric 3: baseline comparison ----------

def eval_baseline(top_k: int = 5) -> dict:
    if not ANTHROPIC_API_KEY:
        return {"metric": "baseline_comparison", "skipped": "no ANTHROPIC_API_KEY"}
    client = Anthropic(api_key=ANTHROPIC_API_KEY)
    personas = _load_personas()
    articles = fetch_articles()

    comparison = {}
    for persona in personas:
        user = _build_ephemeral_user(persona)
        persona_text = user.persona_summary

        tfidf_out = tfidf_rank(articles, persona_text, top_k=top_k)
        llm_out = llm_rank(articles, user, top_k=top_k)

        def score_set(arts):
            scores = [_judge_relevance(client, persona_text, a).get("score", 0) for a in arts]
            return statistics.mean(scores) if scores else 0

        comparison[persona["id"]] = {
            "name": persona["name"],
            "tfidf_mean": score_set(tfidf_out),
            "llm_mean": score_set(llm_out),
        }

    return {
        "metric": "baseline_comparison",
        "top_k": top_k,
        "per_persona": comparison,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--relevance", action="store_true")
    parser.add_argument("--retrieval", action="store_true")
    parser.add_argument("--baseline", action="store_true")
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--out", default=None, help="write JSON results to this path")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    if not ANTHROPIC_API_KEY:
        log.warning("ANTHROPIC_API_KEY not set — relevance and baseline will be skipped")

    out = {"timestamp": time.time()}
    if args.relevance or args.all:
        log.info("running relevance eval...")
        out["relevance"] = eval_relevance()
        if out["relevance"].get("skipped"):
            log.warning("relevance skipped: %s", out["relevance"]["skipped"])
    if args.retrieval or args.all:
        log.info("running retrieval eval...")
        out["retrieval"] = eval_retrieval()
    if args.baseline or args.all:
        log.info("running baseline comparison...")
        out["baseline"] = eval_baseline()
        if out["baseline"].get("skipped"):
            log.warning("baseline skipped: %s", out["baseline"]["skipped"])

    if not any([args.relevance, args.retrieval, args.baseline, args.all]):
        parser.error("pick at least one of --relevance, --retrieval, --baseline, --all")

    dump = json.dumps(out, indent=2)
    if args.out:
        Path(args.out).write_text(dump)
        log.info("wrote %s", args.out)
    else:
        print(dump)


if __name__ == "__main__":
    main()
