"""Offline evaluation harness.

Two metrics are produced against the synthetic personas in data/personas.json:

  1. Retrieval hit-rate@k — does retrieval return the ground-truth article
     for each labeled QA pair?
  2. Baseline comparison — embedding cosine similarity and TF-IDF similarity
     of articles returned by the TF-IDF-only ranker vs. the full LLM ranker,
     scored against each persona summary.

Usage:
    python -m eval.evaluation --all
    python -m eval.evaluation --retrieval
    python -m eval.evaluation --baseline
"""
from __future__ import annotations

import argparse
import json
import logging
import time
import uuid
from pathlib import Path
from typing import List

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity as sk_cosine

from core.article_fetcher import Article, fetch_articles, llm_rank, tfidf_rank
from core.rag_engine import _get_embedder, index_articles, retrieve
from core.user_profile import User, generate_persona_summary

log = logging.getLogger(__name__)

PERSONAS_PATH = Path(__file__).resolve().parent.parent / "data" / "personas.json"


def _load_personas() -> List[dict]:
    with PERSONAS_PATH.open() as f:
        return json.load(f)


def _build_ephemeral_user(persona: dict) -> User:
    """In-memory user for evaluation — not persisted to SQLite."""
    interests = persona["interests"]
    summary = persona.get("persona_summary") or generate_persona_summary(interests)
    return User(
        user_id=f"eval_{persona['id']}_{uuid.uuid4().hex[:6]}",
        phone=f"+1555{persona['id']:07d}",
        interests=interests,
        persona_summary=summary,
        onboarding_state="DONE",
    )


def _mean_embedding_sim(embedder, persona_text: str, arts: List[Article]) -> float:
    if not arts:
        return 0.0
    p_vec = embedder.encode([persona_text], normalize_embeddings=True)
    a_vecs = embedder.encode(
        [f"{a.title} {a.summary}" for a in arts], normalize_embeddings=True
    )
    return float(np.mean((a_vecs @ p_vec.T).ravel()))


def _mean_tfidf_sim(persona_text: str, arts: List[Article]) -> float:
    if not arts:
        return 0.0
    texts = [persona_text] + [f"{a.title} {a.summary}" for a in arts]
    vectorizer = TfidfVectorizer(stop_words="english", max_features=10_000)
    matrix = vectorizer.fit_transform(texts)
    return float(np.mean(sk_cosine(matrix[0:1], matrix[1:]).ravel()))


# ---------- metric 1: retrieval hit-rate ----------

def eval_retrieval(k: int = 4) -> dict:
    from core.rag_engine import drop_user_collection
    personas = _load_personas()
    hits, total = 0, 0
    per_persona = {}

    for persona in personas:
        qa_pairs = persona.get("qa_pairs", [])
        if not qa_pairs:
            continue
        user = _build_ephemeral_user(persona)

        corpus: List[Article] = []
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

        # add distractors so retrieval isn't trivially correct
        for distractor in fetch_articles()[:10]:
            corpus.append(distractor)

        try:
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
        finally:
            drop_user_collection(user.user_id)

    return {
        "metric": "retrieval_hit_rate_at_k",
        "k": k,
        "overall_hit_rate": hits / total if total else 0,
        "per_persona": per_persona,
    }


# ---------- metric 2: baseline comparison ----------

def eval_baseline(top_k: int = 5) -> dict:
    """Compare TF-IDF ranker vs LLM ranker using embedding cosine similarity
    and TF-IDF similarity as scoring signals — no LLM judge calls needed."""
    embedder = _get_embedder()
    personas = _load_personas()
    articles = fetch_articles()

    per_persona = {}
    for persona in personas:
        user = _build_ephemeral_user(persona)
        persona_text = user.persona_summary

        tfidf_out = tfidf_rank(articles, persona_text, top_k=top_k)
        llm_out = llm_rank(articles, user, top_k=top_k)

        per_persona[persona["id"]] = {
            "name": persona["name"],
            "tfidf_ranker": {
                "embedding_similarity": round(_mean_embedding_sim(embedder, persona_text, tfidf_out), 4),
                "tfidf_similarity": round(_mean_tfidf_sim(persona_text, tfidf_out), 4),
            },
            "llm_ranker": {
                "embedding_similarity": round(_mean_embedding_sim(embedder, persona_text, llm_out), 4),
                "tfidf_similarity": round(_mean_tfidf_sim(persona_text, llm_out), 4),
            },
        }

    return {
        "metric": "baseline_comparison",
        "top_k": top_k,
        "per_persona": per_persona,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--retrieval", action="store_true")
    parser.add_argument("--baseline", action="store_true")
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--out", default=None, help="write JSON results to this path")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    out = {"timestamp": time.time()}
    if args.retrieval or args.all:
        log.info("running retrieval eval...")
        out["retrieval"] = eval_retrieval()
    if args.baseline or args.all:
        log.info("running baseline comparison...")
        out["baseline"] = eval_baseline()

    if not any([args.retrieval, args.baseline, args.all]):
        parser.error("pick at least one of --retrieval, --baseline, --all")

    dump = json.dumps(out, indent=2)
    if args.out:
        Path(args.out).write_text(dump)
        log.info("wrote %s", args.out)
    else:
        print(dump)


if __name__ == "__main__":
    main()
