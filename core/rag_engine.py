"""Retrieval-augmented answering over recently pushed articles.

Flow:
  1. `index_articles(user_id, articles)` — called after a proactive push so
     the user's personal corpus reflects what we actually sent.
  2. `handle_followup(user, query, articles=None)` — embeds the query,
     retrieves chunks with MMR, calls the LLM to answer with citations.
"""
from __future__ import annotations

import logging
import os
import re
import time
from dataclasses import dataclass
from typing import List, Optional

import chromadb
import numpy as np
from anthropic import Anthropic
from sentence_transformers import SentenceTransformer

from config import ANTHROPIC_API_KEY, CHROMA_PATH, EMBEDDING_MODEL, LLM_MODEL
from core.article_fetcher import Article, get_cached_article
from core.conv_log import log_event
from core.user_profile import User, llm_configured

log = logging.getLogger(__name__)

_MIN_CHUNK_CHARS = 80
_MAX_CHUNK_CHARS = 900
_MMR_LAMBDA = 0.6

_embedder: Optional[SentenceTransformer] = None
_chroma: Optional[chromadb.PersistentClient] = None


def _get_embedder() -> SentenceTransformer:
    global _embedder
    if _embedder is None:
        _embedder = SentenceTransformer(EMBEDDING_MODEL)
    return _embedder


def _get_chroma() -> chromadb.PersistentClient:
    global _chroma
    if _chroma is None:
        os.makedirs(CHROMA_PATH, exist_ok=True)
        _chroma = chromadb.PersistentClient(path=CHROMA_PATH)
    return _chroma


def _collection(user_id: str):
    return _get_chroma().get_or_create_collection(
        name=f"user_{user_id}",
        metadata={"hnsw:space": "cosine"},
    )


def drop_user_collection(user_id: str) -> bool:
    """Delete this user's Chroma collection. Returns True if it existed."""
    try:
        _get_chroma().delete_collection(name=f"user_{user_id}")
        return True
    except Exception:
        return False


def _chunk_paragraphs(text: str) -> List[str]:
    """Paragraph-level chunks — news articles lose too much context at the
    sentence level, so we split on blank lines and merge small fragments."""
    if not text:
        return []
    raw = re.split(r"\n\s*\n", text.strip())
    chunks: List[str] = []
    buf = ""
    for para in raw:
        para = para.strip()
        if not para:
            continue
        if len(para) > _MAX_CHUNK_CHARS:
            # over-long paragraph — split on sentence boundaries
            sentences = re.split(r"(?<=[.!?])\s+", para)
            current = ""
            for s in sentences:
                if len(current) + len(s) + 1 > _MAX_CHUNK_CHARS and current:
                    chunks.append(current.strip())
                    current = s
                else:
                    current = (current + " " + s).strip()
            if current:
                chunks.append(current.strip())
            continue
        if len(buf) + len(para) + 2 <= _MAX_CHUNK_CHARS:
            buf = (buf + "\n\n" + para).strip()
        else:
            if buf:
                chunks.append(buf)
            buf = para
    if buf:
        chunks.append(buf)
    return [c for c in chunks if len(c) >= _MIN_CHUNK_CHARS]


def index_articles(user_id: str, articles: List[Article]) -> int:
    """Embed and store paragraph chunks for this user's recent articles."""
    if not articles:
        return 0
    coll = _collection(user_id)
    embedder = _get_embedder()

    ids: List[str] = []
    docs: List[str] = []
    metadatas: List[dict] = []
    for art in articles:
        body = f"{art.title}\n\n{art.content or art.summary}"
        for i, chunk in enumerate(_chunk_paragraphs(body)):
            ids.append(f"{art.article_id}__{i}")
            docs.append(chunk)
            metadatas.append({
                "article_id": art.article_id,
                "title": art.title,
                "url": art.url,
                "source": art.source,
                "chunk_idx": i,
            })

    if not docs:
        return 0

    embeddings = embedder.encode(docs, normalize_embeddings=True).tolist()
    coll.upsert(ids=ids, documents=docs, metadatas=metadatas, embeddings=embeddings)
    return len(docs)


def topic_diverse_articles(
    articles: List[Article], interests: List[str], k: int,
) -> List[Article]:
    """Pick one article per stated interest (the article whose title+summary
    is most semantically similar to that interest), then return the top-k of
    those by their existing LLM relevance score.

    If two interests would pick the same article, the second interest falls
    through to its next-best non-claimed article — that way different topics
    yield different articles. If the per-interest winners are fewer than k
    (e.g., the user listed only 1-2 interests), we pad with the next-best
    articles by score.
    """
    if not articles:
        return []
    cleaned = [i.strip() for i in interests if i and i.strip()]
    if not cleaned:
        return sorted(articles, key=lambda a: a.score, reverse=True)[:k]

    embedder = _get_embedder()
    interest_vecs = np.asarray(embedder.encode(cleaned, normalize_embeddings=True))
    article_vecs = np.asarray(embedder.encode(
        [f"{a.title}\n{a.summary}" for a in articles],
        normalize_embeddings=True,
    ))
    sims = article_vecs @ interest_vecs.T  # (n_articles, n_interests)

    chosen_ids: set[str] = set()
    chosen: List[Article] = []
    for j in range(len(cleaned)):
        order = np.argsort(-sims[:, j])
        for idx in order:
            art = articles[int(idx)]
            if art.article_id not in chosen_ids:
                chosen.append(art)
                chosen_ids.add(art.article_id)
                break

    chosen.sort(key=lambda a: a.score, reverse=True)
    if len(chosen) >= k:
        return chosen[:k]
    fillers = [a for a in articles if a.article_id not in chosen_ids]
    fillers.sort(key=lambda a: a.score, reverse=True)
    return chosen + fillers[: k - len(chosen)]


def mmr_diversify_articles(
    articles: List[Article], k: int, lam: float = 0.5,
) -> List[Article]:
    """Pick `k` articles that balance relevance with topic diversity.

    Uses each article's existing `.score` (from the LLM relevance ranker) as
    the relevance signal, and sentence-transformer embeddings of
    title+summary as the similarity signal. Higher `lam` favors relevance,
    lower `lam` favors diversity. With lam=0.5 a topical near-duplicate has
    to outscore the next-best non-duplicate by a wide margin to be picked.
    """
    if len(articles) <= k:
        return articles
    embedder = _get_embedder()
    texts = [f"{a.title}\n{a.summary}" for a in articles]
    doc_vecs = np.asarray(embedder.encode(texts, normalize_embeddings=True))
    rel = np.array([a.score for a in articles], dtype=float)

    selected: List[int] = []
    remaining = set(range(len(articles)))
    for _ in range(k):
        best_idx, best = -1, -np.inf
        for i in remaining:
            if not selected:
                score = rel[i]
            else:
                max_sim = max(float(doc_vecs[i] @ doc_vecs[j]) for j in selected)
                score = lam * rel[i] - (1 - lam) * max_sim
            if score > best:
                best, best_idx = score, i
        selected.append(best_idx)
        remaining.remove(best_idx)
    return [articles[i] for i in selected]


def _mmr(query_vec: np.ndarray, doc_vecs: np.ndarray, k: int, lam: float) -> List[int]:
    """Maximal Marginal Relevance — diversifies retrieval so we don't pull
    three near-duplicate chunks from the same article."""
    n = doc_vecs.shape[0]
    if n == 0:
        return []
    k = min(k, n)
    sim_to_query = doc_vecs @ query_vec
    selected: List[int] = []
    remaining = set(range(n))
    for _ in range(k):
        best_idx, best_score = -1, -np.inf
        for i in remaining:
            if not selected:
                score = sim_to_query[i]
            else:
                max_sim_to_selected = max(
                    float(doc_vecs[i] @ doc_vecs[j]) for j in selected
                )
                score = lam * sim_to_query[i] - (1 - lam) * max_sim_to_selected
            if score > best_score:
                best_score, best_idx = score, i
        selected.append(best_idx)
        remaining.remove(best_idx)
    return selected


@dataclass
class RetrievedChunk:
    text: str
    article_id: str
    title: str
    url: str
    source: str
    score: float


def retrieve(user_id: str, query: str, k: int = 4, fetch_k: int = 20) -> List[RetrievedChunk]:
    coll = _collection(user_id)
    if coll.count() == 0:
        return []
    embedder = _get_embedder()
    q_vec = embedder.encode([query], normalize_embeddings=True)[0]

    res = coll.query(
        query_embeddings=[q_vec.tolist()],
        n_results=min(fetch_k, coll.count()),
        include=["documents", "metadatas", "embeddings", "distances"],
    )
    docs = res["documents"][0]
    metas = res["metadatas"][0]
    emb = np.array(res["embeddings"][0])
    distances = res["distances"][0]

    picked = _mmr(np.array(q_vec), emb, k=k, lam=_MMR_LAMBDA)
    chunks = []
    for idx in picked:
        chunks.append(RetrievedChunk(
            text=docs[idx],
            article_id=metas[idx]["article_id"],
            title=metas[idx]["title"],
            url=metas[idx]["url"],
            source=metas[idx]["source"],
            score=1.0 - float(distances[idx]),
        ))
    return chunks


_ANSWER_PROMPT = """\
You are a knowledgeable friend texting back over iMessage. Be direct, conversational, and concise — 2-4 short paragraphs max. Write in plain prose, no markdown, no bullet points, no headers, no bold or asterisks.

Use ONLY the retrieved context to answer. Cite sources naturally in the text, e.g. "according to CNBC" or "per the BBC". If the context is limited, say what you know and leave it there — do not tell the user to go read other sources.

User's recent conversation:
{history}

User's question:
{query}

Retrieved context:
{context}
"""


def _format_history(history: List[dict], max_turns: int = 4) -> str:
    if not history:
        return "(no prior messages)"
    recent = history[-max_turns * 2:]
    return "\n".join(f"{m['role']}: {m['content']}" for m in recent)


def _format_context(chunks: List[RetrievedChunk]) -> str:
    if not chunks:
        return "(no relevant passages)"
    return "\n\n".join(
        f"[{c.source}] {c.title}\n{c.text}" for c in chunks
    )


def _supplement_from_external(user_id: str, query: str) -> None:
    """Search NewsAPI for articles relevant to the query, index them so the
    RAG engine can use them to answer questions beyond the sent digest."""
    from core.article_fetcher import search_articles
    articles = search_articles(query, top_k=10)
    if articles:
        log.info("indexed %d external articles for query %r", len(articles), query)
        index_articles(user_id, articles)


def handle_followup(user: User, query: str, articles: Optional[List[Article]] = None) -> str:
    """Answer a user follow-up, indexing articles on-demand if provided.

    ``articles`` is optional so the function can be called both from the
    proactive flow (articles available in-memory) and from the SMS webhook
    (articles must be pulled from the cache).
    """
    if articles:
        index_articles(user.user_id, articles)

    # Always search externally so the answer isn't limited to the digest.
    _supplement_from_external(user.user_id, query)

    chunks = retrieve(user.user_id, query, k=6)

    if not chunks:
        return ("I couldn't find any articles relevant to that question. "
                "Try rephrasing, or ask about a topic from your recent digest.")

    def _fallback() -> str:
        seen = set()
        lines = ["Here's what I found in your recent articles:"]
        for chunk in chunks[:3]:
            if chunk.article_id in seen:
                continue
            seen.add(chunk.article_id)
            lines.append(f"\n{chunk.title} ({chunk.source})\n{chunk.url}")
        lines.append("\nAsk me a more specific question and I'll try again.")
        return "\n".join(lines)

    if not llm_configured():
        return _fallback()

    client = Anthropic(api_key=ANTHROPIC_API_KEY)
    prompt = _ANSWER_PROMPT.format(
        history=_format_history(user.conversation_history),
        query=query,
        context=_format_context(chunks),
    )
    for attempt in range(3):
        try:
            msg = client.messages.create(
                model=LLM_MODEL,
                max_tokens=1500,
                messages=[{"role": "user", "content": prompt}],
            )
            answer = msg.content[0].text.strip()
            log_event(
                "llm_call",
                user_id=user.user_id, phone=user.phone,
                purpose="rag_answer", model=LLM_MODEL,
                prompt=prompt, response=answer,
                retrieved_ids=[c.article_id for c in chunks],
            )
            return answer
        except Exception as e:
            if "rate_limit" in str(e).lower() and attempt < 2:
                wait = 10 * (attempt + 1)
                log.warning("Anthropic rate limit hit, retrying in %ds (attempt %d)", wait, attempt + 1)
                time.sleep(wait)
            else:
                log.warning("RAG answer LLM call failed (%s)", e, exc_info=True)
                log_event(
                    "llm_call",
                    user_id=user.user_id, phone=user.phone,
                    purpose="rag_answer", model=LLM_MODEL,
                    prompt=prompt, response=f"<ERROR: {e}>",
                )
                return _fallback()


def rehydrate_articles(article_ids: List[str]) -> List[Article]:
    """Pull articles out of the fetcher's cache by id — used when the RAG
    engine is asked to index articles referenced only by id (e.g. from the
    user's sent-articles history)."""
    out = []
    for aid in article_ids:
        art = get_cached_article(aid)
        if art is not None:
            out.append(art)
    return out
