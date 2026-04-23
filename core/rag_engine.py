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
from dataclasses import dataclass
from typing import List, Optional

import chromadb
import html
import numpy as np
from groq import Groq
from html.parser import HTMLParser
from sentence_transformers import SentenceTransformer

from config import GROQ_API_KEY, CHROMA_PATH, EMBEDDING_MODEL, LLM_MODEL
from core.article_fetcher import Article, get_cached_article
from core.user_profile import User, llm_configured

log = logging.getLogger(__name__)


class _HTMLStripper(HTMLParser):
    def __init__(self):
        super().__init__()
        self._parts: list[str] = []

    def handle_data(self, data: str) -> None:
        self._parts.append(data)

    def get_text(self) -> str:
        return " ".join(self._parts)


def _strip_html(text: str) -> str:
    if "<" not in text:
        return text
    stripper = _HTMLStripper()
    stripper.feed(text)
    return html.unescape(stripper.get_text())


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
        body = _strip_html(f"{art.title}\n\n{art.content or art.summary}")
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
You are a helpful news assistant answering a user's follow-up question about articles they recently received.

Guidelines:
- Answer in 2-4 sentences. Be direct and specific — do not repeat the question.
- Cite sources inline as [Source Name] after each claim drawn from them.
- If the context does not contain enough detail to fully answer, say so briefly and tell the user what the article does cover.
- Do not pad the answer with filler phrases like "Based on the articles..." or "According to the context...".

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


def handle_followup(user: User, query: str, articles: Optional[List[Article]] = None) -> str:
    """Answer a user follow-up, indexing articles on-demand if provided.

    ``articles`` is optional so the function can be called both from the
    proactive flow (articles available in-memory) and from the SMS webhook
    (articles must be pulled from the cache).
    """
    if articles:
        index_articles(user.user_id, articles)

    chunks = retrieve(user.user_id, query, k=4)

    if not chunks:
        return ("I don't have any recent articles indexed for you yet. "
                "Once I push your morning digest, ask me follow-ups about it.")

    def _fallback() -> str:
        top = chunks[0]
        snippet = top.text[:300].rsplit(" ", 1)[0]
        return f"From {top.source} — {top.title}: {snippet}"

    if not llm_configured():
        return _fallback()

    try:
        client = Groq(api_key=GROQ_API_KEY)
        prompt = _ANSWER_PROMPT.format(
            history=_format_history(user.conversation_history),
            query=query,
            context=_format_context(chunks),
        )
        resp = client.chat.completions.create(
            model=LLM_MODEL,
            max_tokens=800,
            messages=[{"role": "user", "content": prompt}],
        )
        return resp.choices[0].message.content.strip()
    except Exception as e:
        log.warning("RAG answer LLM call failed (%s); returning top chunk", e)
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
