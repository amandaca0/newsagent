"""Article fetching, caching, and user-aware ranking.

Two rankers are exposed:
  - tfidf_rank: baseline keyword-similarity ranker (also used in eval)
  - llm_rank: Claude-based reranker using the persona summary

The proactive pipeline uses llm_rank over the TF-IDF top-K to keep LLM
spend bounded.
"""
from __future__ import annotations

import hashlib
import html
import json
import logging
import os
import re
import sqlite3
import time
from dataclasses import asdict, dataclass, field
from typing import Iterable, List, Optional

from anthropic import Anthropic
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from config import ANTHROPIC_API_KEY, DB_PATH, LLM_MODEL, NEWSAPI_KEY
from core.user_profile import User, already_sent_ids, llm_configured

log = logging.getLogger(__name__)


def _strip_html(text: str) -> str:
    text = re.sub(r"<[^>]+>", " ", text)
    text = html.unescape(text)
    return re.sub(r"\s+", " ", text).strip()


ARTICLE_STORE_DAYS = 7         # keep articles for 7 days
FETCH_INTERVAL_SECONDS = 7200  # fetch at most every 2h (7 categories * 12/day = 84 < 100/day quota)
TFIDF_PREFILTER_K = 15
MIN_RELEVANCE_SCORE = 0.3

# NewsAPI's seven supported categories for get_top_headlines. Fanning out
# across all of them gives ~700 raw / ~300+ unique articles per fetch cycle.
NEWS_CATEGORIES = [
    "business", "entertainment", "general", "health",
    "science", "sports", "technology",
]


@dataclass
class Article:
    article_id: str
    title: str
    source: str
    url: str
    published_at: Optional[str]
    summary: str
    content: str = ""
    score: float = 0.0
    rationale: str = ""

    def to_dict(self) -> dict:
        return asdict(self)

    @property
    def text_for_ranking(self) -> str:
        return f"{self.title}\n\n{self.summary}\n\n{self.content}".strip()


_ARTICLE_TABLE = """
CREATE TABLE IF NOT EXISTS articles (
    article_id   TEXT PRIMARY KEY,
    title        TEXT NOT NULL,
    source       TEXT NOT NULL,
    url          TEXT NOT NULL,
    published_at TEXT,
    summary      TEXT NOT NULL,
    content      TEXT NOT NULL,
    fetched_at   REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_articles_fetched ON articles(fetched_at DESC);
"""


def _connect() -> sqlite3.Connection:
    os.makedirs(os.path.dirname(DB_PATH) or ".", exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_cache() -> None:
    with _connect() as conn:
        conn.executescript(_ARTICLE_TABLE)
        # migrate old articles_cache table if it exists
        tables = {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()}
        if "articles_cache" in tables and "articles" not in tables:
            conn.execute("ALTER TABLE articles_cache RENAME TO articles")


def _article_id(url: str) -> str:
    return hashlib.sha1(url.encode("utf-8")).hexdigest()[:16]


def _store_articles(articles: Iterable[Article]) -> None:
    now = time.time()
    rows = [
        (a.article_id, a.title, a.source, a.url, a.published_at or "",
         a.summary, a.content, now)
        for a in articles
    ]
    if not rows:
        return
    with _connect() as conn:
        conn.executemany(
            "INSERT OR IGNORE INTO articles "
            "(article_id, title, source, url, published_at, summary, content, fetched_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            rows,
        )


def _get_articles() -> List[Article]:
    cutoff = time.time() - ARTICLE_STORE_DAYS * 24 * 60 * 60
    with _connect() as conn:
        rows = conn.execute(
            "SELECT * FROM articles WHERE fetched_at >= ? ORDER BY fetched_at DESC",
            (cutoff,),
        ).fetchall()
    return [
        Article(
            article_id=r["article_id"],
            title=r["title"],
            source=r["source"],
            url=r["url"],
            published_at=r["published_at"] or None,
            summary=r["summary"],
            content=r["content"],
        )
        for r in rows
    ]


def _last_fetch_time() -> float:
    with _connect() as conn:
        row = conn.execute("SELECT MAX(fetched_at) as last FROM articles").fetchone()
        return row["last"] or 0.0


def get_cached_article(article_id: str) -> Optional[Article]:
    with _connect() as conn:
        row = conn.execute(
            "SELECT * FROM articles WHERE article_id = ?", (article_id,)
        ).fetchone()
    if row is None:
        return None
    return Article(
        article_id=row["article_id"],
        title=row["title"],
        source=row["source"],
        url=row["url"],
        published_at=row["published_at"] or None,
        summary=row["summary"],
        content=row["content"],
    )


def _fetch_newsapi() -> List[Article]:
    if not NEWSAPI_KEY:
        return []
    try:
        from newsapi import NewsApiClient
    except ImportError:
        log.warning("newsapi-python not installed; skipping")
        return []
    client = NewsApiClient(api_key=NEWSAPI_KEY)
    out: List[Article] = []
    for category in NEWS_CATEGORIES:
        try:
            resp = client.get_top_headlines(
                language="en", category=category, page_size=100,
            )
        except Exception as e:
            log.warning("NewsAPI fetch failed for category=%s: %s", category, e)
            continue
        for item in resp.get("articles", []):
            url = item.get("url", "")
            if not url:
                continue
            out.append(Article(
                article_id=_article_id(url),
                title=item.get("title") or "",
                source=(item.get("source") or {}).get("name", "NewsAPI"),
                url=url,
                published_at=item.get("publishedAt"),
                summary=item.get("description") or "",
                content=item.get("content") or "",
            ))
    log.info("NewsAPI fetched %d raw articles across %d categories", len(out), len(NEWS_CATEGORIES))
    return out


def search_articles(query: str, top_k: int = 10) -> List[Article]:
    """Search NewsAPI for articles matching a specific query and store them."""
    if not NEWSAPI_KEY:
        return []
    try:
        from newsapi import NewsApiClient
    except ImportError:
        return []
    client = NewsApiClient(api_key=NEWSAPI_KEY)
    try:
        resp = client.get_everything(
            q=query, language="en", page_size=top_k, sort_by="relevancy"
        )
    except Exception as e:
        log.warning("NewsAPI search failed for %r: %s", query, e)
        return []
    out: List[Article] = []
    for item in resp.get("articles", []):
        url = item.get("url", "")
        if not url:
            continue
        out.append(Article(
            article_id=_article_id(url),
            title=_strip_html(item.get("title") or ""),
            source=(item.get("source") or {}).get("name", "NewsAPI"),
            url=url,
            published_at=item.get("publishedAt"),
            summary=_strip_html(item.get("description") or ""),
            content=_strip_html(item.get("content") or ""),
        ))
    _store_articles(out)
    return out


def fetch_articles(force_refresh: bool = False) -> List[Article]:
    """Fetch new articles into the persistent store and return all recent articles.

    New articles are fetched at most once per hour to protect API rate limits.
    force_refresh bypasses the interval check and fetches immediately.
    Articles accumulate for ARTICLE_STORE_DAYS days, giving the agent a rich pool.
    """
    since_last = time.time() - _last_fetch_time()
    if not force_refresh and since_last < FETCH_INTERVAL_SECONDS:
        return _get_articles()

    fetched = _fetch_newsapi()
    seen: set[str] = set()
    unique: List[Article] = []
    for a in fetched:
        if a.article_id in seen or not a.title:
            continue
        seen.add(a.article_id)
        unique.append(a)
    _store_articles(unique)
    return _get_articles()


def tfidf_rank(articles: List[Article], query: str, top_k: int = 10) -> List[Article]:
    if not articles or not query.strip():
        return articles[:top_k]
    corpus = [query] + [a.text_for_ranking for a in articles]
    vectorizer = TfidfVectorizer(stop_words="english", max_features=20_000)
    matrix = vectorizer.fit_transform(corpus)
    sims = cosine_similarity(matrix[0:1], matrix[1:]).ravel()
    ranked = sorted(zip(sims, articles), key=lambda t: t[0], reverse=True)
    out = []
    for score, art in ranked[:top_k]:
        art.score = float(score)
        out.append(art)
    return out


_RANK_PROMPT = """\
You are a news-relevance scorer. Given a user's interest profile and a list of candidate articles, score each article from 0.0-1.0 for how well it matches this user. A 1.0 is a must-read for this specific user; 0.0 is irrelevant.

Reply ONLY with JSON in this exact shape:
{{"scores": [{{"article_id": "...", "score": 0.87, "rationale": "<one sentence>"}}, ...]}}

User persona:
{persona}

Candidate articles:
{candidates}
"""


def _format_candidates(articles: List[Article]) -> str:
    lines = []
    for a in articles:
        snippet = (a.summary or a.content)[:400].replace("\n", " ")
        lines.append(f"- id={a.article_id}\n  title: {a.title}\n  snippet: {snippet}")
    return "\n".join(lines)


def llm_rank(articles: List[Article], user: User, top_k: int = 5) -> List[Article]:
    """Rerank with Claude using the persona summary.

    Falls back to TF-IDF ranking if the API is unavailable or the response
    can't be parsed.
    """
    if not articles:
        return []
    persona = user.persona_summary or ", ".join(user.interests)
    if not persona:
        return articles[:top_k]

    # prefilter with TF-IDF to bound the LLM context and cost
    prefiltered = tfidf_rank(articles, persona, top_k=TFIDF_PREFILTER_K)

    if not llm_configured():
        return prefiltered[:top_k]

    client = Anthropic(api_key=ANTHROPIC_API_KEY)
    prompt = _RANK_PROMPT.format(
        persona=persona,
        candidates=_format_candidates(prefiltered),
    )
    try:
        msg = client.messages.create(
            model=LLM_MODEL,
            max_tokens=4096,
            messages=[{"role": "user", "content": prompt}],
        )
        raw = msg.content[0].text
        from core.conv_log import log_event
        log_event(
            "llm_call",
            user_id=user.user_id, phone=user.phone,
            purpose="article_rank", model=LLM_MODEL,
            prompt=prompt, response=raw,
            candidate_count=len(prefiltered),
        )
        data = json.loads(_extract_json(raw))
    except Exception as e:
        log.warning("llm_rank parse failed, falling back to TF-IDF: %s", e)
        return prefiltered[:top_k]

    by_id = {a.article_id: a for a in prefiltered}
    scored: List[Article] = []
    for entry in data.get("scores", []):
        aid = entry.get("article_id")
        if aid in by_id:
            art = by_id[aid]
            art.score = float(entry.get("score", 0.0))
            art.rationale = entry.get("rationale", "")
            scored.append(art)
    scored.sort(key=lambda a: a.score, reverse=True)
    return [a for a in scored[:top_k] if a.score >= MIN_RELEVANCE_SCORE]


def _extract_json(text: str) -> str:
    import re
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1:
        raise ValueError("no JSON object in response")
    raw = text[start : end + 1]
    # strip trailing commas before ] or } (common LLM output quirk)
    return re.sub(r",\s*([}\]])", r"\1", raw)


def fetch_and_rank_articles(user: User, top_k: int = 5, force_refresh: bool = False) -> List[Article]:
    """Main interface consumed by the agent graph."""
    articles = fetch_articles(force_refresh=force_refresh)
    already = already_sent_ids(user.user_id)
    fresh = [a for a in articles if a.article_id not in already]
    return llm_rank(fresh, user, top_k=top_k)
