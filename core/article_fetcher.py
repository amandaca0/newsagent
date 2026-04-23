"""Article fetching, caching, and user-aware ranking.

Two rankers are exposed:
  - tfidf_rank: baseline keyword-similarity ranker (also used in eval)
  - llm_rank: Claude-based reranker using the persona summary

The proactive pipeline uses llm_rank over the TF-IDF top-K to keep LLM
spend bounded.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import sqlite3
import time
from dataclasses import asdict, dataclass, field
from typing import Iterable, List, Optional

import feedparser
from anthropic import Anthropic
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from config import ANTHROPIC_API_KEY, DB_PATH, LLM_MODEL, NEWSAPI_KEY
from core.user_profile import User, already_sent_ids, llm_configured

log = logging.getLogger(__name__)

DEFAULT_RSS_FEEDS = [
    "https://feeds.arstechnica.com/arstechnica/index",
    "https://www.theverge.com/rss/index.xml",
    "https://hnrss.org/frontpage",
    "https://rss.nytimes.com/services/xml/rss/nyt/Technology.xml",
    "https://feeds.bbci.co.uk/news/world/rss.xml",
]

CACHE_TTL_SECONDS = 60 * 60 * 6
TFIDF_PREFILTER_K = 40


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
CREATE TABLE IF NOT EXISTS articles_cache (
    article_id   TEXT PRIMARY KEY,
    title        TEXT NOT NULL,
    source       TEXT NOT NULL,
    url          TEXT NOT NULL,
    published_at TEXT,
    summary      TEXT NOT NULL,
    content      TEXT NOT NULL,
    fetched_at   REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_articles_fetched ON articles_cache(fetched_at DESC);
"""


def _connect() -> sqlite3.Connection:
    os.makedirs(os.path.dirname(DB_PATH) or ".", exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_cache() -> None:
    with _connect() as conn:
        conn.executescript(_ARTICLE_TABLE)


def _article_id(url: str) -> str:
    return hashlib.sha1(url.encode("utf-8")).hexdigest()[:16]


def _cache_put(articles: Iterable[Article]) -> None:
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
            "INSERT OR REPLACE INTO articles_cache "
            "(article_id, title, source, url, published_at, summary, content, fetched_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            rows,
        )


def _cache_recent(max_age_seconds: int = CACHE_TTL_SECONDS) -> List[Article]:
    cutoff = time.time() - max_age_seconds
    with _connect() as conn:
        rows = conn.execute(
            "SELECT * FROM articles_cache WHERE fetched_at >= ? "
            "ORDER BY fetched_at DESC",
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


def get_cached_article(article_id: str) -> Optional[Article]:
    with _connect() as conn:
        row = conn.execute(
            "SELECT * FROM articles_cache WHERE article_id = ?", (article_id,)
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


def _fetch_rss(feed_urls: Iterable[str]) -> List[Article]:
    out: List[Article] = []
    for feed_url in feed_urls:
        try:
            parsed = feedparser.parse(feed_url)
        except Exception as e:
            log.warning("RSS fetch failed for %s: %s", feed_url, e)
            continue
        source = parsed.feed.get("title", feed_url)
        for entry in parsed.entries[:25]:
            url = entry.get("link", "")
            if not url:
                continue
            summary = entry.get("summary", "") or entry.get("description", "")
            content_pieces = entry.get("content", [])
            content = content_pieces[0].get("value", "") if content_pieces else summary
            out.append(Article(
                article_id=_article_id(url),
                title=entry.get("title", "").strip(),
                source=source,
                url=url,
                published_at=entry.get("published", ""),
                summary=summary,
                content=content,
            ))
    return out


def _fetch_newsapi() -> List[Article]:
    if not NEWSAPI_KEY:
        return []
    try:
        from newsapi import NewsApiClient
    except ImportError:
        log.warning("newsapi-python not installed; skipping")
        return []
    client = NewsApiClient(api_key=NEWSAPI_KEY)
    try:
        resp = client.get_top_headlines(language="en", page_size=50)
    except Exception as e:
        log.warning("NewsAPI fetch failed: %s", e)
        return []
    out: List[Article] = []
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
    return out


def fetch_articles(force_refresh: bool = False) -> List[Article]:
    """Return fresh articles, using the shared cache when possible.

    Cache is keyed globally (not per-user) so the NewsAPI quota is a
    one-time-per-cycle cost no matter how many users we serve.
    """
    if not force_refresh:
        cached = _cache_recent()
        if cached:
            return cached
    fetched = _fetch_newsapi() + _fetch_rss(DEFAULT_RSS_FEEDS)
    # dedupe by article_id, preferring NewsAPI (appears first)
    seen: set[str] = set()
    unique: List[Article] = []
    for a in fetched:
        if a.article_id in seen or not a.title:
            continue
        seen.add(a.article_id)
        unique.append(a)
    _cache_put(unique)
    return unique


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
            max_tokens=2000,
            messages=[{"role": "user", "content": prompt}],
        )
        raw = msg.content[0].text
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
    return scored[:top_k]


def _extract_json(text: str) -> str:
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1:
        raise ValueError("no JSON object in response")
    return text[start : end + 1]


def fetch_and_rank_articles(user: User, top_k: int = 5) -> List[Article]:
    """Main interface consumed by the agent graph."""
    articles = fetch_articles()
    already = already_sent_ids(user.user_id)
    fresh = [a for a in articles if a.article_id not in already]
    return llm_rank(fresh, user, top_k=top_k)
