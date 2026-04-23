"""LangGraph agent state machine.

Entry points:
  - run_inbound(user_id, text) -> reply string
      Handles SMS from a user. Routes onboarding vs. follow-up.
  - run_proactive_push(user_id) -> (reply_text, articles)
      Produces the morning digest. Called by the scheduler.

States: ONBOARDING -> IDLE -> PROACTIVE_PUSH -> AWAITING_REPLY -> RAG_RESPONSE -> IDLE
Only the transitions we actually use are wired as edges; others are implicit.
"""
from __future__ import annotations

import logging
from typing import List, Literal, Optional, TypedDict

from langgraph.graph import END, StateGraph

from config import MAX_ARTICLES_PER_PUSH, MAX_ARTICLES_TO_INDEX
from core.article_fetcher import Article, fetch_and_rank_articles
from core.rag_engine import handle_followup, index_articles, rehydrate_articles
from core.user_profile import (
    User,
    append_message,
    already_indexed_ids,
    generate_persona_summary,
    get_user_profile,
    mark_articles_indexed,
    mark_articles_sent,
    set_interests,
    set_onboarding_state,
    set_persona_summary,
)

log = logging.getLogger(__name__)


class AgentState(TypedDict, total=False):
    user_id: str
    trigger: Literal["inbound", "proactive"]
    inbound_text: str
    user: User
    articles: List[Article]
    reply: str
    route: str


# ---------- nodes ----------

def load_user(state: AgentState) -> AgentState:
    user = get_user_profile(state["user_id"])
    if user is None:
        raise ValueError(f"unknown user_id: {state['user_id']}")
    return {"user": user}


def route_inbound(state: AgentState) -> AgentState:
    user = state["user"]
    if user.onboarding_state != "DONE":
        return {"route": "onboarding"}
    return {"route": "followup"}


def onboarding_node(state: AgentState) -> AgentState:
    """Single-turn onboarding: parse interests out of the user's message,
    generate a persona summary, mark DONE. A multi-turn flow would branch
    on onboarding_state values like ASKED_INTERESTS / ASKED_CONFIRM."""
    user: User = state["user"]
    text = state.get("inbound_text", "").strip()

    if user.onboarding_state == "NEEDS_ONBOARDING" and not text:
        reply = (
            "Welcome to NewsAgent! I'll send you a short personalized news "
            "digest each morning. To start, text me 3-5 topics you want to "
            "follow (e.g. 'EU AI regulation, NBA trades, climate policy')."
        )
        set_onboarding_state(user.user_id, "AWAITING_INTERESTS")
        append_message(user.user_id, "assistant", reply)
        return {"reply": reply}

    interests = [t.strip() for t in text.replace(";", ",").split(",") if t.strip()]
    if not interests:
        reply = "Please reply with a comma-separated list of topics you'd like to follow."
        append_message(user.user_id, "assistant", reply)
        return {"reply": reply}

    set_interests(user.user_id, interests)
    summary = generate_persona_summary(interests)
    set_persona_summary(user.user_id, summary)
    set_onboarding_state(user.user_id, "DONE")

    reply = (
        f"Got it — I'll track: {', '.join(interests)}. "
        "You'll get your first digest in the morning. Reply anytime to ask "
        "follow-ups about the stories I send."
    )
    append_message(user.user_id, "assistant", reply)
    return {"reply": reply}


def _rehydrate_indexed_articles(user_id: str) -> List[Article]:
    """Rehydrate all recently indexed articles for RAG follow-ups.
    Uses the indexed set (up to 10 per push cycle) rather than only the 3 sent,
    giving the RAG engine a richer corpus to answer from."""
    indexed = list(already_indexed_ids(user_id))
    return rehydrate_articles(indexed)


def followup_node(state: AgentState) -> AgentState:
    user: User = state["user"]
    query = state.get("inbound_text", "").strip()
    append_message(user.user_id, "user", query)

    articles = _rehydrate_indexed_articles(user.user_id)
    reply = handle_followup(user, query, articles=articles or None)
    append_message(user.user_id, "assistant", reply)
    return {"reply": reply}


def proactive_fetch_node(state: AgentState) -> AgentState:
    user: User = state["user"]
    if user.onboarding_state != "DONE":
        # skip — user hasn't finished onboarding yet
        return {"articles": [], "reply": ""}
    articles = fetch_and_rank_articles(user, top_k=MAX_ARTICLES_TO_INDEX)
    return {"articles": articles}


def proactive_format_node(state: AgentState) -> AgentState:
    user: User = state["user"]
    articles: List[Article] = state.get("articles", [])
    if not articles:
        return {"reply": ""}

    # Index all fetched articles for RAG, but only surface the top N in the digest.
    index_articles(user.user_id, articles)
    mark_articles_indexed(user.user_id, [a.article_id for a in articles])

    digest_articles = articles[:MAX_ARTICLES_PER_PUSH]
    lines = [f"Your {len(digest_articles)}-story digest:"]
    for i, a in enumerate(digest_articles, 1):
        raw = (a.summary or "")[:280]
        cut = max(raw.rfind(". "), raw.rfind("! "), raw.rfind("? "))
        snippet = raw[:cut + 1] if cut != -1 else raw
        lines.append(f"{i}. {a.title} ({a.source})\n   {snippet}\n   {a.url}")
    lines.append("Reply with a question about any of these.")
    reply = "\n".join(lines)

    mark_articles_sent(user.user_id, [a.article_id for a in digest_articles])
    append_message(user.user_id, "assistant", reply)
    return {"reply": reply}


# ---------- graph assembly ----------

def _build_inbound_graph():
    g = StateGraph(AgentState)
    g.add_node("load_user", load_user)
    g.add_node("route_inbound", route_inbound)
    g.add_node("onboarding", onboarding_node)
    g.add_node("followup", followup_node)

    g.set_entry_point("load_user")
    g.add_edge("load_user", "route_inbound")
    g.add_conditional_edges(
        "route_inbound",
        lambda s: s["route"],
        {"onboarding": "onboarding", "followup": "followup"},
    )
    g.add_edge("onboarding", END)
    g.add_edge("followup", END)
    return g.compile()


def _build_proactive_graph():
    g = StateGraph(AgentState)
    g.add_node("load_user", load_user)
    g.add_node("fetch", proactive_fetch_node)
    g.add_node("format", proactive_format_node)

    g.set_entry_point("load_user")
    g.add_edge("load_user", "fetch")
    g.add_edge("fetch", "format")
    g.add_edge("format", END)
    return g.compile()


_INBOUND_GRAPH = None
_PROACTIVE_GRAPH = None


def _inbound_graph():
    global _INBOUND_GRAPH
    if _INBOUND_GRAPH is None:
        _INBOUND_GRAPH = _build_inbound_graph()
    return _INBOUND_GRAPH


def _proactive_graph():
    global _PROACTIVE_GRAPH
    if _PROACTIVE_GRAPH is None:
        _PROACTIVE_GRAPH = _build_proactive_graph()
    return _PROACTIVE_GRAPH


# ---------- public API ----------

def run_inbound(user_id: str, text: str) -> str:
    result = _inbound_graph().invoke({
        "user_id": user_id,
        "trigger": "inbound",
        "inbound_text": text,
    })
    return result.get("reply", "")


def run_proactive_push(user_id: str) -> tuple[str, List[Article]]:
    result = _proactive_graph().invoke({
        "user_id": user_id,
        "trigger": "proactive",
    })
    return result.get("reply", ""), result.get("articles", [])
