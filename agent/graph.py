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

from config import MAX_ARTICLES_PER_PUSH
from core.article_fetcher import Article, fetch_and_rank_articles
from core.rag_engine import (
    handle_followup,
    index_articles,
    rehydrate_articles,
    topic_diverse_articles,
)
from core.user_profile import (
    User,
    append_message,
    already_sent_ids,
    generate_persona_summary,
    get_user_profile,
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
    force_refresh: bool
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


def _rehydrate_sent_articles(user_id: str, limit: int = 15) -> List[Article]:
    """For follow-ups, the user may be asking about anything we've sent them
    recently — not just today's push. Rehydrate from cache."""
    sent = list(already_sent_ids(user_id))[-limit:]
    return rehydrate_articles(sent)


def followup_node(state: AgentState) -> AgentState:
    user: User = state["user"]
    query = state.get("inbound_text", "").strip()
    append_message(user.user_id, "user", query)

    articles = _rehydrate_sent_articles(user.user_id)
    reply = handle_followup(user, query, articles=articles or None)
    append_message(user.user_id, "assistant", reply)
    return {"reply": reply}


def proactive_fetch_node(state: AgentState) -> AgentState:
    user: User = state["user"]
    if user.onboarding_state != "DONE":
        # skip — user hasn't finished onboarding yet
        return {"articles": [], "reply": ""}
    force_refresh = state.get("force_refresh", False)
    # Pull a wider candidate pool from the LLM ranker, then for each of the
    # user's interests pick the article that best matches that interest, then
    # take the top MAX_ARTICLES_PER_PUSH of those by relevance score. This
    # guarantees the digest spans different topics when the user has named
    # more than one.
    candidate_pool = max(10, MAX_ARTICLES_PER_PUSH * 3)
    candidates = fetch_and_rank_articles(user, top_k=candidate_pool, force_refresh=force_refresh)
    articles = topic_diverse_articles(candidates, user.interests, k=MAX_ARTICLES_PER_PUSH)
    return {"articles": articles}


def proactive_format_node(state: AgentState) -> AgentState:
    user: User = state["user"]
    articles: List[Article] = state.get("articles", [])
    if not articles:
        return {"reply": ""}

    import re
    def _clean(text: str) -> str:
        text = re.sub(r"[\[\(]?[…\.]{2,}[\]\)]?\s*$", "", text).strip()
        return text

    lines = [f"Your {len(articles)}-story digest:", ""]
    for i, a in enumerate(articles, 1):
        snippet = _clean(a.rationale or (a.summary or "")[:400])
        lines.append(f"{i}. {a.title} ({a.source})")
        lines.append(f"   {snippet}")
        lines.append(f"   {a.url}")
        lines.append("")
    lines.append("Reply with a question about any of these.")
    reply = "\n".join(lines)

    # index for RAG and mark sent
    index_articles(user.user_id, articles)
    mark_articles_sent(user.user_id, [a.article_id for a in articles])
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


def run_proactive_push(user_id: str, force_refresh: bool = False) -> tuple[str, List[Article]]:
    result = _proactive_graph().invoke({
        "user_id": user_id,
        "trigger": "proactive",
        "force_refresh": force_refresh,
    })
    return result.get("reply", ""), result.get("articles", [])
