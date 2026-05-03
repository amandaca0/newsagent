"""Unified LLM client.

Provider preference (checked in order):
  1. Anthropic — when ANTHROPIC_API_KEY is set and not the placeholder
  2. Groq      — when GROQ_API_KEY is set and not the placeholder
  3. None      — callers fall back to TF-IDF / keyword joins

A key is considered "real" when it is non-empty and does not end in "..."
(matches the placeholder convention in .env.example).

Public API:
  - llm_configured() -> bool
  - active_provider() -> "anthropic" | "groq" | None
  - active_model(purpose) -> str         (used by callers for logging)
  - complete(prompt, *, max_tokens, purpose, max_retries) -> str
"""
from __future__ import annotations

import logging
import time
from typing import Literal, Optional

from anthropic import Anthropic
from anthropic import RateLimitError as AnthropicRateLimitError
from groq import Groq
from groq import RateLimitError as GroqRateLimitError

from config import (
    ANTHROPIC_API_KEY,
    ANTHROPIC_JUDGE_MODEL,
    ANTHROPIC_MODEL,
    GROQ_API_KEY,
    JUDGE_MODEL,
    LLM_MODEL,
)

log = logging.getLogger(__name__)

Purpose = Literal["chat", "judge"]
Provider = Literal["anthropic", "groq"]


def _key_real(key: str) -> bool:
    if not key:
        return False
    if key.endswith("..."):
        return False
    return True


def llm_configured() -> bool:
    return _key_real(ANTHROPIC_API_KEY) or _key_real(GROQ_API_KEY)


def active_provider() -> Optional[Provider]:
    if _key_real(ANTHROPIC_API_KEY):
        return "anthropic"
    if _key_real(GROQ_API_KEY):
        return "groq"
    return None


_ANTHROPIC_MODELS = {"chat": ANTHROPIC_MODEL, "judge": ANTHROPIC_JUDGE_MODEL}
_GROQ_MODELS = {"chat": LLM_MODEL, "judge": JUDGE_MODEL}


def active_model(purpose: Purpose = "chat") -> str:
    provider = active_provider()
    if provider == "anthropic":
        return _ANTHROPIC_MODELS[purpose]
    if provider == "groq":
        return _GROQ_MODELS[purpose]
    return ""


def complete(
    prompt: str,
    *,
    max_tokens: int = 1500,
    purpose: Purpose = "chat",
    max_retries: int = 3,
) -> str:
    """Send one user prompt, return the assistant's text reply.

    Picks the provider based on which API key is set, retries on rate limits
    with exponential-ish backoff (matching the prior rag_engine behaviour:
    10s, 20s on attempts 0, 1). Other exceptions propagate immediately so
    callers can fall back to TF-IDF rather than hang on a permanent error.
    """
    provider = active_provider()
    if provider is None:
        raise RuntimeError("no LLM provider configured")

    model = active_model(purpose)
    last_exc: Optional[Exception] = None

    for attempt in range(max_retries):
        try:
            if provider == "anthropic":
                return _anthropic_call(prompt, model=model, max_tokens=max_tokens)
            return _groq_call(prompt, model=model, max_tokens=max_tokens)
        except (AnthropicRateLimitError, GroqRateLimitError) as e:
            last_exc = e
            if attempt == max_retries - 1:
                break
            wait = 10 * (attempt + 1)
            log.warning("%s rate limit hit, retrying in %ds (attempt %d)",
                        provider, wait, attempt + 1)
            time.sleep(wait)

    assert last_exc is not None
    raise last_exc


def _anthropic_call(prompt: str, *, model: str, max_tokens: int) -> str:
    client = Anthropic(api_key=ANTHROPIC_API_KEY)
    msg = client.messages.create(
        model=model,
        max_tokens=max_tokens,
        messages=[{"role": "user", "content": prompt}],
    )
    # response.content is a list of content blocks; for plain text replies
    # we get a single TextBlock at index 0.
    parts = []
    for block in msg.content:
        text = getattr(block, "text", None)
        if text:
            parts.append(text)
    return "".join(parts)


def _groq_call(prompt: str, *, model: str, max_tokens: int) -> str:
    client = Groq(api_key=GROQ_API_KEY)
    msg = client.chat.completions.create(
        model=model,
        max_tokens=max_tokens,
        messages=[{"role": "user", "content": prompt}],
    )
    return msg.choices[0].message.content or ""
