"""Classification LLM call over an OpenAI-compatible chat endpoint.

The classifier is backend-agnostic: point CLASSIFIER_BASE_URL at any
OpenAI-compatible /v1 server — LiteLLM, OpenAI, vLLM, or Ollama's own
http://127.0.0.1:11434/v1 — and select the model with CLASSIFIER_MODEL. No
model is pinned locally; the /classify contract is unchanged.

Structured outputs: some gateways reject `response_format`, so we do NOT send
it. Instead the prompt asks for JSON only and `_extract_json` parses
defensively. Use an instruct/coder model — "thinking" models leak
<think>…</think> into the content (stripped here as a safety net, but they cost
extra latency and tokens).
"""

from __future__ import annotations

import json
import logging
import os
import re

import httpx

logger = logging.getLogger("classifier_svc")

# OpenAI-compatible base URL, e.g. http://127.0.0.1:4000/v1
BASE_URL = os.environ.get("CLASSIFIER_BASE_URL", "http://127.0.0.1:4000/v1").rstrip("/")
# Model id as the endpoint knows it. Required — set per deployment.
CLASSIFIER_MODEL = os.environ.get("CLASSIFIER_MODEL", "")
# Bearer token for the endpoint. Empty = no auth header (e.g. local Ollama).
# LITELLM_MASTER_KEY is honored as a fallback since lite-router targets LiteLLM.
API_KEY = os.environ.get("CLASSIFIER_API_KEY") or os.environ.get("LITELLM_MASTER_KEY", "")
REQUEST_TIMEOUT_S = float(os.environ.get("CLASSIFIER_TIMEOUT_S", "15"))

_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)
_FENCE_RE = re.compile(r"```(?:json)?\s*|\s*```", re.IGNORECASE)

_client: httpx.AsyncClient | None = None


def _get_client() -> httpx.AsyncClient:
    global _client
    if _client is None:
        _client = httpx.AsyncClient(timeout=REQUEST_TIMEOUT_S)
    return _client


def _extract_json(content: str) -> dict | None:
    """Best-effort parse of a JSON object from model output.

    Tolerates <think> blocks, ```json fences, and prose around the object by
    stripping known wrappers and grabbing the outermost {...} span.
    """
    if not content:
        return None
    text = _THINK_RE.sub("", content)
    text = _FENCE_RE.sub("", text).strip()
    try:
        obj = json.loads(text)
        return obj if isinstance(obj, dict) else None
    except json.JSONDecodeError:
        pass
    start, end = text.find("{"), text.rfind("}")
    if start != -1 and end > start:
        try:
            obj = json.loads(text[start : end + 1])
            return obj if isinstance(obj, dict) else None
        except json.JSONDecodeError:
            return None
    return None


async def chat_classify(
    system: str, user: str, schema: dict, timeout_s: float | None = None
) -> dict | None:
    """Return parsed {policy_name, confidence} or None on any failure.

    `schema` is accepted for signature compatibility but not sent on the wire —
    the enum is enforced by the prompt and validated by the caller, so the call
    works even against gateways that reject response_format. `timeout_s`
    overrides the request timeout (the warmup call uses a longer one).
    """
    payload = {
        "model": CLASSIFIER_MODEL,
        "stream": False,
        "temperature": 0,
        "max_tokens": 64,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
    }
    headers = {"Content-Type": "application/json"}
    if API_KEY:
        headers["Authorization"] = f"Bearer {API_KEY}"
    try:
        resp = await _get_client().post(
            f"{BASE_URL}/chat/completions", json=payload, headers=headers,
            timeout=timeout_s if timeout_s is not None else REQUEST_TIMEOUT_S,
        )
        resp.raise_for_status()
        content = resp.json()["choices"][0]["message"].get("content", "")
        return _extract_json(content)
    except Exception:
        logger.warning("classify call failed", exc_info=True)
        return None


async def aclose() -> None:
    global _client
    if _client is not None:
        await _client.aclose()
        _client = None
