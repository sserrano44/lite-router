"""Ollama chat call with structured outputs. Backend-agnostic /classify contract."""

from __future__ import annotations

import json
import logging
import os

import httpx

logger = logging.getLogger("classifier_svc")

OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://127.0.0.1:11434").rstrip("/")
CLASSIFIER_MODEL = os.environ.get("CLASSIFIER_MODEL", "qwen3-coder:30b")
OLLAMA_TIMEOUT_S = float(os.environ.get("OLLAMA_TIMEOUT_S", "10"))

_client: httpx.AsyncClient | None = None


def _get_client() -> httpx.AsyncClient:
    global _client
    if _client is None:
        _client = httpx.AsyncClient(timeout=OLLAMA_TIMEOUT_S)
    return _client


async def chat_classify(system: str, user: str, schema: dict) -> dict | None:
    """Return parsed {policy_name, confidence} or None on any failure."""
    payload = {
        "model": CLASSIFIER_MODEL,
        "stream": False,
        "keep_alive": -1,
        "format": schema,
        "options": {"temperature": 0, "num_predict": 64},
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
    }
    try:
        resp = await _get_client().post(f"{OLLAMA_URL}/api/chat", json=payload)
        resp.raise_for_status()
        content = resp.json().get("message", {}).get("content", "")
        return json.loads(content)
    except Exception:
        logger.warning("ollama classify failed", exc_info=True)
        return None


async def aclose() -> None:
    global _client
    if _client is not None:
        await _client.aclose()
        _client = None
