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
# "json": free-form JSON, enum enforced by prompt + code validation (~2x
# faster: schema-constrained decoding recompiles a grammar per request).
# "schema": hard-constrained decoding via structured outputs.
OLLAMA_FORMAT_MODE = os.environ.get("OLLAMA_FORMAT_MODE", "json")

_client: httpx.AsyncClient | None = None


def _get_client() -> httpx.AsyncClient:
    global _client
    if _client is None:
        _client = httpx.AsyncClient(timeout=OLLAMA_TIMEOUT_S)
    return _client


async def chat_classify(
    system: str, user: str, schema: dict, timeout_s: float | None = None
) -> dict | None:
    """Return parsed {policy_name, confidence} or None on any failure.

    timeout_s overrides the default request timeout — the warmup call uses a
    long one because a cold model load into VRAM can take minutes.
    """
    payload = {
        "model": CLASSIFIER_MODEL,
        "stream": False,
        "keep_alive": -1,
        "format": schema if OLLAMA_FORMAT_MODE == "schema" else "json",
        "options": {"temperature": 0, "num_predict": 64},
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
    }
    try:
        resp = await _get_client().post(
            f"{OLLAMA_URL}/api/chat", json=payload,
            timeout=timeout_s if timeout_s is not None else OLLAMA_TIMEOUT_S,
        )
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
