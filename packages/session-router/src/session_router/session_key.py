"""Session key derivation. Pure functions over the LiteLLM `data` dict."""

from __future__ import annotations

from typing import Any

from router_common.hashing import derive_fallback_session_key, short_hash

SESSION_HEADER = "x-claude-code-session-id"
AGENT_HEADER = "x-claude-code-agent-id"
PARENT_AGENT_HEADER = "x-claude-code-parent-agent-id"
ESCALATE_HEADER = "x-router-escalate"


def extract_headers(data: dict) -> dict[str, str]:
    """Merge headers from every location LiteLLM may put them, lowercased.

    /v1/messages requests carry them in litellm_metadata, /chat/completions in
    metadata, and both in proxy_server_request.
    """
    merged: dict[str, str] = {}
    sources = (
        (data.get("proxy_server_request") or {}).get("headers"),
        (data.get("metadata") or {}).get("headers"),
        (data.get("litellm_metadata") or {}).get("headers"),
    )
    for headers in sources:
        if isinstance(headers, dict):
            for k, v in headers.items():
                if isinstance(k, str) and isinstance(v, str):
                    merged[k.lower()] = v
    return merged


def flatten_content(content: Any, limit: int = 4000) -> str:
    """Flatten Anthropic message content (str or block list) to plain text."""
    if isinstance(content, str):
        return content[:limit]
    parts: list[str] = []
    total = 0
    if isinstance(content, list):
        for block in content:
            if total >= limit:
                break
            text: str | None = None
            if isinstance(block, str):
                text = block
            elif isinstance(block, dict):
                if isinstance(block.get("text"), str):
                    text = block["text"]
                elif isinstance(block.get("content"), (str, list)):
                    text = flatten_content(block["content"], limit - total)
            if text:
                parts.append(text[: limit - total])
                total += len(parts[-1])
    return "".join(parts)


def first_user_message_text(data: dict, limit: int = 4000) -> str:
    for msg in data.get("messages") or []:
        if isinstance(msg, dict) and msg.get("role") == "user":
            return flatten_content(msg.get("content"), limit)
    return ""


def system_text(data: dict, limit: int = 8000) -> str:
    return flatten_content(data.get("system") or "", limit)


def derive_session_key(data: dict, headers: dict[str, str], api_key_alias: str) -> tuple[str, str]:
    """Return (session_key, source) where source is "header" or "derived"."""
    header_key = headers.get(SESSION_HEADER, "").strip()
    if header_key:
        return header_key, "header"
    key = derive_fallback_session_key(
        api_key_alias, system_text(data), first_user_message_text(data)
    )
    return key, "derived"


def first_message_hash(data: dict) -> str:
    return short_hash(first_user_message_text(data))


def extract_agent_ids(headers: dict[str, str]) -> tuple[str | None, str | None]:
    """Return (agent_id, parent_agent_id); parent set means subagent traffic."""
    return headers.get(AGENT_HEADER) or None, headers.get(PARENT_AGENT_HEADER) or None
