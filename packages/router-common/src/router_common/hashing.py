"""Hash helpers shared by the hook and analysis scripts."""

from __future__ import annotations

import hashlib


def sha256_hex(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8", errors="replace")).hexdigest()


def short_hash(text: str, length: int = 16) -> str:
    return sha256_hex(text)[:length]


def derive_fallback_session_key(
    api_key_alias: str, system_text: str, first_user_message: str
) -> str:
    """Session key for clients that don't send X-Claude-Code-Session-Id.

    sha256(api_key + system_prompt_hash + first_user_message_hash)[:16].
    Stable within a conversation because the replayed history always starts
    with the same first user message and system prompt.
    """
    material = api_key_alias + short_hash(system_text) + short_hash(first_user_message)
    return short_hash(material, 16)
