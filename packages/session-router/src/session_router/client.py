"""Client detection. Pure — infers which agent/CLI issued the request.

Used to label decision events and to gate Claude-Code-specific behavior
(subagent routing, system-prompt repo-hint parsing). Detection must never
raise; an unknown client degrades to CLIENT_GENERIC.
"""

from __future__ import annotations

CLIENT_CLAUDE_CODE = "claude-code"
CLIENT_OPENCODE = "opencode"
CLIENT_GENERIC = "generic"


def detect_client(headers: dict[str, str], call_type: str | None = None) -> str:
    """Return one of claude-code / opencode / generic.

    Claude Code is identified by its x-claude-code-* headers (authoritative);
    OpenCode by its user-agent. Anything else is generic. `call_type` is
    accepted for future use but is not decisive (opencode and generic both
    arrive as OpenAI completions).
    """
    if any(isinstance(k, str) and k.startswith("x-claude-code-") for k in headers):
        return CLIENT_CLAUDE_CODE
    if "opencode" in headers.get("user-agent", "").lower():
        return CLIENT_OPENCODE
    return CLIENT_GENERIC
