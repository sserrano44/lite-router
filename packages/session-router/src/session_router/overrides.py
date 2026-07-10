"""Repo-hint extraction and mission-critical path-override matching. Pure."""

from __future__ import annotations

import re
from dataclasses import dataclass

from router_common.policies import PoliciesConfig

TIER_HEADER = "x-lite-tier"

# Claude Code embeds an env block in its system prompt, e.g.
#   Working directory: /home/user/capyfi
# (historically also "Primary working directory:" / cwd lines).
# This auto-detection is Claude-Code-specific — the hook only feeds it a
# populated system prompt for that client. Every other client relies on the
# universal x-lite-tier header (see match_path_override) for path overrides.
_CWD_RE = re.compile(
    r"(?:working directory|cwd)\s*:\s*(?P<path>[^\n<]+)", re.IGNORECASE
)
_CLAUDE_MD_RE = re.compile(r"CLAUDE\.md", re.IGNORECASE)


@dataclass(slots=True)
class RepoHints:
    cwd: str = ""
    claude_md_excerpt: str = ""


def extract_repo_hints(system_text: str, excerpt_len: int = 1000) -> RepoHints:
    cwd = ""
    m = _CWD_RE.search(system_text)
    if m:
        cwd = m.group("path").strip()
    excerpt = ""
    m = _CLAUDE_MD_RE.search(system_text)
    if m:
        excerpt = system_text[m.start() : m.start() + excerpt_len]
    return RepoHints(cwd=cwd, claude_md_excerpt=excerpt)


def match_path_override(
    hints: RepoHints, headers: dict[str, str], policies: PoliciesConfig
) -> str | None:
    """Return the matched pattern (or header marker), else None.

    Case-insensitive substring match of policy patterns against the working
    directory and CLAUDE.md excerpt; the x-lite-tier header is an explicit
    per-repo fallback marker (R12b) that force-pins when it names the
    override tier.
    """
    if headers.get(TIER_HEADER, "").strip().lower() == policies.path_overrides.force_tier:
        return f"header:{TIER_HEADER}"
    haystack = f"{hints.cwd}\n{hints.claude_md_excerpt}".lower()
    if not haystack.strip():
        return None
    for pattern in policies.path_overrides.patterns:
        if pattern.lower() in haystack:
            return pattern
    return None
