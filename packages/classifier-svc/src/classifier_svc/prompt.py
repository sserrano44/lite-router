"""Classification prompt construction from policies.yaml."""

from __future__ import annotations

from router_common.policies import PoliciesConfig

# Few-shot examples keep the small model honest on the tier boundaries.
_EXAMPLES = [
    ("How do I check if a key exists in a Python dict?", "quick_lookup"),
    ("Write a regex that matches ISO-8601 dates.", "quick_lookup"),
    ("Add a POST /users endpoint with validation and a unit test.", "standard_dev"),
    ("Bump fastapi to 0.115 and fix any deprecation warnings.", "standard_dev"),
    ("Refactor the payments module to support multi-currency settlement; "
     "it touches the ledger, the reconciliation job, and the public API.", "high"),
    ("Our request queue deadlocks under load, sometimes. Find out why.", "high"),
    ("Design a lock-free concurrent hashmap and argue it's linearizable.", "ultra-think"),
]

MAX_PROMPT_CHARS = 6000


def system_prompt(policies: PoliciesConfig) -> str:
    tier_lines = "\n".join(
        f"- {t.name}: {' '.join(t.description.split())}" for t in policies.tiers
    )
    example_lines = "\n".join(f'  "{q}" -> {name}' for q, name in _EXAMPLES)
    return (
        "You classify the FIRST message of a software-engineering session into "
        "exactly one policy tier, based on how much model capability the task needs.\n\n"
        f"Tiers:\n{tier_lines}\n\n"
        f"Examples:\n{example_lines}\n\n"
        "Respond with JSON only: {\"policy_name\": <tier>, \"confidence\": <0..1>}. "
        "When torn between two tiers, pick the higher one."
    )


def user_prompt(first_message: str, system_summary: str, repo_hints: dict) -> str:
    parts = []
    cwd = (repo_hints or {}).get("cwd") or ""
    if cwd:
        parts.append(f"Working directory: {cwd}")
    excerpt = (repo_hints or {}).get("claude_md_excerpt") or ""
    if excerpt:
        parts.append(f"Project context excerpt:\n{excerpt[:500]}")
    if system_summary:
        parts.append(f"System prompt summary:\n{system_summary[:1000]}")
    parts.append(f"First user message:\n{first_message[:4000]}")
    return "\n\n".join(parts)[:MAX_PROMPT_CHARS]


def response_schema(policies: PoliciesConfig) -> dict:
    """JSON schema for Ollama structured outputs — constrains the enum."""
    return {
        "type": "object",
        "properties": {
            "policy_name": {"type": "string", "enum": policies.tier_names()},
            "confidence": {"type": "number", "minimum": 0, "maximum": 1},
        },
        "required": ["policy_name", "confidence"],
    }
