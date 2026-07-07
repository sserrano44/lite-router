# lite-router

Ripio session-pinned model router (`ripio-auto`): a LiteLLM proxy hook that
classifies each new Claude Code session once, pins a model
(Haiku → Sonnet → Opus) for the session's lifetime, escalates one tier on
explicit failure signals, and logs every decision for a future learned router.

Why session pinning: agent turns mostly replay context, and prompt-cache reads
are ~10x cheaper than fresh input tokens — per-request routing destroys the
cache. Classify once, pin, never silently change model mid-session.

## Layout

```
packages/router-common/    shared policy models, hashing, event types (pydantic+pyyaml only)
packages/session-router/   the LiteLLM pre-call hook (Redis pinning, escalation, logging)
packages/classifier-svc/   FastAPI wrapper around Ollama qwen3-coder:30b
policies.yaml              tiers, path overrides, escalation rules — PR to change
migrations/ + scripts/     Postgres schema, migrate/replay/load-test/purge tools
deploy/                    LiteLLM Dockerfile+config, GPU-box systemd units, RUNBOOK.md
tests/                     unit (pure core), integration (docker-compose), mock classifier
```

## Development

```bash
uv sync                                   # workspace env
uv run pytest tests/unit                  # 87 tests, no services needed

docker compose up -d --build              # LiteLLM + Redis + Postgres + mock classifier
ROUTER_DATABASE_URL=postgresql://postgres:postgres@127.0.0.1:5433/postgres \
  uv run scripts/migrate.py
uv run pytest tests/integration           # 12 tests against the stack

uv run classifier-svc                     # real classifier against local Ollama
uv run scripts/load_test.py               # pinned-path overhead check
```

Try it with a real Claude Code client:

```bash
ANTHROPIC_BASE_URL=http://127.0.0.1:4000 ANTHROPIC_AUTH_TOKEN=sk-test-master \
  ANTHROPIC_MODEL=ripio-auto claude -p "what does this regex do: ^a+$"
# -> routed:claude-haiku-4-5
```

## How routing works

1. **Session key**: `X-Claude-Code-Session-Id` header (verified to survive the
   proxy path); fallback `sha256(key + system + first message)[:16]` for other
   clients. Records live in Redis, sliding 8h TTL.
2. **First request**: mission-critical repos (`capyfi`, `contracts`, `bridge`,
   `custody`, ... — see `policies.yaml`) force-pin Opus with no classifier
   call; otherwise `classifier-svc` picks a tier (once, ~500ms, 1s timeout).
3. **Pinned requests**: same model for the whole session; the hook adds
   ~0.2ms. No downgrades, ever.
4. **Escalation**: one tier up (sticky, max 2) on user retry phrases, two
   consecutive failing tool results, or the `x-router-escalate: true` header.
5. **Overrides**: a concrete model name bypasses the router entirely (logged).
6. **Fail-open**: Redis down, classifier down, any hook bug → default model
   (Sonnet). The gateway never fails because of the router.

Shadow mode (`SHADOW_MODE=true`, the launch default) runs everything — 
classification, pinning, escalation, logging — but always routes the default
model. See `deploy/RUNBOOK.md` for rollout, rollback, and client setup.

## Measured (this box, RTX 3090)

- Classifier tier agreement: 51/51 on a hand-labeled prompt set; p50 507ms /
  p95 568ms (`OLLAMA_FORMAT_MODE=json`; schema-constrained mode ~2x slower).
- In-hook pinned-path cost: p95 0.2ms against real Redis (budget < 5ms).
- Full integration: 12 scenarios incl. escalation ratchet+cap, path override,
  classifier timeout, Redis outage, 20-way classification race, shadow mode,
  subagent tier-down (R1b, flag-off by default).
