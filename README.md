# lite-router

A session-pinned model router for [LiteLLM](https://github.com/BerriAI/litellm).
It's a pre-call hook that classifies each new coding session **once**, pins a
model tier for the session's lifetime, escalates one tier on explicit failure
signals, and logs every decision for a future learned router.

It's **client-agnostic** — it works with Claude Code (Anthropic `/v1/messages`)
and OpenCode / any OpenAI-compatible client (`/chat/completions`) — and
**provider-agnostic**: tiers map to bare model ids that LiteLLM binds to any
provider, so a tier can be an open-weight model (Ollama/vLLM/llama.cpp) or any
cloud (Anthropic, xAI, OpenAI, Bedrock, …). The bundled example ladder mixes
providers on purpose.

You expose a single virtual model — `auto` — to your clients. The router
decides, per session, which real model actually serves the traffic.

Why session pinning: agent turns mostly replay context, and prompt-cache reads
are ~10x cheaper than fresh input tokens — per-request routing destroys the
cache. Classify once, pin, never silently change model mid-session.

## Layout

```
packages/router-common/    shared policy models, hashing, event types (pydantic+pyyaml only)
packages/session-router/   the LiteLLM pre-call hook (Redis pinning, escalation, logging)
packages/classifier-svc/   FastAPI wrapper around an Ollama model (default: qwen3-coder:30b)
policies.yaml              tiers, path overrides, escalation rules — edit to fit your models
migrations/ + scripts/     Postgres schema, migrate/replay/load-test/purge tools
deploy/                    LiteLLM Dockerfile+config, GPU-box systemd units, RUNBOOK.md
tests/                     unit (pure core), integration (docker-compose), mock classifier
```

## Install on an existing LiteLLM proxy

The router ships as two Python packages plus a callbacks module. To add it to a
LiteLLM proxy you already run:

**1. Install the packages into the LiteLLM environment.**

```bash
pip install ./packages/router-common ./packages/session-router
```

If you run LiteLLM in Docker, build the bundled image instead — it installs
both packages into the proxy's venv:

```bash
docker build -f deploy/litellm/Dockerfile -t litellm-router .
```

**2. Register the callback.** Drop a `custom_callbacks.py` next to your config
(one is provided in `deploy/litellm/`):

```python
from session_router.hook import LiteAutoRouter

proxy_handler_instance = LiteAutoRouter()
```

**3. Wire it into `litellm_config.yaml`** — a virtual model that the hook
routes, the real models it can route to, and the callback:

```yaml
model_list:
  # Virtual model clients ask for. When ROUTER_ENABLED=false the hook is a
  # no-op and this becomes a plain alias for its litellm_params model.
  - model_name: auto
    litellm_params:
      model: xai/grok-4.5            # the default tier's model
      api_key: os.environ/XAI_API_KEY
  # One deployment per tier model in policies.yaml. Mix providers freely —
  # this is the only place a provider is named.
  - model_name: claude-sonnet-5
    litellm_params: { model: anthropic/claude-sonnet-5, api_key: os.environ/ANTHROPIC_API_KEY }
  - model_name: grok-4.5
    litellm_params: { model: xai/grok-4.5, api_key: os.environ/XAI_API_KEY }
  - model_name: claude-opus-4-8
    litellm_params: { model: anthropic/claude-opus-4-8, api_key: os.environ/ANTHROPIC_API_KEY }
  - model_name: claude-fable-5
    litellm_params: { model: anthropic/claude-fable-5, api_key: os.environ/ANTHROPIC_API_KEY }

litellm_settings:
  callbacks: custom_callbacks.proxy_handler_instance
  drop_params: true   # drop params a backend rejects (e.g. OpenCode's reasoningSummary)

router_settings:
  # Fallbacks must never go DOWN a tier — that would violate the
  # no-downgrade invariant. Only ever fall back upward.
  fallbacks:
    - claude-sonnet-5: ["grok-4.5"]
    - grok-4.5: ["claude-opus-4-8"]
    - claude-opus-4-8: ["claude-fable-5"]
```

The virtual model name (`auto` above, `lite-auto` in the reference config in
`deploy/litellm/`) is arbitrary — pick whatever your clients should request.
The real model names must match the `tiers` in `policies.yaml`.

**4. Point the hook at its dependencies** via environment variables:

| Var | Example | Notes |
|---|---|---|
| `ROUTER_ENABLED` | `true` | `false` = full bypass: the virtual model becomes a plain alias, no routing/logging |
| `SHADOW_MODE` | `true` | classify/pin/log everything, but always route the default model (safe rollout) |
| `ROUTER_REDIS_URL` | `redis://localhost:6379/0` | session pin store; engine ≥ 6.2 preferred (GETEX) |
| `ROUTER_CLASSIFIER_URL` | `http://gpu-box:8891` | the classifier service (see below) |
| `ROUTER_DATABASE_URL` | `postgresql://...` | decision logging; empty string disables it |
| `ROUTER_POLICIES_PATH` | `/app/policies.yaml` | hot-reloaded on mtime change |

See `deploy/RUNBOOK.md` for the full variable list, rollout, and rollback.

The classifier is optional-but-recommended: it's a small FastAPI service
(`packages/classifier-svc`) that wraps a local Ollama model. Without it the
router still pins sessions — it just falls back to the default tier for the
first request. Run it with `uv run classifier-svc` (see below).

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

Try it with a real Claude Code client (the dev stack registers the virtual
model as `lite-auto`):

```bash
ANTHROPIC_BASE_URL=http://127.0.0.1:4000 ANTHROPIC_AUTH_TOKEN=sk-test-master \
  ANTHROPIC_MODEL=lite-auto claude -p "what does this regex do: ^a+$"
# -> routed:claude-sonnet-5
```

Or with OpenCode, which connects as an OpenAI-compatible client — point a
provider at the proxy's `/v1` base URL and request `lite-auto` (see the
OpenCode section in `deploy/RUNBOOK.md`). OpenCode can't send a per-conversation
session header, so the router pins it by the content-hash fallback below.

## How routing works

1. **Session key**: a session-id header (`x-claude-code-session-id`, then
   `x-session-id` — configurable via `ROUTER_SESSION_HEADERS`); fallback
   `sha256(key + system + first message)[:16]` for clients that send none
   (e.g. OpenCode). The fallback is stable because agents replay the same
   system prompt + first user message every turn. Records live in Redis,
   sliding 8h TTL.
2. **Request shape**: the system prompt and tool results are read from either
   the Anthropic shape (top-level `system`, `tool_result` blocks) or the OpenAI
   shape (`messages[0]` role=system, `role:tool` messages) — both fully
   supported. Each decision is labelled with the detected `client`
   (claude-code / opencode / generic).
3. **First request**: repos flagged in `path_overrides` (see `policies.yaml`)
   force-pin the top tier with no classifier call — via the Claude Code
   system-prompt cwd, or the universal `x-lite-tier` header for any client;
   otherwise `classifier-svc` picks a tier (once, ~500ms, 1s timeout).
4. **Pinned requests**: same model for the whole session; the hook adds
   ~0.2ms. No downgrades, ever.
5. **Escalation**: one tier up (sticky, max 3 across the 4-tier example) on
   user retry phrases, two consecutive failing tool results (either shape), or
   the `x-router-escalate: true` header.
6. **Overrides**: a concrete model name bypasses the router entirely (logged).
7. **Fail-open**: Redis down, classifier down, any hook bug → default model.
   The gateway never fails because of the router.

Shadow mode (`SHADOW_MODE=true`, the recommended launch default) runs
everything — classification, pinning, escalation, logging — but always routes
the default model. See `deploy/RUNBOOK.md` for rollout, rollback, and client
setup.

## Configuring policies

Everything routing-specific lives in `policies.yaml`: the tier→model mapping,
which repo-path patterns force the top tier, the escalation retry phrases and
failure markers, session TTL, and per-model pricing (used only for spend
projection in `scripts/replay_shadow.py`). Edit it to match your own model
lineup and sensitivity rules — no code changes needed, and it hot-reloads.

## Measured (RTX 3090, qwen3-coder:30b classifier)

- Classifier tier agreement: 51/51 on a hand-labeled prompt set; p50 507ms /
  p95 568ms (`OLLAMA_FORMAT_MODE=json`; schema-constrained mode ~2x slower).
- In-hook pinned-path cost: p95 0.2ms against real Redis (budget < 5ms).
- Full integration: 12 scenarios incl. escalation ratchet+cap, path override,
  classifier timeout, Redis outage, 20-way classification race, shadow mode,
  subagent tier-down (R1b, flag-off by default).
```