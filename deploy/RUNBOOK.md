# lite-router — Deploy & Operations Runbook

This runbook describes a reference deployment: the LiteLLM proxy on a gateway
host, the classifier on a GPU box next to Ollama. The virtual model is named
`lite-auto` here — rename it to whatever your clients request.

## Components

| Unit | Where | What |
|---|---|---|
| LiteLLM proxy + `session_router` hook | Gateway host | Routes the virtual model, pins sessions in Redis, logs to Postgres |
| `classifier-svc` | GPU box (next to Ollama) | `POST /classify` via qwen3-coder:30b, once per session |
| `policies.yaml` | this repo, mounted into both | Tiers, path overrides, escalation rules — change via PR |

## Deploying the hook (gateway host)

1. Build the image: `docker build -f deploy/litellm/Dockerfile -t litellm-router .`
   (installs `router-common` + `session-router` into the LiteLLM venv).
2. Mount/COPY next to each other in `/app`: `litellm_config.yaml` (as
   `config.yaml`), `custom_callbacks.py`, `policies.yaml`.
3. Environment:

   | Var | Shadow launch | Notes |
   |---|---|---|
   | `ROUTER_ENABLED` | `true` | `false` = full rollback: the virtual model becomes a plain alias for the default tier, zero routing, zero logging |
   | `ROUTER_SESSION_HEADERS` | `x-claude-code-session-id,x-session-id` | session-id headers, priority order; clients that send none pin via content hash |
   | `SHADOW_MODE` | `true` | classify/pin/log everything, but always route the default model. Flip to `false` for Phase 2 |
   | `SUBAGENT_ROUTING_ENABLED` | `false` | R1b tier-below-parent routing; leave off until shadow data sizes the win |
   | `ROUTER_REDIS_URL` | `redis://<elasticache>:6379/0` | engine >= 6.2 preferred (GETEX); < 6.2 works via GET+EXPIRE fallback |
   | `ROUTER_CLASSIFIER_URL` | `http://<gpu-box>:8891` | |
   | `ROUTER_DATABASE_URL` | `postgresql://...` | decision logging; empty string disables logging entirely |
   | `ROUTER_POLICIES_PATH` | `/app/policies.yaml` | hot-reloaded on mtime change (<= 60s lag) |
   | `ROUTER_CAPTURE_FIRST_MESSAGES` | `true` | raw first messages into the restricted table (90d retention) |

4. Migrations: `ROUTER_DATABASE_URL=... uv run scripts/migrate.py`
5. Retention: schedule `scripts/retention_purge.py` daily (cron/timer).

## Deploying classifier-svc (GPU box)

```bash
sudo cp deploy/gpu-box/ollama-env.conf /etc/systemd/system/ollama.service.d/router.conf
sudo systemctl daemon-reload && sudo systemctl restart ollama
cp deploy/gpu-box/classifier-svc.service ~/.config/systemd/user/
systemctl --user daemon-reload && systemctl --user enable --now classifier-svc
curl http://127.0.0.1:8891/healthz   # 503 while warming, 200 when the model is in VRAM
```

Health monitoring: poll `/healthz` from the gateway side -> alert. An outage
is invisible to users (hook times out at 1s and pins the default model) but
must be visible in Metabase: watch the `fallback` event rate.

## Client setup

### Claude Code (Anthropic `/v1/messages`)

```bash
export ANTHROPIC_BASE_URL="https://<gateway-host>"
export ANTHROPIC_AUTH_TOKEN="sk-<personal litellm key>"
export ANTHROPIC_MODEL="lite-auto"
# Claude Code's background/haiku calls must name a model in your model_list:
export ANTHROPIC_DEFAULT_HAIKU_MODEL="claude-sonnet-5"
```

Claude Code sends `x-claude-code-session-id`, so sessions pin on that header
and get subagent-aware routing and system-prompt path overrides.

### OpenCode (OpenAI-compatible `/chat/completions`)

OpenCode connects as an OpenAI-compatible provider. Add to `opencode.json`
(global `~/.config/opencode/opencode.json` or per-project):

```json
{
  "provider": {
    "litellm": {
      "npm": "@ai-sdk/openai-compatible",
      "name": "lite-router",
      "options": {
        "baseURL": "https://<gateway-host>/v1",
        "apiKey": "sk-<personal litellm key>"
      },
      "models": { "lite-auto": { "name": "lite-auto (session router)" } }
    }
  }
}
```

OpenCode **cannot** send a per-conversation session-id header, so the router
pins its sessions by the content-hash fallback (`sha256(key + system + first
message)`). This is stable as long as OpenCode replays the same system prompt +
first user message each turn — if it injects volatile content (timestamps,
cwd) into the system prompt, sessions re-classify each turn (safe, but no pin
persistence). Watch the `classified`-event rate per `client=opencode` to
confirm pins hold. Path overrides for OpenCode use the `x-lite-tier` header
(the system-prompt cwd auto-detection is Claude-Code-only).

## Escalation / override cheat sheet

- Force a bigger model for one session: send header `x-router-escalate: true`
  (one rung up, max 3 per session in the example ladder), or just pick a
  concrete model in the client — explicit model names always bypass the router
  (logged as override).
- Force-pin a repo to the top-of-ladder tier: add a pattern to `path_overrides`
  in `policies.yaml` via PR. As a client-side, any-client fallback the
  `x-lite-tier: high` header does the same per request (use whatever tier name
  `path_overrides.force_tier` is set to).

## Rollback

| Symptom | Action |
|---|---|
| Anything router-related broken | `ROUTER_ENABLED=false`, restart proxy — virtual model = default-tier alias |
| Routing quality bad, keep data | `SHADOW_MODE=true`, restart proxy |
| Classifier down | Nothing to do (fail-open); fix the GPU box, watch `fallback` rate |
| Redis down | Sessions fail open to the default tier; no restart needed |

## Verifying a deploy

```bash
# 1. Routed request pins per classifier:
curl -s $GW/v1/messages -H "Authorization: Bearer $KEY" \
  -H "x-claude-code-session-id: deploy-check-$RANDOM" \
  -d '{"model":"lite-auto","max_tokens":10,"messages":[{"role":"user","content":"what does ls -la do?"}]}'
# 2. Decision rows appear:
psql -c "SELECT event_type, policy_name, model, shadow FROM router_decisions ORDER BY id DESC LIMIT 5"
# 3. Tier distribution sane after some traffic:
psql -c "SELECT * FROM router_session_rollup ORDER BY started_at DESC LIMIT 20"
```

## Known gotchas

- **Fallback chains must never go down-tier** (see comment in
  `litellm_config.yaml`) — a downward fallback silently violates the
  no-downgrade invariant.
- **Bedrock/Vertex deployments** have known `anthropic-beta` header-forwarding
  issues in LiteLLM (prompt caching breaks): keep upstreams direct-Anthropic,
  or strip/rewrite the header per target before adding such deployments.
- The hook only escalates on evidence it can see in the request body; if a
  client sends truncated histories, only user-retry text and the escalate
  header will fire.
