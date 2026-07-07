# ripio-auto Router — Deploy & Operations Runbook

## Components

| Unit | Where | What |
|---|---|---|
| LiteLLM proxy + `session_router` hook | AWS gateway | Routes `ripio-auto`, pins sessions in Redis, logs to Postgres |
| `classifier-svc` | RTX 3090 box (next to Ollama) | `POST /classify` via qwen3-coder:30b, once per session |
| `policies.yaml` | this repo, mounted into both | Tiers, path overrides, escalation rules — change via PR |

## Deploying the hook (AWS)

1. Build the image: `docker build -f deploy/litellm/Dockerfile -t litellm-ripio .`
   (installs `router-common` + `session-router` into the LiteLLM venv).
2. Mount/COPY next to each other in `/app`: `litellm_config.yaml` (as
   `config.yaml`), `custom_callbacks.py`, `policies.yaml`.
3. Environment:

   | Var | Shadow launch | Notes |
   |---|---|---|
   | `ROUTER_ENABLED` | `true` | `false` = full rollback: `ripio-auto` becomes a plain Sonnet alias, zero routing, zero logging |
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

Health monitoring: poll `/healthz` from the AWS side -> Slack alert. An outage
is invisible to users (hook times out at 1s and pins the default model) but
must be visible in Metabase: watch the `fallback` event rate.

## Client setup (engineers, macOS)

```bash
export ANTHROPIC_BASE_URL="https://<gateway-host>"
export ANTHROPIC_AUTH_TOKEN="sk-<personal litellm key>"
export ANTHROPIC_MODEL="ripio-auto"
export ANTHROPIC_DEFAULT_HAIKU_MODEL="claude-haiku-4-5"
```

## Escalation / override cheat sheet

- Force a bigger model for one session: send header `x-router-escalate: true`
  (one rung up, max 2 per session), or just pick a concrete model in the
  client — explicit model names always bypass the router (logged as override).
- Force-pin a repo to Opus: add a pattern to `path_overrides` in
  `policies.yaml` via PR. As a client-side fallback the `x-ripio-tier:
  hard_dev` header does the same per request source.

## Rollback

| Symptom | Action |
|---|---|
| Anything router-related broken | `ROUTER_ENABLED=false`, restart proxy — `ripio-auto` = Sonnet alias |
| Routing quality bad, keep data | `SHADOW_MODE=true`, restart proxy |
| Classifier down | Nothing to do (fail-open); fix the GPU box, watch `fallback` rate |
| Redis down | Sessions fail open to Sonnet; no restart needed |

## Verifying a deploy

```bash
# 1. Routed request pins per classifier:
curl -s $GW/v1/messages -H "Authorization: Bearer $KEY" \
  -H "x-claude-code-session-id: deploy-check-$RANDOM" \
  -d '{"model":"ripio-auto","max_tokens":10,"messages":[{"role":"user","content":"what does ls -la do?"}]}'
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
