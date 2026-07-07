"""classifier-svc: POST /classify, GET /healthz, POST /reload.

/classify always answers 200 with a valid tier — internal failures map to the
default policy with confidence 0, so the router hook's only failure mode is
an HTTP-level timeout.
"""

from __future__ import annotations

import asyncio
import logging
import os
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, Response
from pydantic import BaseModel, Field
from router_common.policies import PoliciesConfig, load_policies

from classifier_svc import ollama, prompt

logger = logging.getLogger("classifier_svc")

POLICIES_PATH = os.environ.get("ROUTER_POLICIES_PATH", "policies.yaml")
RELOAD_WATCH_INTERVAL_S = 5.0


class ClassifyRequest(BaseModel):
    first_message: str = Field(default="", max_length=8000)
    system_summary: str = Field(default="", max_length=2000)
    repo_hints: dict = Field(default_factory=dict)


class ClassifyResponse(BaseModel):
    policy_name: str
    model: str
    confidence: float
    latency_ms: int


class State:
    policies: PoliciesConfig
    policies_mtime: float = 0.0
    ready: bool = False
    started_at: float = 0.0


state = State()


def _load_policies() -> None:
    state.policies = load_policies(POLICIES_PATH)
    state.policies_mtime = os.stat(POLICIES_PATH).st_mtime
    logger.info("policies loaded (%d tiers)", len(state.policies.tiers))


async def _warmup() -> None:
    """Fire one real classification so Ollama loads the model into VRAM;
    healthz stays 503 until this succeeds."""
    result = await ollama.chat_classify(
        prompt.system_prompt(state.policies),
        prompt.user_prompt("warmup: explain this regex", "", {}),
        prompt.response_schema(state.policies),
    )
    if result is None:
        raise RuntimeError("warmup classification against Ollama failed")
    logger.info("warmup ok: %s", result)


async def _watch_policies() -> None:
    while True:
        await asyncio.sleep(RELOAD_WATCH_INTERVAL_S)
        try:
            mtime = os.stat(POLICIES_PATH).st_mtime
            if mtime != state.policies_mtime:
                _load_policies()
        except Exception:
            logger.warning("policies hot-reload failed", exc_info=True)


@asynccontextmanager
async def lifespan(app: FastAPI):
    state.started_at = time.time()
    _load_policies()
    retry_delay = 2.0
    while True:
        try:
            await _warmup()
            break
        except Exception:
            logger.warning("warmup failed, retrying in %.0fs", retry_delay, exc_info=True)
            await asyncio.sleep(retry_delay)
            retry_delay = min(retry_delay * 2, 30.0)
    state.ready = True
    watcher = asyncio.create_task(_watch_policies())
    yield
    watcher.cancel()
    await ollama.aclose()


app = FastAPI(title="ripio-auto classifier-svc", lifespan=lifespan)


@app.get("/healthz")
async def healthz(response: Response):
    if not state.ready:
        response.status_code = 503
        return {"status": "warming_up"}
    return {
        "status": "ok",
        "model": ollama.CLASSIFIER_MODEL,
        "policies_version": state.policies.version,
        "uptime_s": int(time.time() - state.started_at),
    }


@app.post("/reload")
async def reload_policies():
    _load_policies()
    return {"status": "reloaded", "tiers": state.policies.tier_names()}


@app.post("/classify", response_model=ClassifyResponse)
async def classify(req: ClassifyRequest):
    t0 = time.perf_counter()
    policies = state.policies
    default_tier = policies.default_tier()
    result = await ollama.chat_classify(
        prompt.system_prompt(policies),
        prompt.user_prompt(req.first_message, req.system_summary, req.repo_hints),
        prompt.response_schema(policies),
    )
    latency_ms = int((time.perf_counter() - t0) * 1000)

    policy_name, confidence = default_tier.name, 0.0
    if isinstance(result, dict):
        candidate = result.get("policy_name")
        tier = policies.tier_by_name(candidate) if isinstance(candidate, str) else None
        if tier is not None:
            policy_name = tier.name
            try:
                confidence = max(0.0, min(1.0, float(result.get("confidence", 0.0))))
            except (TypeError, ValueError):
                confidence = 0.0

    tier = policies.tier_by_name(policy_name) or default_tier
    return ClassifyResponse(
        policy_name=tier.name, model=tier.model,
        confidence=confidence, latency_ms=latency_ms,
    )


def main() -> None:
    import uvicorn

    logging.basicConfig(level=logging.INFO)
    uvicorn.run(app, host=os.environ.get("HOST", "0.0.0.0"),
                port=int(os.environ.get("PORT", "8891")))


if __name__ == "__main__":
    main()
