"""Deterministic mock classifier for integration tests.

Keyword routing: "solidity" -> hard_dev, "what does"/"explain" -> quick_lookup,
else standard_dev. POST /control adjusts delay/failure at runtime; GET /stats
exposes the call count so tests can assert the classifier was (not) called.
"""

from __future__ import annotations

import asyncio

from fastapi import FastAPI, Response
from pydantic import BaseModel

app = FastAPI(title="mock-classifier")

MODELS = {
    "quick_lookup": "claude-haiku-4-5",
    "standard_dev": "claude-sonnet-4-6",
    "hard_dev": "claude-opus-4-8",
}

state = {"calls": 0, "delay_ms": 0, "fail": False}


class ClassifyRequest(BaseModel):
    first_message: str = ""
    system_summary: str = ""
    repo_hints: dict = {}


class Control(BaseModel):
    delay_ms: int | None = None
    fail: bool | None = None


@app.post("/classify")
async def classify(req: ClassifyRequest, response: Response):
    state["calls"] += 1
    if state["delay_ms"]:
        await asyncio.sleep(state["delay_ms"] / 1000)
    if state["fail"]:
        response.status_code = 500
        return {"error": "induced failure"}
    text = req.first_message.lower()
    if "solidity" in text:
        policy = "hard_dev"
    elif "what does" in text or text.startswith("explain"):
        policy = "quick_lookup"
    else:
        policy = "standard_dev"
    return {
        "policy_name": policy,
        "model": MODELS[policy],
        "confidence": 0.99,
        "latency_ms": state["delay_ms"],
    }


@app.post("/control")
async def control(cfg: Control):
    if cfg.delay_ms is not None:
        state["delay_ms"] = cfg.delay_ms
    if cfg.fail is not None:
        state["fail"] = cfg.fail
    return state


@app.get("/stats")
async def stats():
    return state


@app.get("/healthz")
async def healthz():
    return {"status": "ok"}
