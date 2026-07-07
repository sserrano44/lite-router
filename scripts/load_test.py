#!/usr/bin/env python3
"""Load test the pinned-request hot path (R: p95 overhead < 5ms).

Runs N sessions x M pinned requests against the compose stack twice —
ROUTER_ENABLED comparison is done by targeting two proxy ports or by
env-flipping the stack between runs. Practical default: compare the
routed proxy against itself with concrete-model requests (bypass path),
which isolates the router's added work on the same deployment.

Usage:
    uv run scripts/load_test.py [--sessions 200] [--requests 10] [--url http://127.0.0.1:4000]
"""

from __future__ import annotations

import argparse
import asyncio
import statistics
import time
import uuid

import httpx

MASTER_KEY = "sk-test-master"


async def run_session(client: httpx.AsyncClient, url: str, model: str,
                      requests_per_session: int, latencies: list[float]) -> None:
    session = f"load-{uuid.uuid4().hex[:12]}"
    messages = [{"role": "user", "content": "add a unit test for the parser"}]
    for i in range(requests_per_session):
        t0 = time.perf_counter()
        resp = await client.post(
            f"{url}/v1/messages",
            json={"model": model, "max_tokens": 20, "messages": messages},
            headers={
                "Authorization": f"Bearer {MASTER_KEY}",
                "x-claude-code-session-id": session,
            },
        )
        elapsed = (time.perf_counter() - t0) * 1000
        resp.raise_for_status()
        if i > 0:  # skip the first (classification) request
            latencies.append(elapsed)
        messages = messages + [
            {"role": "assistant", "content": "done"},
            {"role": "user", "content": f"follow-up {i}"},
        ]


async def measure(url: str, model: str, sessions: int, requests_per_session: int,
                  concurrency: int) -> list[float]:
    latencies: list[float] = []
    sem = asyncio.Semaphore(concurrency)
    limits = httpx.Limits(max_connections=concurrency + 10)
    async with httpx.AsyncClient(timeout=60, limits=limits) as client:
        async def bounded():
            async with sem:
                await run_session(client, url, model, requests_per_session, latencies)

        await asyncio.gather(*[bounded() for _ in range(sessions)])
    return latencies


def stats(name: str, lat: list[float]) -> dict:
    lat = sorted(lat)
    n = len(lat)
    out = {
        "n": n,
        "p50": lat[n // 2],
        "p95": lat[int(n * 0.95)],
        "p99": lat[int(n * 0.99)],
        "mean": statistics.mean(lat),
    }
    print(f"{name:14s} n={n:5d}  p50={out['p50']:.1f}ms  p95={out['p95']:.1f}ms  "
          f"p99={out['p99']:.1f}ms  mean={out['mean']:.1f}ms")
    return out


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:4000")
    ap.add_argument("--sessions", type=int, default=200)
    ap.add_argument("--requests", type=int, default=10)
    ap.add_argument("--concurrency", type=int, default=50)
    args = ap.parse_args()

    print(f"warming up connections against {args.url} ...")
    await measure(args.url, "claude-sonnet-4-6", 5, 3, 5)

    print(f"\n== bypass path (concrete model, router skipped) ==")
    bypass = await measure(args.url, "claude-sonnet-4-6", args.sessions, args.requests,
                           args.concurrency)
    b = stats("bypass", bypass)

    print(f"\n== routed path (ripio-auto, pinned requests) ==")
    routed = await measure(args.url, "ripio-auto", args.sessions, args.requests,
                           args.concurrency)
    r = stats("routed", routed)

    overhead = r["p95"] - b["p95"]
    print(f"\npinned-path p95 overhead vs bypass: {overhead:.2f}ms (budget: < 5ms)")
    if overhead >= 5:
        print("!! OVER BUDGET")
        raise SystemExit(1)
    print("OK")


if __name__ == "__main__":
    asyncio.run(main())
