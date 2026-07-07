#!/usr/bin/env python3
"""Shadow-mode analysis: replay logged first messages through a classifier,
report tier distribution, disagreement vs. logged decisions, and projected
spend delta vs. an all-Opus baseline.

Usage:
    ROUTER_DATABASE_URL=postgresql://... uv run scripts/replay_shadow.py \
        [--classifier http://127.0.0.1:8891] [--days 7] [--no-replay]

Spend projection uses policies.yaml pricing and per-session token volumes.
Without LiteLLM spend-log access it falls back to a flat tokens-per-session
assumption (override with --mtok-in/--mtok-out per session).
"""

from __future__ import annotations

import argparse
import asyncio
import os
from collections import Counter
from pathlib import Path

import asyncpg
import httpx

from router_common.policies import load_policies

REPO_ROOT = Path(__file__).resolve().parent.parent


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--classifier", default="http://127.0.0.1:8891")
    ap.add_argument("--days", type=int, default=7)
    ap.add_argument("--no-replay", action="store_true",
                    help="skip re-classification; only report logged distribution")
    ap.add_argument("--mtok-in", type=float, default=2.0,
                    help="assumed input Mtok per session for spend projection")
    ap.add_argument("--mtok-out", type=float, default=0.05,
                    help="assumed output Mtok per session for spend projection")
    args = ap.parse_args()

    policies = load_policies(os.environ.get("ROUTER_POLICIES_PATH",
                                            REPO_ROOT / "policies.yaml"))
    db_url = os.environ.get(
        "ROUTER_DATABASE_URL", "postgresql://postgres:postgres@127.0.0.1:5433/postgres"
    )
    conn = await asyncpg.connect(db_url)
    try:
        rows = await conn.fetch(
            """
            SELECT d.session_key, d.policy_name, d.model, d.confidence,
                   m.raw_message, m.system_excerpt
            FROM router_decisions d
            LEFT JOIN router_first_messages m
                   ON m.first_message_hash = d.first_message_hash
            WHERE d.event_type = 'pinned'
              AND d.ts > now() - make_interval(days => $1)
            """,
            args.days,
        )
    finally:
        await conn.close()

    if not rows:
        print("no pinned sessions found in window")
        return

    dist = Counter(r["policy_name"] for r in rows)
    n = len(rows)
    print(f"sessions: {n} (last {args.days}d)")
    print("\ntier distribution (logged):")
    for tier in policies.tiers:
        c = dist.get(tier.name, 0)
        print(f"  {tier.name:14s} {c:5d}  {100*c/n:5.1f}%")

    # Spend projection vs. all-Opus baseline.
    opus = policies.tiers[-1].model
    baseline = spent = 0.0
    for r in rows:
        p_base = policies.pricing.get(opus)
        p_used = policies.pricing.get(r["model"]) or p_base
        if not p_base:
            print("\nno pricing in policies.yaml; skipping spend projection")
            break
        baseline += args.mtok_in * p_base.in_ + args.mtok_out * p_base.out
        spent += args.mtok_in * p_used.in_ + args.mtok_out * p_used.out
    else:
        delta = 100 * (1 - spent / baseline) if baseline else 0.0
        print(f"\nprojected spend (flat {args.mtok_in}/{args.mtok_out} Mtok in/out per session):")
        print(f"  all-opus baseline: ${baseline:,.0f}")
        print(f"  routed:            ${spent:,.0f}   ({delta:+.1f}% vs baseline; target -30%)")

    if args.no_replay:
        return

    # Replay stored first messages through the (possibly newer) classifier.
    replayable = [r for r in rows if r["raw_message"]]
    if not replayable:
        print("\nno stored raw first messages to replay")
        return
    disagree = 0
    replay_dist: Counter = Counter()
    async with httpx.AsyncClient(timeout=30) as client:
        for r in replayable:
            resp = await client.post(f"{args.classifier}/classify", json={
                "first_message": r["raw_message"],
                "system_summary": r["system_excerpt"] or "",
                "repo_hints": {},
            })
            got = resp.json()["policy_name"]
            replay_dist[got] += 1
            if got != r["policy_name"]:
                disagree += 1
    rn = len(replayable)
    print(f"\nreplay against {args.classifier}: {rn} messages")
    for tier in policies.tiers:
        c = replay_dist.get(tier.name, 0)
        print(f"  {tier.name:14s} {c:5d}  {100*c/rn:5.1f}%")
    print(f"disagreement with logged decisions: {disagree}/{rn} ({100*disagree/rn:.1f}%)")


if __name__ == "__main__":
    asyncio.run(main())
