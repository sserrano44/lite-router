import asyncio
import subprocess
import time

import pytest
import redis.asyncio as aioredis

from it_helpers import (
    REDIS_URL,
    failing_history,
    routed_model,
    send,
    session_id,
    user,
    wait_for_rows,
)


async def test_session_pinning_same_model_five_requests(client, db):
    session = session_id()
    first = await send(client, session, [user("what does this decorator do?")])
    pinned = routed_model(first)
    assert pinned == "claude-haiku-4-5"
    for i in range(4):
        out = await send(client, session, [user(f"follow-up question {i}")])
        assert routed_model(out) == pinned
    rows = await wait_for_rows(db, session, "classified")
    assert rows[0]["policy_name"] == "quick_lookup"
    # Exactly one classification for the whole session.
    all_classified = await db.fetch(
        "SELECT * FROM router_decisions WHERE session_key=$1 AND event_type='classified'",
        session,
    )
    assert len(all_classified) == 1


async def test_redis_ttl_refreshes_on_each_request(client):
    session = session_id()
    await send(client, session, [user("hello there")])
    r = aioredis.from_url(REDIS_URL, decode_responses=True)
    try:
        key = f"router:session:{session}"
        await r.expire(key, 60)
        await send(client, session, [user("second request")])
        ttl = await r.ttl(key)
        assert ttl > 60, f"TTL was not refreshed: {ttl}"
    finally:
        await r.aclose()


async def test_escalation_ratchet_and_cap(client, db):
    session = session_id()
    out = await send(client, session, [user("what does this regex do?")])
    assert routed_model(out) == "claude-haiku-4-5"

    out = await send(client, session, failing_history(2, "a"))
    assert routed_model(out) == "claude-sonnet-4-6"

    out = await send(client, session, failing_history(4, "b"))
    assert routed_model(out) == "claude-opus-4-8"

    # Max 2 escalations: further failures keep the opus pin, no new event.
    out = await send(client, session, failing_history(6, "c"))
    assert routed_model(out) == "claude-opus-4-8"

    rows = await wait_for_rows(db, session, "escalated", min_count=2)
    assert len(rows) == 2
    assert [r["model"] for r in rows] == ["claude-sonnet-4-6", "claude-opus-4-8"]


async def test_escalate_header(client, db):
    session = session_id()
    out = await send(client, session, [user("write a unit test for this function")])
    assert routed_model(out) == "claude-sonnet-4-6"
    out = await send(
        client, session, [user("continue")], headers={"x-router-escalate": "true"}
    )
    assert routed_model(out) == "claude-opus-4-8"
    rows = await wait_for_rows(db, session, "escalated")
    assert rows[0]["detail"] is not None


async def test_concrete_model_bypasses_router(client, db):
    session = session_id()
    out = await send(client, session, [user("hi")], model="claude-opus-4-8")
    assert routed_model(out) == "claude-opus-4-8"
    rows = await wait_for_rows(db, session, "override")
    assert rows[0]["model"] == "claude-opus-4-8"


async def test_path_override_pins_opus_without_classifier(client, db, classifier_control):
    before = (await classifier_control.get("/stats")).json()["calls"]
    session = session_id()
    out = await send(
        client, session, [user("small tweak to the readme")],
        system="Working directory: /home/dev/capyfi\nIs a git repository: true",
    )
    assert routed_model(out) == "claude-opus-4-8"
    after = (await classifier_control.get("/stats")).json()["calls"]
    assert after == before, "classifier must not be called for path-override sessions"
    rows = await wait_for_rows(db, session, "pinned")
    assert rows[0]["detail"]["path_override"] == "capyfi"


async def test_classifier_failure_pins_default(client, db, classifier_control):
    await classifier_control.post("/control", json={"fail": True})
    session = session_id()
    out = await send(client, session, [user("what does this regex do?")])
    assert routed_model(out) == "claude-sonnet-4-6"  # default despite quick_lookup phrasing
    rows = await wait_for_rows(db, session, "fallback")
    assert rows[0]["detail"]["reason"] == "classifier_unavailable"


async def test_classifier_slow_times_out_to_default(client, classifier_control):
    await classifier_control.post("/control", json={"delay_ms": 2000})
    session = session_id()
    t0 = time.perf_counter()
    out = await send(client, session, [user("what does this regex do?")])
    elapsed = time.perf_counter() - t0
    assert routed_model(out) == "claude-sonnet-4-6"
    assert elapsed < 2.0, f"hook must give up at ~1s, took {elapsed:.2f}s"


async def test_concurrent_first_requests_single_classification(client, db):
    session = session_id()
    results = await asyncio.gather(
        *[send(client, session, [user("what does this regex do?")]) for _ in range(20)]
    )
    models = {routed_model(r) for r in results}
    # Winner gets the classified pin; racers may get the default for their
    # one in-flight request. No third model may ever appear.
    assert models <= {"claude-haiku-4-5", "claude-sonnet-4-6"}
    await asyncio.sleep(3)  # let the log flush settle
    rows = await db.fetch(
        "SELECT * FROM router_decisions WHERE session_key=$1 AND event_type='classified'",
        session,
    )
    assert len(rows) == 1


async def test_shadow_mode_routes_default_logs_real_tier(shadow_client, db):
    session = session_id()
    out = await send(shadow_client, session, [user("what does this regex do?")])
    # Shadow: actual routing is always the default model...
    assert routed_model(out) == "claude-sonnet-4-6"
    # ...but the decision rows carry the would-be tier, flagged shadow.
    rows = await wait_for_rows(db, session, "pinned")
    assert rows[0]["model"] == "claude-haiku-4-5"
    assert rows[0]["shadow"] is True


async def test_subagent_pins_one_tier_below_parent(client, db):
    session = session_id()
    out = await send(client, session, [user("refactor this solidity contract please")])
    assert routed_model(out) == "claude-opus-4-8"  # classifier: hard_dev

    out = await send(
        client, session, [user("explore the repo layout")],
        headers={
            "x-claude-code-agent-id": "sub-1",
            "x-claude-code-parent-agent-id": "parent-1",
        },
    )
    assert routed_model(out) == "claude-sonnet-4-6"  # one tier below hard_dev
    rows = await wait_for_rows(db, f"{session}:sub-1", "pinned")
    assert rows[0]["detail"]["subagent"] is True
    assert rows[0]["agent_id"] == "sub-1"


@pytest.mark.disruptive
async def test_redis_outage_fails_open(client):
    subprocess.run(["docker", "compose", "pause", "redis"], check=True, capture_output=True)
    try:
        session = session_id()
        out = await send(client, session, [user("what does this regex do?")])
        assert routed_model(out) == "claude-sonnet-4-6"  # default, no error
    finally:
        subprocess.run(["docker", "compose", "unpause", "redis"], check=True,
                       capture_output=True)
