"""Shared helpers for integration tests (importable, unlike conftest)."""

import asyncio
import os
import uuid

LITELLM_URL = os.environ.get("IT_LITELLM_URL", "http://127.0.0.1:4000")
SHADOW_URL = os.environ.get("IT_SHADOW_URL", "http://127.0.0.1:4001")
CLASSIFIER_URL = os.environ.get("IT_CLASSIFIER_URL", "http://127.0.0.1:8891")
DATABASE_URL = os.environ.get(
    "IT_DATABASE_URL", "postgresql://postgres:postgres@127.0.0.1:5433/postgres"
)
REDIS_URL = os.environ.get("IT_REDIS_URL", "redis://127.0.0.1:6390/0")
MASTER_KEY = "sk-test-master"


def session_id() -> str:
    return f"it-{uuid.uuid4().hex[:12]}"


def messages_payload(messages, model="ripio-auto", system=None):
    body = {"model": model, "max_tokens": 50, "messages": messages}
    if system is not None:
        body["system"] = system
    return body


async def send(client, session, messages, model="ripio-auto", system=None, headers=None):
    hdrs = {"Authorization": f"Bearer {MASTER_KEY}"}
    if session:
        hdrs["x-claude-code-session-id"] = session
    hdrs.update(headers or {})
    resp = await client.post(
        "/v1/messages", json=messages_payload(messages, model, system), headers=hdrs
    )
    resp.raise_for_status()
    return resp.json()


def routed_model(response_json) -> str:
    """mock_response content encodes which deployment served the request."""
    text = response_json["content"][0]["text"]
    assert text.startswith("routed:"), text
    return text.removeprefix("routed:")


def user(text):
    return {"role": "user", "content": text}


def assistant(text="working on it"):
    return {"role": "assistant", "content": text}


def tool_failure(text="FAILED tests/test_x.py :: 3 failed"):
    return {"role": "user", "content": [
        {"type": "tool_result", "tool_use_id": "t1", "content": text}
    ]}


def failing_history(n_failures, salt=""):
    msgs = [user(f"run the test suite {salt}")]
    for i in range(n_failures):
        msgs.append(assistant())
        msgs.append(tool_failure(f"FAILED run {i} {salt}"))
    return msgs


async def wait_for_rows(db, session, event_type=None, min_count=1, timeout_s=10):
    """Decision logging is async (2s batch window) — poll for rows."""
    query = "SELECT * FROM router_decisions WHERE session_key = $1"
    args = [session]
    if event_type:
        query += " AND event_type = $2"
        args.append(event_type)
    query += " ORDER BY id"
    deadline = asyncio.get_running_loop().time() + timeout_s
    while True:
        rows = await db.fetch(query, *args)
        if len(rows) >= min_count:
            return rows
        if asyncio.get_running_loop().time() > deadline:
            raise AssertionError(
                f"expected >= {min_count} {event_type or 'any'} rows for {session}, "
                f"got {len(rows)}"
            )
        await asyncio.sleep(0.5)
