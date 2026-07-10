"""OpenCode / OpenAI-compatible client routing (POST /chat/completions).

OpenCode connects as an OpenAI-compatible provider, carries the system prompt
as messages[0], sends tool results as role=tool messages, and cannot inject a
per-conversation session header — so it pins via the content-hash fallback.
"""

from it_helpers import (
    assistant,
    openai_failing_history,
    routed_model_openai,
    send_openai,
    session_id,
    user,
    wait_for_rows,
)


async def test_opencode_content_hash_pins_across_turns(client, classifier_control):
    """No session header: a stable system prompt + first user message must pin
    the session, so the classifier fires exactly once across turns."""
    before = (await classifier_control.get("/stats")).json()["calls"]
    # Salt the system prompt so the derived key is unique per run (Redis pins
    # persist across runs), but identical across this test's three turns.
    system = f"You are a coding agent. run={session_id()}"
    opener = user("what does this function do?")

    r1 = await send_openai(client, None, [opener], system=system)
    r2 = await send_openai(
        client, None, [opener, assistant(), user("and this one?")], system=system
    )
    r3 = await send_openai(
        client, None,
        [opener, assistant(), user("and this one?"), assistant(), user("thanks")],
        system=system,
    )
    after = (await classifier_control.get("/stats")).json()["calls"]

    assert routed_model_openai(r1) == "claude-sonnet-5"  # quick_lookup
    assert routed_model_openai(r2) == "claude-sonnet-5"
    assert routed_model_openai(r3) == "claude-sonnet-5"
    assert after - before == 1, "content-hash pin must classify only once"


async def test_opencode_client_label_logged(client, db):
    session = session_id()
    await send_openai(client, session, [user("what does this function do?")])
    rows = await wait_for_rows(db, session, "pinned")
    assert rows[0]["client"] == "opencode"


async def test_claude_code_client_label_logged(client, db):
    from it_helpers import send

    session = session_id()
    await send(client, session, [user("what does this function do?")])
    rows = await wait_for_rows(db, session, "pinned")
    assert rows[0]["client"] == "claude-code"


async def test_opencode_openai_tool_failure_escalates(client, db):
    session = session_id()
    r = await send_openai(client, session, [user("write a helper function")])
    assert routed_model_openai(r) == "grok-4.5"  # standard_dev

    r = await send_openai(client, session, openai_failing_history(2, "run the tests"))
    assert routed_model_openai(r) == "claude-opus-4-8"  # -> high

    rows = await wait_for_rows(db, session, "escalated")
    assert rows[0]["client"] == "opencode"
    assert rows[0]["detail"]["reason"] == "tool_failures"
