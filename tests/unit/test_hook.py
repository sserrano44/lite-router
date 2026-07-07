"""Hook orchestration tests: routing decisions, fail-open, shadow, overrides."""

import pytest
from fakeredis import aioredis as fakeaioredis

from router_common.events import EventType

from session_router import config
from session_router.hook import RipioAutoRouter
from session_router.session_store import SessionStore
from session_router.state_machine import ClassifyResult


class FakeClassifier:
    def __init__(self, result=None):
        self.result = result
        self.calls = 0

    async def classify(self, first_message, system_summary, repo_hints):
        self.calls += 1
        return self.result


class CapturingLog:
    def __init__(self):
        self.events = []

    def emit(self, event):
        self.events.append(event)

    def by_type(self, event_type):
        return [e for e in self.events if e.event_type == event_type]


class FakeAuth:
    key_alias = "test-alias"
    user_id = "u1"


def make_router(classify_result=None):
    store = SessionStore(ttl_seconds=100)
    store._client = fakeaioredis.FakeRedis(decode_responses=True)
    classifier = FakeClassifier(classify_result)
    log = CapturingLog()
    return RipioAutoRouter(store=store, classifier=classifier, decision_log=log)


def make_data(session_id="sess-1", message="add a login endpoint", system="", model="ripio-auto",
              messages=None, extra_headers=None):
    headers = {}
    if session_id:
        headers["x-claude-code-session-id"] = session_id
    headers.update(extra_headers or {})
    return {
        "model": model,
        "system": system,
        "messages": messages if messages is not None else [{"role": "user", "content": message}],
        "proxy_server_request": {"headers": headers},
    }


@pytest.fixture(autouse=True)
def live_mode(monkeypatch):
    """Default tests to live routing; shadow tests override."""
    monkeypatch.setattr(config, "SHADOW_MODE", False)
    monkeypatch.setattr(config, "ROUTER_ENABLED", True)
    monkeypatch.setattr(config, "SUBAGENT_ROUTING_ENABLED", False)


async def test_first_request_pins_classifier_result():
    router = make_router(ClassifyResult("quick_lookup", "claude-haiku-4-5", 0.9, 42))
    data = make_data()
    out = await router.async_pre_call_hook(FakeAuth(), None, data, "anthropic_messages")
    assert out["model"] == "claude-haiku-4-5"
    assert router.decision_log.by_type(EventType.CLASSIFIED)
    assert router.decision_log.by_type(EventType.PINNED)


async def test_session_stickiness_no_second_classify():
    router = make_router(ClassifyResult("quick_lookup", "claude-haiku-4-5", 0.9))
    for _ in range(3):
        data = make_data()
        out = await router.async_pre_call_hook(FakeAuth(), None, data, "anthropic_messages")
        assert out["model"] == "claude-haiku-4-5"
    assert router.classifier.calls == 1


async def test_path_override_skips_classifier():
    router = make_router(ClassifyResult("quick_lookup", "claude-haiku-4-5", 0.9))
    data = make_data(system="Working directory: /home/dev/capyfi\n")
    out = await router.async_pre_call_hook(FakeAuth(), None, data, "anthropic_messages")
    assert out["model"] == "claude-opus-4-8"
    assert router.classifier.calls == 0
    pinned = router.decision_log.by_type(EventType.PINNED)
    assert pinned and pinned[0].detail["path_override"] == "capyfi"


async def test_classifier_down_pins_default():
    router = make_router(None)
    data = make_data()
    out = await router.async_pre_call_hook(FakeAuth(), None, data, "anthropic_messages")
    assert out["model"] == "claude-sonnet-4-6"
    fallback = router.decision_log.by_type(EventType.FALLBACK)
    assert fallback and fallback[0].detail["reason"] == "classifier_unavailable"


async def test_escalation_ratchet_and_cap():
    router = make_router(ClassifyResult("quick_lookup", "claude-haiku-4-5", 0.9))
    out = await router.async_pre_call_hook(FakeAuth(), None, make_data(), "anthropic_messages")
    assert out["model"] == "claude-haiku-4-5"

    def failing_history(n_failures, total):
        msgs = [{"role": "user", "content": "task"}]
        for i in range(n_failures):
            msgs.append({"role": "assistant", "content": "trying"})
            msgs.append({"role": "user", "content": [
                {"type": "tool_result", "tool_use_id": "t", "content": f"FAILED run {i} of {total}"}
            ]})
        return msgs

    out = await router.async_pre_call_hook(
        FakeAuth(), None, make_data(messages=failing_history(2, "a")), "anthropic_messages")
    assert out["model"] == "claude-sonnet-4-6"

    out = await router.async_pre_call_hook(
        FakeAuth(), None, make_data(messages=failing_history(4, "b")), "anthropic_messages")
    assert out["model"] == "claude-opus-4-8"

    # Third escalation would exceed the ladder; stays pinned at opus.
    out = await router.async_pre_call_hook(
        FakeAuth(), None, make_data(messages=failing_history(6, "c")), "anthropic_messages")
    assert out["model"] == "claude-opus-4-8"
    assert len(router.decision_log.by_type(EventType.ESCALATED)) == 2


async def test_escalate_header():
    router = make_router(ClassifyResult("standard_dev", "claude-sonnet-4-6", 0.8))
    await router.async_pre_call_hook(FakeAuth(), None, make_data(), "anthropic_messages")
    out = await router.async_pre_call_hook(
        FakeAuth(), None, make_data(extra_headers={"x-router-escalate": "true"}),
        "anthropic_messages")
    assert out["model"] == "claude-opus-4-8"


async def test_concrete_model_bypasses():
    router = make_router()
    data = make_data(model="claude-opus-4-8")
    out = await router.async_pre_call_hook(FakeAuth(), None, data, "anthropic_messages")
    assert out is None
    overrides_ = router.decision_log.by_type(EventType.OVERRIDE)
    assert len(overrides_) == 1
    # Repeated side-channel calls with the same model are deduped.
    await router.async_pre_call_hook(FakeAuth(), None, make_data(model="claude-opus-4-8"),
                                     "anthropic_messages")
    assert len(router.decision_log.by_type(EventType.OVERRIDE)) == 1


async def test_router_disabled_returns_none(monkeypatch):
    monkeypatch.setattr(config, "ROUTER_ENABLED", False)
    router = make_router()
    out = await router.async_pre_call_hook(FakeAuth(), None, make_data(), "anthropic_messages")
    assert out is None
    assert router.classifier.calls == 0


async def test_unroutable_call_type_ignored():
    router = make_router()
    out = await router.async_pre_call_hook(FakeAuth(), None, make_data(), "aembedding")
    assert out is None


async def test_shadow_mode_routes_default_but_logs_tier(monkeypatch):
    monkeypatch.setattr(config, "SHADOW_MODE", True)
    router = make_router(ClassifyResult("hard_dev", "claude-opus-4-8", 0.95))
    data = make_data()
    out = await router.async_pre_call_hook(FakeAuth(), None, data, "anthropic_messages")
    assert out["model"] == "claude-sonnet-4-6"  # routed to default
    stash = out["metadata"]["ripio_router"]
    assert stash["model"] == "claude-opus-4-8"  # decision preserved
    assert stash["shadow"] is True
    pinned = router.decision_log.by_type(EventType.PINNED)
    assert pinned[0].model == "claude-opus-4-8" and pinned[0].shadow is True


async def test_redis_down_fails_open():
    router = make_router(ClassifyResult("quick_lookup", "claude-haiku-4-5", 0.9))

    class Broken:
        def __getattr__(self, name):
            async def boom(*a, **kw):
                raise ConnectionError("redis down")
            return boom

    router.store._client = Broken()
    out = await router.async_pre_call_hook(FakeAuth(), None, make_data(), "anthropic_messages")
    assert out["model"] == "claude-sonnet-4-6"


async def test_internal_exception_fails_open():
    router = make_router()

    async def boom(key):
        raise RuntimeError("unexpected")

    router.store.get_and_refresh = boom
    out = await router.async_pre_call_hook(FakeAuth(), None, make_data(), "anthropic_messages")
    assert out["model"] == "claude-sonnet-4-6"


async def test_lost_claim_race_gets_default():
    router = make_router(ClassifyResult("quick_lookup", "claude-haiku-4-5", 0.9))
    await router.store.claim_for_classification("sess-1", {"state": "classifying"})
    out = await router.async_pre_call_hook(FakeAuth(), None, make_data(), "anthropic_messages")
    assert out["model"] == "claude-sonnet-4-6"
    assert router.classifier.calls == 0


async def test_derived_key_for_non_claude_code_clients():
    router = make_router(ClassifyResult("standard_dev", "claude-sonnet-4-6", 0.7))
    data = make_data(session_id=None)
    out = await router.async_pre_call_hook(FakeAuth(), None, data, "anthropic_messages")
    assert out["model"] == "claude-sonnet-4-6"
    stash = out["metadata"]["ripio_router"]
    assert stash["key_source"] == "derived"
    # Same conversation replayed -> same session key -> no reclassification.
    await router.async_pre_call_hook(FakeAuth(), None, make_data(session_id=None),
                                     "anthropic_messages")
    assert router.classifier.calls == 1


async def test_metadata_stash_prefers_litellm_metadata():
    router = make_router(ClassifyResult("standard_dev", "claude-sonnet-4-6", 0.7))
    data = make_data()
    data["litellm_metadata"] = {"headers": {}}
    out = await router.async_pre_call_hook(FakeAuth(), None, data, "anthropic_messages")
    assert "ripio_router" in out["litellm_metadata"]
    assert "metadata" not in out
