import asyncio

import pytest
from fakeredis import aioredis as fakeaioredis
from redis.exceptions import ConnectionError as RedisConnectionError

from session_router.session_store import SessionStore


@pytest.fixture
def store():
    s = SessionStore(ttl_seconds=100, key_prefix="router:session:")
    s._client = fakeaioredis.FakeRedis(decode_responses=True)
    return s


async def test_miss_returns_none(store):
    assert await store.get_and_refresh("nope") is None


async def test_write_and_read(store):
    await store.write_pin("k1", {"state": "pinned", "model": "m"})
    rec = await store.get_and_refresh("k1")
    assert rec == {"state": "pinned", "model": "m"}


async def test_ttl_refresh_on_read(store):
    await store.write_pin("k1", {"state": "pinned"})
    await store._client.expire("router:session:k1", 5)
    await store.get_and_refresh("k1")
    ttl = await store._client.ttl("router:session:k1")
    assert ttl > 5


async def test_claim_exactly_once_under_contention(store):
    results = await asyncio.gather(
        *[store.claim_for_classification("race", {"state": "classifying"}) for _ in range(20)]
    )
    assert sum(results) == 1


async def test_claim_placeholder_readable(store):
    await store.claim_for_classification("k", {"state": "classifying", "model": "d"})
    rec = await store.get_and_refresh("k")
    assert rec["state"] == "classifying"


async def test_update_keeps_ttl(store):
    await store.write_pin("k1", {"state": "pinned", "escalations": 0})
    await store._client.expire("router:session:k1", 42)
    await store.update("k1", {"state": "pinned", "escalations": 1})
    ttl = await store._client.ttl("router:session:k1")
    assert 0 < ttl <= 42
    rec = await store.get_and_refresh("k1")
    assert rec["escalations"] == 1


class _BrokenRedis:
    def __getattr__(self, name):
        async def boom(*args, **kwargs):
            raise RedisConnectionError("down")

        return boom

    def pipeline(self, *a, **kw):
        raise RedisConnectionError("down")


@pytest.fixture
def broken_store():
    s = SessionStore(ttl_seconds=100, key_prefix="router:session:")
    s._client = _BrokenRedis()
    return s


async def test_fail_open_get(broken_store):
    assert await broken_store.get_and_refresh("k") is None


async def test_fail_open_claim(broken_store):
    assert await broken_store.claim_for_classification("k", {}) is False


async def test_fail_open_writes(broken_store):
    await broken_store.write_pin("k", {"a": 1})
    await broken_store.update("k", {"a": 1})  # must not raise
