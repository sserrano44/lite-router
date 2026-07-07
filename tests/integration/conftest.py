"""Fixtures for integration tests against the docker-compose stack.

Prereq: docker compose up -d --build, migrations applied to :5433
(ROUTER_DATABASE_URL=postgresql://postgres:postgres@127.0.0.1:5433/postgres
 uv run scripts/migrate.py).
"""

import json

import asyncpg
import httpx
import pytest

from it_helpers import CLASSIFIER_URL, DATABASE_URL, LITELLM_URL, SHADOW_URL


def pytest_collection_modifyitems(config, items):
    # Disruptive tests (pause shared containers) must run last.
    items.sort(key=lambda item: item.get_closest_marker("disruptive") is not None)


def pytest_configure(config):
    config.addinivalue_line("markers", "disruptive: pauses shared compose services")


@pytest.fixture
async def client():
    async with httpx.AsyncClient(base_url=LITELLM_URL, timeout=30) as c:
        yield c


@pytest.fixture
async def shadow_client():
    async with httpx.AsyncClient(base_url=SHADOW_URL, timeout=30) as c:
        yield c


@pytest.fixture
async def db():
    conn = await asyncpg.connect(DATABASE_URL)
    await conn.set_type_codec(
        "jsonb", encoder=json.dumps, decoder=json.loads, schema="pg_catalog"
    )
    yield conn
    await conn.close()


@pytest.fixture
async def classifier_control():
    async with httpx.AsyncClient(base_url=CLASSIFIER_URL, timeout=10) as c:
        yield c
        await c.post("/control", json={"delay_ms": 0, "fail": False})
