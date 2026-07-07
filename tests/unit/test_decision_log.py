import asyncio

from router_common.events import DecisionEvent, EventType

from session_router.decision_log import DecisionLog


def _event(i=0):
    return DecisionEvent(
        session_key=f"s{i}", event_type=EventType.PINNED, model="claude-sonnet-4-6"
    )


async def test_no_database_url_is_noop():
    log = DecisionLog(database_url="")
    log.emit(_event())
    assert log._queue is None


async def test_queue_full_drops_without_raising():
    log = DecisionLog(database_url="postgresql://ignored")
    log._queue = asyncio.Queue(maxsize=1)
    log._task = asyncio.get_running_loop().create_future()  # prevent flusher spawn
    log.emit(_event(1))
    log.emit(_event(2))
    assert log.dropped == 1
    assert log._queue.qsize() == 1


async def test_batch_flush_shape(monkeypatch):
    log = DecisionLog(database_url="postgresql://ignored")
    written = []

    async def fake_write(batch):
        written.extend(batch)

    monkeypatch.setattr(log, "_write_batch", fake_write)
    for i in range(3):
        log.emit(_event(i))
    for _ in range(50):
        await asyncio.sleep(0.05)
        if len(written) == 3:
            break
    assert [e.session_key for e in written] == ["s0", "s1", "s2"]
    await log.aclose()


async def test_write_failure_requeues_and_backs_off(monkeypatch):
    log = DecisionLog(database_url="postgresql://ignored")
    attempts = []

    async def failing_write(batch):
        attempts.append(list(batch))
        raise RuntimeError("pg down")

    monkeypatch.setattr(log, "_write_batch", failing_write)
    log.emit(_event(0))
    # Retry cadence is backoff (1s) + batch window (2s); allow up to 8s.
    for _ in range(160):
        await asyncio.sleep(0.05)
        if len(attempts) >= 2:
            break
    # The same event is retried after the write failure.
    assert len(attempts) >= 2
    assert attempts[0][0].session_key == "s0"
    await log.aclose()
