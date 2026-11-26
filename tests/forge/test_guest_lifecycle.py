"""FORGE launcher and lease ownership checks without starting a Windows guest."""

import asyncio
from uuid import uuid4

import pytest

from jarvis.agent_screen.providers import windows_sandbox
from jarvis.core.protocols import ExecutionContext
from jarvis.forge import desktop


def test_forge_launcher_ignores_path_override(monkeypatch):
    monkeypatch.setattr(windows_sandbox, "_sandbox_binaries", lambda: ("untrusted", None))
    monkeypatch.setattr(windows_sandbox, "_system_sandbox_binaries", lambda: (None, "system"))
    assert windows_sandbox.WindowsSandboxProvider(
        networking=False, forge_fixture=True
    )._binaries() == (
        None,
        "system",
    )
    assert windows_sandbox.WindowsSandboxProvider()._binaries() == ("untrusted", None)


@pytest.mark.asyncio
async def test_repeated_cancellation_still_releases_late_guest(monkeypatch):
    entered = asyncio.Event()
    finish_boot = asyncio.Event()
    released = asyncio.Event()
    lease = object()

    async def acquire(self, *args, **kwargs):
        entered.set()
        await finish_boot.wait()
        return lease

    async def release(self, acquired):
        assert acquired is lease
        released.set()

    monkeypatch.setattr(desktop.AgentScreenManager, "acquire", acquire)
    monkeypatch.setattr(desktop.AgentScreenManager, "release", release)
    factory = desktop.fixture_factory(None, memory_mb=2048)
    ctx = ExecutionContext(uuid4(), "forge learn alpha", {}, None)

    async def run():
        async with factory("run", ctx):
            pytest.fail("cancelled run must not receive the guest")

    task = asyncio.create_task(run())
    await asyncio.wait_for(entered.wait(), 1)
    task.cancel()
    await asyncio.sleep(0)
    task.cancel()
    await asyncio.sleep(0)
    finish_boot.set()
    with pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(task, 1)
    assert released.is_set()


@pytest.mark.asyncio
async def test_teardown_finishes_before_cancellation_escapes():
    entered = asyncio.Event()
    finish_release = asyncio.Event()

    async def release():
        entered.set()
        await finish_release.wait()

    owner = asyncio.create_task(desktop._finish_owned_task(asyncio.create_task(release())))
    await asyncio.wait_for(entered.wait(), 1)
    owner.cancel()
    await asyncio.sleep(0)
    assert not owner.done()
    finish_release.set()
    with pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(owner, 1)
