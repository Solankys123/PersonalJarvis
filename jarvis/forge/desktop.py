"""Bind the restricted Windows fixture to the existing screen and tool ports."""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from typing import Any

from jarvis.agent_screen.manager import AgentScreenManager, ScreenSettings
from jarvis.agent_screen.port import RemoteScreenPort
from jarvis.agent_screen.protocol import AgentScreenUnavailable
from jarvis.agent_screen.providers.windows_sandbox import WindowsSandboxProvider
from jarvis.agent_screen.tools import screen_bound_tools

from .engine import FIXTURE, MARKERS, Refused, State


class WindowsFixture:
    kind = FIXTURE
    network_enabled = False

    def __init__(self, session: Any, executor: Any, context: Any) -> None:
        self._session = session
        self._port = RemoteScreenPort(session)
        self._tool = screen_bound_tools(session)["type_text"]
        self._executor = executor
        self._context = context
        self.isolated = session.isolated and session.kind == "windows-sandbox"

    @property
    def identity(self) -> str:
        return self._session.screen_id

    async def fingerprint(self) -> tuple[Any, ...]:
        target = await asyncio.to_thread(self._session.foreground)
        return target.signature() if target.app == FIXTURE and target.handle else ("none",)

    async def observe(self, marker: str) -> State:
        matched = await self._port.verify_typed_text(MARKERS[marker])
        if matched is True:
            return State.MATCH
        if matched is None:
            return State.UNKNOWN
        blank = await self._port.verify_typed_text("")
        if blank is None:
            return State.UNKNOWN
        return State.BLANK if blank else State.DRIFT

    async def apply(self, marker: str, fingerprint: tuple[Any, ...], token: Any) -> bool:
        if not self.isolated or marker not in MARKERS or await self.fingerprint() != fingerprint:
            raise Refused("capability_denied")
        result = await self._executor.execute(
            self._tool,
            {"text": MARKERS[marker], "_expected_window_signature": fingerprint},
            trace_id=self._context.trace_id,
            config_snapshot=self._context.config,
            cancel_token=token,
        )
        return result.success


def fixture_factory(executor: Any, *, memory_mb: int):
    provider = WindowsSandboxProvider(memory_mb=memory_mb, networking=False, forge_fixture=True)
    manager = AgentScreenManager(
        settings=ScreenSettings(provider="windows-sandbox", max_screens=1),
        providers={"windows-sandbox": provider},
    )

    @asynccontextmanager
    async def open_fixture(run_id: str, context: Any):
        # The provider starts a blocking system process. Keep acquisition owned
        # through cancellation so a late guest is still released.
        pending = asyncio.create_task(manager.acquire(run_id, purpose=FIXTURE))
        try:
            lease = await asyncio.shield(pending)
        except asyncio.CancelledError:
            try:
                lease = await pending
            except (AgentScreenUnavailable, OSError):
                pass
            else:
                await manager.release(lease)
            raise
        except (AgentScreenUnavailable, OSError):
            raise Refused("sandbox_unavailable") from None
        try:
            yield WindowsFixture(lease.session, executor, context)
        finally:
            await manager.release(lease)

    return open_fixture
