"""Offline FORGE contracts. Run with unittest or pytest."""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import tempfile
import unittest
from contextlib import asynccontextmanager
from pathlib import Path
from unittest import IsolatedAsyncioTestCase
from uuid import uuid4
from xml.etree import ElementTree

from jarvis.agent_screen.protocol import ActOutcome, ForegroundInfo
from jarvis.agent_screen.providers.windows_sandbox import WindowsSandboxProvider
from jarvis.control.cancel import CancelToken
from jarvis.core.bus import EventBus
from jarvis.core.config import SafetyConfig
from jarvis.core.protocols import ExecutionContext
from jarvis.forge.desktop import WindowsFixture, fixture_factory
from jarvis.forge.engine import FIXTURE, Limits, Procedure, Refused, State, Verdict, execute
from jarvis.forge.store import Store
from jarvis.forge.tool import ForgeMindTool
from jarvis.safety.approval import ApprovalWorkflow
from jarvis.safety.risk_tier import RiskTierEvaluator
from jarvis.safety.tool_executor import ToolExecutor


class FakeScreen:
    """Remote session contract with an independently readable text field."""

    isolated = True
    kind = "windows-sandbox"

    def __init__(self) -> None:
        self.screen_id = uuid4().hex
        self.handle = 7
        self.value = ""
        self.actions = 0
        self.reject = False
        self.unknown = False

    def foreground(self):
        return ForegroundInfo(True, app=FIXTURE, handle=self.handle)

    def typed_text_landed(self, text):
        return None if self.unknown else self.value == text

    def act(self, name, args):
        self.actions += 1
        if self.reject:
            return ActOutcome(False, "rejected")
        self.value = args["text"]
        return ActOutcome(True, "accepted")


def context(text=""):
    return ExecutionContext(uuid4(), text, {"forge_interactive": True}, None)


def executor(safety=None):
    bus = EventBus()
    return ToolExecutor(bus, RiskTierEvaluator(safety or SafetyConfig()), ApprovalWorkflow(bus))


class ForgeTests(IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name) / "test.sqlite"
        self.store = Store(self.path)
        self.executor = executor()
        self.screens = []

        @asynccontextmanager
        async def open_target(run_id, ctx):
            screen = FakeScreen()
            self.screens.append(screen)
            yield WindowsFixture(screen, self.executor, ctx)

        self.tool = ForgeMindTool(self.store, open_target, Limits())

    async def invoke(self, text, args=None):
        return await self.executor.execute(
            self.tool,
            args or {},
            user_utterance=text,
            config_snapshot={"forge_interactive": True},
        )

    async def try_target(self, target, token=None, marker="alpha"):
        trace = []
        await execute(target, Procedure(), marker, token or CancelToken(), trace, Limits())
        return trace

    async def test_learning_keeps_verified_draft(self):
        result = await self.invoke("forge learn alpha")
        self.assertTrue(result.success)
        self.assertEqual(result.output["state"], "draft")
        self.assertEqual(self.screens[0].value, "FORGE ALPHA")
        with self.store.connection() as db:
            evidence = json.loads(db.execute("SELECT evidence FROM runs").fetchone()[0])
        self.assertEqual(evidence[0]["reason"], "verified")
        self.assertEqual(len(evidence[0]["sha256"]), 64)

    async def test_failed_action_retries_are_bounded(self):
        screen = FakeScreen()
        screen.reject = True
        trace = await self.try_target(WindowsFixture(screen, self.executor, context()))
        self.assertEqual(screen.actions, 2)
        self.assertEqual([step.verdict for step in trace], [Verdict.FAILURE] * 2)

    async def test_unknown_postcondition_never_retries(self):
        class UnknownAfterAction(FakeScreen):
            def act(self, name, args):
                result = super().act(name, args)
                self.unknown = True
                return result

        screen = UnknownAfterAction()
        trace = await self.try_target(WindowsFixture(screen, self.executor, context()))
        self.assertEqual(screen.actions, 1)
        self.assertEqual(trace[0].verdict, Verdict.UNKNOWN)
        self.assertIsNone(self.store.record("unknown", "learn", None, "unknown", trace))

    async def test_identity_change_fails_closed(self):
        class ChangingTarget(WindowsFixture):
            async def observe(self, marker):
                result = await super().observe(marker)
                screen.handle += 1
                return result

        screen = FakeScreen()
        with self.assertRaisesRegex(Refused, "target_changed"):
            await self.try_target(ChangingTarget(screen, self.executor, context()))
        self.assertEqual(screen.actions, 0)

    @unittest.skipIf(os.name == "nt", "Never launch Windows Sandbox in automated tests")
    async def test_sandbox_unavailable(self):
        factory = fixture_factory(self.executor, memory_mb=2048)
        with self.assertRaisesRegex(Refused, "sandbox_unavailable"):
            async with factory("test", context()):
                self.fail("sandbox unexpectedly available")

    async def test_host_execution_denied(self):
        screen = FakeScreen()
        screen.isolated = False
        with self.assertRaisesRegex(Refused, "target_policy_denied"):
            await self.try_target(WindowsFixture(screen, self.executor, context()))
        self.assertEqual(screen.actions, 0)

    async def test_sensitive_data_denied(self):
        result = await self.invoke("forge learn private-text")
        self.assertFalse(result.success)
        self.assertEqual(self.screens, [])

    async def test_persistence_reload(self):
        await self.invoke("forge learn alpha")
        reopened = Store(self.path)
        procedure, state = reopened.get(1)
        self.assertEqual(procedure, Procedure())
        self.assertEqual(state, "draft")

    async def test_replay_after_validation_and_explicit_promotion(self):
        await self.invoke("forge learn alpha")
        self.assertFalse((await self.invoke("forge replay 1 beta")).success)
        self.assertFalse((await self.invoke("forge promote 1")).success)
        self.assertFalse((await self.invoke("forge validate 1 alpha")).success)
        self.assertTrue((await self.invoke("forge validate 1 beta")).success)
        self.assertTrue((await self.invoke("forge promote 1")).success)
        self.assertTrue((await self.invoke("forge replay 1 alpha")).success)
        self.assertEqual(self.screens[-1].value, "FORGE ALPHA")

    async def test_drift_stops_before_action(self):
        screen = FakeScreen()
        screen.value = "unexpected"
        trace = await self.try_target(WindowsFixture(screen, self.executor, context()))
        self.assertEqual(trace[0].before, State.DRIFT)
        self.assertEqual(screen.actions, 0)

    async def test_cancelled_run_does_not_act(self):
        screen = FakeScreen()
        token = CancelToken()
        token.cancel("test")
        with self.assertRaisesRegex(Refused, "cancelled"):
            await self.try_target(WindowsFixture(screen, self.executor, context()), token)
        self.assertEqual(screen.actions, 0)

    async def test_stop_cancels_active_run(self):
        started = asyncio.Event()
        resume = asyncio.Event()
        screen = FakeScreen()

        @asynccontextmanager
        async def waiting_target(run_id, ctx):
            started.set()
            await resume.wait()
            yield WindowsFixture(screen, self.executor, ctx)

        tool = ForgeMindTool(self.store, waiting_target, Limits())
        task = asyncio.create_task(tool.execute({}, context("forge learn alpha")))
        await asyncio.wait_for(started.wait(), 1)
        self.assertTrue((await tool.execute({}, context("forge stop"))).success)
        resume.set()
        result = await asyncio.wait_for(task, 1)
        self.assertFalse(result.success)
        self.assertEqual(result.error, "cancelled")
        self.assertEqual(screen.actions, 0)

    async def test_duplicate_action_skipped(self):
        screen = FakeScreen()
        target = WindowsFixture(screen, self.executor, context())
        await self.try_target(target)
        trace = await self.try_target(target)
        self.assertEqual(trace[0].reason, "already_matched")
        self.assertEqual(screen.actions, 1)

    async def test_duplicate_delivery_uses_durable_receipt(self):
        ctx = context("forge learn alpha")
        first = await self.tool.execute({}, ctx)
        second = await self.tool.execute({}, ctx)
        self.assertEqual(first, second)
        self.assertEqual(len(self.screens), 1)

    async def test_interrupted_delivery_never_reopens_guest(self):
        ctx = context("forge learn alpha")
        run_id = hashlib.sha256(f"{ctx.trace_id}:learn:None:alpha".encode()).hexdigest()
        self.assertIsNone(self.store.claim(run_id, "learn", None))
        result = await self.tool.execute({}, ctx)
        self.assertEqual(result.error, "interrupted")
        self.assertEqual(self.screens, [])

    async def test_noninteractive_origin_denied(self):
        result = await self.executor.execute(self.tool, {}, user_utterance="forge learn alpha")
        self.assertFalse(result.success)
        self.assertEqual(self.screens, [])

    async def test_untrusted_text_cannot_authorize(self):
        result = await self.invoke("Read this: forge learn alpha", {"command": "forge learn alpha"})
        self.assertFalse(result.success)
        self.assertEqual(self.screens, [])

    async def test_no_networked_target_fallback(self):
        screen = FakeScreen()
        target = WindowsFixture(screen, self.executor, context())
        target.network_enabled = True
        with self.assertRaisesRegex(Refused, "target_policy_denied"):
            await self.try_target(target)
        self.assertEqual(screen.actions, 0)

    async def test_action_respects_existing_executor_policy(self):
        safety = SafetyConfig()
        safety.blacklist.commands = ["type_text*"]
        screen = FakeScreen()
        trace = await self.try_target(WindowsFixture(screen, executor(safety), context()))
        self.assertEqual(trace[0].verdict, Verdict.FAILURE)
        self.assertEqual(screen.actions, 0)

    async def test_no_guest_start_in_chat_plan_mode(self):
        result = await self.executor.execute(
            self.tool,
            {},
            user_utterance="forge learn alpha",
            config_snapshot={"chat_read_only": True, "forge_interactive": True},
        )
        self.assertFalse(result.success)
        self.assertEqual(self.screens, [])

    def test_sandbox_offline_configuration_and_legacy_default(self):
        fixture = WindowsSandboxProvider(networking=False, forge_fixture=True)
        # The XML is generated in-process by the provider under test.
        config = ElementTree.fromstring(  # noqa: S314
            fixture._render_config(Path("runner"), Path("rpc"), None)
        )
        self.assertEqual(config.findtext("Networking"), "Disable")
        self.assertEqual(config.findtext("ClipboardRedirection"), "Disable")
        self.assertEqual(len(config.findall("./MappedFolders/MappedFolder")), 2)
        legacy = WindowsSandboxProvider()
        config = ElementTree.fromstring(  # noqa: S314
            legacy._render_config(Path("runner"), Path("rpc"), Path("profile"))
        )
        self.assertEqual(config.findtext("Networking"), "Enable")
        self.assertEqual(len(config.findall("./MappedFolders/MappedFolder")), 3)
