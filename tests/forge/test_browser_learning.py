"""Acceptance coverage for the offline job-search skill."""

import asyncio
import os
import sqlite3

import pytest

from jarvis.control.cancel import CancelToken
from jarvis.forge import browser
from jarvis.forge.engine import Limits, Procedure, Refused
from jarvis.forge.store import Store
from jarvis.forge.tool import ForgeMindTool, command
from tests.forge.test_forge import context


def test_job_commands_accept_only_public_parameters():
    assert command("Forge learn jobs Engineer Remote.") == ("learn", "jobs", "engineer", "remote")
    assert command("forge replay 123 designer onsite") == ("replay", "123", "designer", "onsite")
    assert command("forge learn jobs secret@example.org remote") is None
    assert command("forge replay 123 engineer https://example.org") is None
    with pytest.raises(Refused, match="invalid_job_search_parameters"):
        browser.Search.read("private value", "remote")


def test_existing_sqlite_schema_and_procedures_are_preserved(tmp_path):
    store = Store(tmp_path / "forge.sqlite")
    with store.connection() as db:
        db.execute("DROP TABLE skill_lineage")
        db.execute("PRAGMA user_version = 1")
        db.execute(
            "INSERT INTO procedures VALUES (1, ?, 'promoted')",
            (
                '{"schema_version":1,"target":"forge-text-v1","action":"type_text",'
                '"parameter":"marker","precondition":"blank","postcondition":"exact_marker"}',
            ),
        )
    assert store.get(1) == (Procedure(), "promoted")
    with store.connection() as db:
        assert db.execute("PRAGMA user_version").fetchone() == (2,)


@pytest.mark.asyncio
async def test_browser_cancellation_and_retry(tmp_path, monkeypatch):
    started = asyncio.Event()
    store = Store(tmp_path / "forge.sqlite")
    tool = ForgeMindTool(store, None, Limits())
    delivery = context("forge learn jobs engineer remote")

    async def run(search, token, trace, saved):
        started.set()
        await token.wait_until_cancelled()
        raise Refused("cancelled")

    monkeypatch.setattr("jarvis.forge.tool.run_browser", run)
    task = asyncio.create_task(tool.execute({}, delivery))
    await started.wait()
    assert (await tool.execute({}, context("forge stop"))).success
    assert (await task).error == "cancelled"
    assert (await tool.execute({}, delivery)).error == "cancelled"
    assert store.list() == []

    async def recovered(search, token, trace, saved):
        procedure = browser.BrowserProcedure(1)
        trace.append(browser._verified(search, procedure))
        return procedure

    monkeypatch.setattr("jarvis.forge.tool.run_browser", recovered)
    assert (await tool.execute({}, context(delivery.user_utterance))).success


@pytest.mark.asyncio
@pytest.mark.skipif(not os.environ.get("FORGE_BROWSER_ACCEPTANCE"), reason="requires Chromium")
async def test_browser_learning_lifecycle_in_chromium(tmp_path, monkeypatch):
    store = Store(tmp_path / "forge.sqlite")
    tool = ForgeMindTool(store, None, Limits(seconds=45))
    revision = 1
    calls = 0

    async def fixture(search, token, trace, saved):
        nonlocal calls
        calls += 1
        return await browser.run_fixture(search, token, trace, saved, revision=revision)

    monkeypatch.setattr("jarvis.forge.tool.run_browser", fixture)

    async def invoke(text):
        return await tool.execute({}, context(f"forge {text}"))

    learn = context("forge learn jobs engineer remote")
    rejected = await invoke("learn jobs private@example.org remote")
    assert rejected.error == "exact_forge_command_required"
    learned = await tool.execute({}, learn)
    assert learned.success, learned
    assert learned.output["jobs"] == ["engineering-remote"]
    assert (await tool.execute({}, learn)).output == learned.output
    assert calls == 1
    version = learned.output["version"]
    assert (await invoke("learn jobs designer onsite")).error == "procedure_already_saved"
    assert len(store.list()) == 1
    assert (await invoke(f"replay {version} designer onsite")).error == "promoted_version_required"
    assert (
        await invoke(f"validate {version} engineer remote")
    ).error == "independent_validation_required"
    assert (await invoke(f"validate {version} designer onsite")).success
    assert (await invoke(f"promote {version}")).success

    tool = ForgeMindTool(Store(store.path), None, Limits(seconds=45))
    replayed = await invoke(f"replay {version} designer remote")
    assert replayed.success and replayed.output["jobs"] == ["design-remote"]
    revision = 2
    assert (await invoke(f"replay {version} designer remote")).error == "interface_drift"
    inspection = await invoke(f"inspect {version}")
    assert inspection.output["recent_failures"][0] == "interface_drift"
    improved = await invoke(f"improve {version} engineer remote")
    assert improved.success
    child = improved.output["version"]
    assert store.parent(child) == version
    assert (await invoke(f"validate {child} designer onsite")).success
    assert (await invoke(f"promote {child}")).success
    assert (await invoke(f"replay {child} engineer onsite")).success
    assert (await invoke(f"rollback {child}")).success
    assert (await invoke(f"replay {version} designer remote")).error == "interface_drift"

    with sqlite3.connect(store.path) as db:
        dump = "\n".join(db.iterdump())
    assert "private@example.org" not in dump
    assert "<form" not in dump and "https://" not in dump


@pytest.mark.asyncio
@pytest.mark.skipif(not os.environ.get("FORGE_BROWSER_ACCEPTANCE"), reason="requires Chromium")
async def test_wrong_result_is_not_verified(monkeypatch):
    monkeypatch.setattr(browser, "JOBS", ())
    trace = []
    with pytest.raises(Refused, match="goal_not_verified"):
        await browser.run_fixture(
            browser.Search.read("engineer", "remote"), CancelToken(), trace, None
        )
    assert trace == []
