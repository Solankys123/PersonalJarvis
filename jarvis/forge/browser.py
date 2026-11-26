"""Read-only procedural learning against an in-memory job-search website."""

from __future__ import annotations

import asyncio
import json
import logging
import os
import sys
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlsplit
from uuid import uuid4

from .engine import Evidence, Refused, State, Verdict

logger = logging.getLogger(__name__)
FIXTURE = "forge-job-search"
URL = "https://forge-jobs.invalid/search"


class Role(StrEnum):
    ENGINEER = "engineer"
    DESIGNER = "designer"


class Location(StrEnum):
    REMOTE = "remote"
    ONSITE = "onsite"


class BrowserStep(StrEnum):
    FILL_ROLE = "fill_role"
    SELECT_LOCATION = "select_location"
    SUBMIT_GET = "submit_get"
    VERIFY_RESULTS = "verify_results"


JOBS = (
    ("engineering-remote", Role.ENGINEER, Location.REMOTE),
    ("engineering-onsite", Role.ENGINEER, Location.ONSITE),
    ("design-remote", Role.DESIGNER, Location.REMOTE),
    ("design-onsite", Role.DESIGNER, Location.ONSITE),
    ("design-studio", Role.DESIGNER, Location.ONSITE),
)
EXPECTED = {
    (Role.ENGINEER, Location.REMOTE): ["engineering-remote"],
    (Role.ENGINEER, Location.ONSITE): ["engineering-onsite"],
    (Role.DESIGNER, Location.REMOTE): ["design-remote"],
    (Role.DESIGNER, Location.ONSITE): ["design-onsite", "design-studio"],
}


@dataclass(frozen=True)
class Search:
    role: Role
    location: Location

    @classmethod
    def read(cls, role: str, location: str) -> Search:
        try:
            return cls(Role(role), Location(location))
        except ValueError:
            raise Refused("invalid_job_search_parameters") from None

    @property
    def marker(self) -> str:
        return f"{self.role}:{self.location}"


@dataclass(frozen=True)
class BrowserProcedure:
    fixture_version: int
    schema_version: int = 4
    target: str = FIXTURE
    steps: tuple[BrowserStep, ...] = tuple(BrowserStep)

    def validate(self) -> None:
        if type(self.fixture_version) is not int or self.fixture_version not in {1, 2}:
            raise Refused("unsupported_procedure")
        if self != BrowserProcedure(self.fixture_version):
            raise Refused("unsupported_procedure")

    def verifies(self, trace: list[Evidence]) -> bool:
        if len(trace) != 1:
            return False
        event = trace[0]
        return (
            event.fixture == self.target
            and event.fixture_version == self.fixture_version
            and event.action == "verify_results"
            and event.before == State.BLANK
            and event.after == State.MATCH
            and event.reason == "verified"
            and event.verdict == Verdict.SUCCESS
            and bool(event.target_id)
            and event.marker in {Search(role, place).marker for role in Role for place in Location}
        )


def _layout(revision: int) -> tuple[str, str]:
    return ("Job title", "Work location") if revision == 1 else ("Role", "Work arrangement")


def _document(revision: int, query: dict[str, list[str]]) -> str:
    role_label, location_label = _layout(revision)
    results = ""
    if set(query) == {"role", "location"}:
        results = "".join(
            f'<li data-job-id="{job}">{job}</li>'
            for job, role, location in JOBS
            if query["role"] == [role] and query["location"] == [location]
        )
    return f"""<!doctype html><html lang="en"><title>Fixture jobs</title>
        <main data-revision="{revision}"><h1>Fixture jobs</h1>
        <form method="get" action="/search">
        <label for="role">{role_label}</label><input id="role" name="role" required>
        <label for="location">{location_label}</label>
        <select id="location" name="location" required>
        <option value="">Choose a location</option>
        <option value="remote">Remote</option><option value="onsite">On-site</option>
        </select><button type="submit">Search</button>
        </form><ul aria-label="Job results">{results}</ul></main></html>"""


async def _search_page(
    page: Any, search: Search, token: Any, saved: BrowserProcedure | None
) -> BrowserProcedure:
    def check_cancel() -> None:
        if token.is_cancelled():
            raise Refused("cancelled")

    check_cancel()
    await page.goto(URL)
    revision = await page.locator("main").get_attribute("data-revision")
    if revision not in {"1", "2"}:
        raise Refused("interface_drift")
    procedure = BrowserProcedure(int(revision))
    if saved is not None and procedure != saved:
        raise Refused("interface_drift")

    role_label, location_label = _layout(procedure.fixture_version)
    role = page.get_by_label(role_label, exact=True)
    location = page.get_by_label(location_label, exact=True)
    submit = page.get_by_role("button", name="Search", exact=True)
    if [await control.count() for control in (role, location, submit)] != [1, 1, 1]:
        raise Refused("interface_drift")
    form = page.locator("form")
    if (
        await form.count() != 1
        or await form.get_attribute("method") != "get"
        or await form.get_attribute("action") != "/search"
    ):
        raise Refused("interface_drift")

    recorded = []
    check_cancel()
    await role.fill(search.role)
    recorded.append(BrowserStep.FILL_ROLE)
    check_cancel()
    await location.select_option(search.location)
    recorded.append(BrowserStep.SELECT_LOCATION)
    check_cancel()
    await submit.click()
    await page.wait_for_url(f"{URL}?*")
    recorded.append(BrowserStep.SUBMIT_GET)
    check_cancel()

    # Verify from the new document and public catalog, independently of the click receipt.
    query = parse_qs(urlsplit(page.url).query)
    displayed = await page.locator("[data-job-id]").evaluate_all(
        "(items) => items.map(item => item.getAttribute('data-job-id'))"
    )
    requested = {"role": [search.role], "location": [search.location]}
    expected = EXPECTED[search.role, search.location]
    if query != requested or displayed != expected:
        raise Refused("goal_not_verified")
    recorded.append(BrowserStep.VERIFY_RESULTS)
    return BrowserProcedure(procedure.fixture_version, steps=tuple(recorded))


async def run_fixture(
    search: Search,
    token: Any,
    trace: list[Evidence],
    saved: BrowserProcedure | None,
    *,
    revision: int = 1,
    executable: str | None = None,
) -> BrowserProcedure:
    """Run only bundled HTML in a fresh, sandboxed browser without network routes."""
    try:
        from playwright.async_api import Error, async_playwright
    except ImportError:
        raise Refused("isolated_browser_unavailable") from None

    if token.is_cancelled():
        raise Refused("cancelled")

    async def serve(route: Any) -> None:
        request = route.request
        destination, _, query = request.url.partition("?")
        if destination != URL or request.method != "GET" or request.resource_type != "document":
            await route.abort()
            return
        await route.fulfill(
            content_type="text/html",
            headers={"Content-Security-Policy": "default-src 'none'; form-action 'self'"},
            body=_document(revision, parse_qs(query, max_num_fields=2)),
        )

    try:
        async with async_playwright() as browser_runtime:
            browser = await browser_runtime.chromium.launch(
                executable_path=executable,
                chromium_sandbox=True,
                args=[
                    "--disable-background-networking",
                    "--host-resolver-rules=MAP * ~NOTFOUND",
                    "--no-proxy-server",
                ],
            )
            try:
                context = await browser.new_context(
                    java_script_enabled=False,
                    service_workers="block",
                    accept_downloads=False,
                    offline=True,
                )
                await context.route("**/*", serve)
                page = await context.new_page()
                page.set_default_timeout(3000)
                procedure = await _search_page(page, search, token, saved)
            finally:
                await browser.close()
    except Error:
        logger.debug("FORGE browser capability failed")
        raise Refused("isolated_browser_failed") from None

    if token.is_cancelled():
        raise Refused("cancelled")
    trace.append(_verified(search, procedure))
    return procedure


async def run_browser(
    search: Search,
    token: Any,
    trace: list[Evidence],
    saved: BrowserProcedure | None,
) -> BrowserProcedure:
    """Use Jarvis's managed Playwright runtime without opening a user profile."""
    from jarvis.core.process_tree import make_process_tree
    from jarvis.core.process_utils import NO_WINDOW_CREATIONFLAGS
    from jarvis.society.browser import install

    source = await asyncio.to_thread(Path(__file__).resolve)
    if not await asyncio.to_thread(source.is_file) or not await asyncio.to_thread(
        install.is_installed
    ):
        raise Refused("isolated_browser_unavailable")
    if token.is_cancelled():
        raise Refused("cancelled")
    tree = make_process_tree("forge-browser")
    if not tree.supports_containment:
        tree.close()
        raise Refused("isolated_browser_unavailable")
    executable, python, env = await asyncio.to_thread(
        lambda: (install.browser_executable(), install.venv_python(), install.worker_env())
    )
    request = json.dumps(
        [
            search.role,
            search.location,
            saved.fixture_version if saved else None,
            str(executable),
        ]
    ).encode("utf-8")
    child = None
    tasks: list[asyncio.Task] = []
    try:
        # Import only the trusted source checkout in the dependency-isolated worker.
        child = await asyncio.create_subprocess_exec(
            str(python),
            "-I",
            "-B",
            "-c",
            "import sys; sys.path.insert(0, sys.argv[1]); "
            "from jarvis.forge.browser import worker; worker()",
            str(source.parents[2]),
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.DEVNULL,
            env=env,
            creationflags=NO_WINDOW_CREATIONFLAGS,
            **({"start_new_session": True} if os.name != "nt" else {}),
        )
        tree.assign(child.pid)
        exchange = asyncio.create_task(child.communicate(request))
        stopped = asyncio.create_task(token.wait_until_cancelled())
        tasks = [exchange, stopped]
        await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
        if token.is_cancelled():
            raise Refused("cancelled")
        stdout, _ = exchange.result()
        if child.returncode != 0:
            raise Refused("isolated_browser_failed")
        reply = json.loads(stdout)
        if reply in ("interface_drift", "goal_not_verified", "isolated_browser_unavailable"):
            logger.debug("FORGE browser worker refused the run: %s", reply)
            raise Refused(reply)
        if not isinstance(reply, list) or len(reply) != 2 or not isinstance(reply[1], list):
            raise Refused("isolated_browser_failed")
        procedure = BrowserProcedure(reply[0], steps=tuple(map(BrowserStep, reply[1])))
        procedure.validate()
        if saved is not None and procedure != saved:
            raise Refused("interface_drift")
        trace.append(_verified(search, procedure))
        return procedure
    except (OSError, ValueError):
        logger.debug("FORGE could not use the managed browser worker")
        raise Refused("isolated_browser_failed") from None
    finally:
        tree.close()
        if child is not None and child.returncode is None:
            try:
                child.kill()
            except ProcessLookupError:
                logger.debug("FORGE browser worker already exited")
            await child.wait()
        for task in tasks:
            task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)


def _verified(search: Search, procedure: BrowserProcedure) -> Evidence:
    return Evidence(
        uuid4().hex,
        "verify_results",
        search.marker,
        State.BLANK,
        State.MATCH,
        Verdict.SUCCESS,
        "verified",
        1,
        FIXTURE,
        procedure.fixture_version,
    )


def worker() -> None:
    """Entry point for the managed runtime. The parent owns the deadline."""

    class WorkerToken:
        def is_cancelled(self) -> bool:
            return False

    role, location, revision, executable = json.load(sys.stdin)
    saved = BrowserProcedure(revision) if revision is not None else None
    try:
        procedure = asyncio.run(
            run_fixture(
                Search.read(role, location), WorkerToken(), [], saved, executable=executable
            )
        )
        reply: list | str = [procedure.fixture_version, procedure.steps]
    except Refused as exc:
        # Send the refusal over IPC so the parent can log and record the outcome.
        reply = str(exc)
    sys.stdout.buffer.write(json.dumps(reply).encode("utf-8"))
