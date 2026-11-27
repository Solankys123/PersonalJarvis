"""Ephemeral evidence capture for the managed Browser-Use search agent."""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import sys
from contextlib import asynccontextmanager, nullcontext
from pathlib import Path
from time import monotonic
from urllib.parse import parse_qs, urlsplit
from uuid import uuid4


class SearchStopped(Exception):
    pass


log = logging.getLogger(__name__)


async def _visibility(context, page_session, foreground):
    if not foreground:
        return "background"
    if sys.platform != "win32":
        return "unsupported"
    from native_window import activate_owned_chrome  # type: ignore[import-not-found]

    try:
        # CDP normalizes a minimized window before the one native activation
        # attempt. The handle is selected solely from this temporary browser's PID.
        async with asyncio.timeout(3):
            session = await context.browser.new_browser_cdp_session()
            try:
                target = await page_session.send("Target.getTargetInfo")
                window = await session.send(
                    "Browser.getWindowForTarget", {"targetId": target["targetInfo"]["targetId"]}
                )
                if window["bounds"]["windowState"] == "minimized":
                    await session.send(
                        "Browser.setWindowBounds",
                        {"windowId": window["windowId"], "bounds": {"windowState": "normal"}},
                    )
                await page_session.send("Page.bringToFront")
                processes = await session.send("SystemInfo.getProcessInfo")
                pid = next(p["id"] for p in processes["processInfo"] if p["type"] == "browser")
                visible, active = await asyncio.to_thread(activate_owned_chrome, int(pid))
            finally:
                await session.detach()
    except Exception as exc:
        log.debug("FORGE browser activation unavailable: %s", type(exc).__name__)
        return "window_unavailable"
    if not visible:
        return "window_unavailable"
    return "foreground" if active else "foreground_denied"


class SearchCapture:
    def __init__(self, page, session, args, emit):
        self.page = page
        self.session = session
        self.args = args
        self.emit = emit
        self.steps: list[str] = []
        self.input_seen = False
        self.initial = None
        self.last_submit = None
        self.error = None
        self.finish_from_evidence = False

    def same_origin(self, url):
        parts = urlsplit(url)
        return (
            f"{parts.scheme}://{parts.netloc}" == self.args["origin"]
            and not parts.username
            and not parts.password
        )

    async def snapshot(self):
        if not self.same_origin(self.page.url) or len(self.page.context.pages) != 1:
            raise SearchStopped("browser_origin_mismatch")
        state = await self.page.evaluate(
            """query => {
                const blocked = document.querySelector(
                    'input[type=password],iframe[src*="captcha" i],[id*="captcha" i],'
                    + '[class*="captcha" i],input[autocomplete^="cc-" i]');
                const text = document.body?.innerText.slice(0, 64000) || '';
                const visible = element => element.getClientRects().length > 0
                    && getComputedStyle(element).visibility === 'visible';
                const searchLinks = Array.from(document.querySelectorAll(
                    '[id*="search-results" i] a[href], [class~="search-results" i] a[href]'
                ));
                const resultHeadings = document.querySelectorAll('h1,h2,[role=heading]');
                return {blocked: !!blocked, text,
                    matched: text.toLowerCase().includes(query.toLowerCase()),
                    renderedResults: searchLinks.some(visible) || Array.from(resultHeadings).some(
                        heading => visible(heading) && /\\b(results?|matches)\\b/i.test(
                            heading.innerText
                        ) && heading.innerText.toLowerCase().includes(query.toLowerCase())
                    ),
                    active: document.activeElement?.tagName || ''};
            }""",
            self.args["query"],
        )
        if state["blocked"]:
            raise SearchStopped("browser_restricted_page")
        return self.page.url, hashlib.sha256(state["text"].encode()).digest(), state

    def matches_results(self, state):
        query = self.args["query"]
        url_query = parse_qs(urlsplit(state[0]).query)
        matched = state[2]["matched"] or any(query in values for values in url_query.values())
        return state[1] != self.initial and state[:2] != self.last_submit and matched

    async def observed_results(self):
        if self.last_submit is None:
            return False
        expected = self.args["steps"]
        if expected and self.steps != list(expected[:-1]):
            return False
        state = await self.snapshot()
        return state[2]["renderedResults"] and self.matches_results(state)

    async def target(self, browser, index=None):
        if index is None:
            object_id = (
                await self.session.send(
                    "Runtime.evaluate", {"expression": "document.activeElement"}
                )
            )["result"].get("objectId")
        else:
            node = await browser.get_element_by_index(index)
            if node is None or node.target_id != browser.agent_focus_target_id:
                raise SearchStopped("browser_target_unavailable")
            object_id = (
                await self.session.send("DOM.resolveNode", {"backendNodeId": node.backend_node_id})
            )["object"].get("objectId")
        if not object_id:
            raise SearchStopped("browser_target_unavailable")
        try:
            result = await self.session.send(
                "Runtime.callFunctionOn",
                {
                    "objectId": object_id,
                    "returnByValue": True,
                    "functionDeclaration": """function () {
                        const fields = 'input:not([type]),input[type=text],'
                            + 'input[type=search],textarea';
                        const selector = fields + ',button,input[type=submit],[role=button]';
                        if (!this.isConnected || this.ownerDocument !== document
                            || !this.matches(selector) || !this.getClientRects().length
                            || this.matches(':disabled,[aria-disabled=true]')) return null;
                        const field = this.matches(fields);
                        const role = this.getAttribute('role');
                        const kind = !field ? 'button'
                            : role === 'searchbox' ? 'searchbox' : 'textbox';
                        const autocomplete = this.getAttribute('autocomplete') || '';
                        if (field && (this.readOnly || !['', 'off'].includes(autocomplete)))
                            return null;
                        const label = this.getAttribute('aria-label') || this.innerText
                            || this.getAttribute('value') || '';
                        if (/(captcha|verify you|sign in|log in|register|consent|agree|accept)/i
                            .test(label)) return null;
                        const siblings = Array.from(document.querySelectorAll(selector))
                            .filter(e => e.getClientRects().length);
                        return {field, value: field ? this.value : '',
                            shape: [kind, this.tagName, !!this.form, siblings.length,
                                siblings.indexOf(this)]};
                    }""",
                },
            )
        finally:
            await self.session.send("Runtime.releaseObject", {"objectId": object_id})
        data = result["result"].get("value")
        if data is None:
            raise SearchStopped("browser_target_restricted")
        signature = hashlib.sha256(json.dumps(data["shape"]).encode()).hexdigest()
        return data["field"], data["value"], signature

    def expect(self, step):
        expected = self.args["steps"]
        position = len(self.steps)
        if position >= self.args["budget"]:
            raise SearchStopped("action_budget_exceeded")
        if expected and (position >= len(expected) or step != expected[position]):
            raise SearchStopped("search_page_drift")

    async def prepare(self, browser, proposal):
        name, params = next(iter(proposal.items()))
        if name == "navigate":
            if self.steps or params.get("url") != self.args["url"] or params.get("new_tab"):
                raise SearchStopped("browser_origin_mismatch")
            self.expect(name)
            return name, None, None
        before = await self.snapshot()
        if name == "done":
            self.expect("verify_results")
            if params.get("success") is not True:
                reason = "model_declined"
            elif not self.input_seen or self.last_submit is None:
                reason = "missing_search_action"
            elif before[1] == self.initial or before[:2] == self.last_submit:
                reason = "page_unchanged"
            elif not self.matches_results(before):
                reason = "query_unconfirmed"
            elif self.finish_from_evidence and not before[2]["renderedResults"]:
                reason = "rendered_results_missing"
            else:
                reason = ""
            if reason:
                self.emit("forge_stage", stage="result_evidence", status=reason, elapsed_ms=0)
                raise SearchStopped("search_results_unverified")
            return "verify_results", before, None
        index = params.get("index")
        if name in {"click", "input_text"} and type(index) is int and index > 0:
            field, value, signature = await self.target(browser, index)
            if name == "input_text" and (
                not field or value or params.get("text") != self.args["query"]
            ):
                raise SearchStopped("browser_input_restricted")
        elif name == "send_keys" and params == {"keys": "Enter"} and self.input_seen:
            field, value, signature = await self.target(browser)
            if not field or value != self.args["query"]:
                raise SearchStopped("browser_target_restricted")
        else:
            raise SearchStopped("browser_action_restricted")
        step = f"{name}:{signature}"
        self.expect(step)
        return step, before, index

    async def verify(self, browser, step, before, index):
        name = step.split(":")[0]
        if name == "input_text":
            field, value, signature = await self.target(browser, index)
            if not field or value != self.args["query"] or step != f"{name}:{signature}":
                raise SearchStopped("search_page_drift")
            self.input_seen = True
        elif name in {"click", "send_keys"}:
            for _ in range(20):
                after = await self.snapshot()
                if after[:2] != before[:2] or after[2]["active"] != before[2]["active"]:
                    break
                await asyncio.sleep(0.1)
            else:
                raise SearchStopped("search_page_drift")
            if self.input_seen:
                self.last_submit = before[:2]
        elif name == "navigate":
            after = await self.snapshot()
            if after[0] != self.args["url"]:
                raise SearchStopped("search_page_drift")
            self.initial = after[1]
        self.steps.append(step)


async def _close_search(context, browser, limit):
    async def close(operation, component, seconds):
        task = asyncio.create_task(operation())
        try:
            done, _ = await asyncio.wait({task}, timeout=seconds)
            if not done:
                log.warning("FORGE browser teardown exceeded its deadline: %s", component)
                return "timeout"
            task.result()
        except Exception as exc:
            log.warning("FORGE browser teardown failed: %s %s", component, type(exc).__name__)
            return "failed"
        finally:
            if not task.done():
                task.cancel()
        return "ok"

    # Close the owner first. Browser-Use is a CDP client of this temporary
    # context, and its shutdown events can stall while Chromium is still open.
    context_status = await close(
        context.close, "context", limit / 2 if browser is not None else limit
    )
    browser_status = (
        await close(browser.reset, "engine", limit / 2) if browser is not None else "ok"
    )
    return max((browser_status, context_status), key=("ok", "failed", "timeout").index)


@asynccontextmanager
async def search_browser(worker, args, browser_class, emit):
    """Fresh JavaScript-enabled profile with scoped read-only network access."""
    if not worker.owns_context:
        raise SearchStopped("managed_browser_required")
    foreground = bool(args.get("foreground", True))
    # The host removes the profile after it has reaped the owned worker. A
    # Windows file lock during worker teardown cannot mask the run's result.
    with nullcontext(Path(args["temporary_profile"])) as profile:
        context = browser = None
        started = monotonic()

        def report(stage, status, since):
            emit(
                "forge_stage",
                stage=stage,
                status=status,
                elapsed_ms=int((monotonic() - since) * 1000),
            )

        report("browser_startup", "started", started)
        try:
            async with asyncio.timeout(args.get("startup_seconds", 45)):
                context = await worker.launch_context(
                    str(profile),
                    executable_path=worker.browser_args["executable"],
                    headless=not foreground or sys.platform != "win32",
                    accept_downloads=False,
                    service_workers="block",
                    args=["--remote-debugging-port=0", "--remote-debugging-address=127.0.0.1"],
                )
                page = context.pages[0]
                capture = SearchCapture(page, await context.new_cdp_session(page), args, emit)

                async def route_request(route):
                    request = route.request
                    if (
                        request.method != "GET"
                        or not capture.same_origin(request.url)
                        or not await worker.allowed_destination(request.url)
                    ):
                        await route.abort()
                        return
                    # Fetch without following redirects, since Playwright only routes
                    # the first request in a browser redirect chain.
                    try:
                        headers = {
                            key: value
                            for key, value in request.headers.items()
                            if key.lower() not in {"cookie", "authorization", "proxy-authorization"}
                        }
                        headers["cookie"] = ""
                        response = await route.fetch(headers=headers, max_redirects=0, timeout=8000)
                    except Exception:
                        # Report navigation failures without exposing the URL or request details.
                        if request.is_navigation_request():
                            capture.error = "browser_site_unreachable"
                        await route.abort()  # The failed request is visible to the agent.
                        return
                    try:
                        headers = {
                            key: value
                            for key, value in response.headers.items()
                            if key.lower() not in {"set-cookie", "content-disposition"}
                        }
                        if 300 <= response.status < 400 or (
                            request.is_navigation_request()
                            and (
                                request.frame != page.main_frame
                                or response.status >= 400
                                or "text/html" not in headers.get("content-type", "").lower()
                            )
                        ):
                            if request.is_navigation_request():
                                capture.error = (
                                    "browser_restricted_page"
                                    if response.status in {401, 403, 429}
                                    else "browser_navigation_blocked"
                                )
                            await route.abort()
                        else:
                            await route.fulfill(response=response, headers=headers)
                    finally:
                        await response.dispose()

                await context.route("**/*", route_request)
                await context.route_web_socket("**/*", lambda socket: socket.close())
                async with asyncio.timeout(10):
                    port_file = profile / "DevToolsActivePort"
                    while True:
                        try:
                            port = int(port_file.read_text().splitlines()[0])
                            break
                        except (OSError, ValueError, IndexError):
                            # Chromium may still be publishing the port file during startup.
                            await asyncio.sleep(0.05)
                browser = browser_class(
                    cdp_url=f"http://127.0.0.1:{port}",
                    keep_alive=True,
                    use_cloud=False,
                    enable_default_extensions=False,
                    captcha_solver=False,
                    auto_download_pdfs=False,
                    allowed_domains=[args["origin"]],
                    downloads_path=str(profile / "downloads"),
                )
                await browser.start()
                visibility_started = monotonic()
                visibility = await _visibility(context, capture.session, foreground)
                report("visibility", visibility, visibility_started)
                capture.visibility = visibility
        except TimeoutError as exc:
            report("browser_startup", "timeout", started)
            raise SearchStopped("browser_startup_timeout") from exc
        except asyncio.CancelledError:
            report("browser_startup", "cancelled", started)
            raise
        except Exception as exc:
            report("browser_startup", "failed", started)
            log.warning("FORGE browser context startup failed: %s", type(exc).__name__)
            raise SearchStopped("browser_startup_failed") from None
        else:
            report("browser_startup", "ok", started)
            yield browser, capture, profile, uuid4().hex
        finally:
            if context is not None:
                started = monotonic()
                report("browser_cleanup", "started", started)
                status = await _close_search(context, browser, args.get("cleanup_seconds", 10))
                report("browser_cleanup", status, started)
                if status != "ok":
                    if not sys.exc_info()[0]:
                        raise SearchStopped(f"browser_cleanup_{status}")
