"""Bounded public search inside the managed browser worker."""

from __future__ import annotations

import asyncio
from urllib.parse import urlencode, urlsplit
from uuid import uuid4


async def run_search(worker, args, emit):
    if not worker.owns_context:
        raise ValueError("FORGE requires the managed browser")
    start, query, origin = args["url"], args["query"], args["origin"]
    if not start.startswith(origin + "/") or urlsplit(start).netloc != urlsplit(origin).netloc:
        raise ValueError("Invalid search start address")

    context = await worker.context.browser.new_context(
        java_script_enabled=False, service_workers="block", accept_downloads=False
    )
    try:
        result_url = None
        permitted_url = None

        async def route_request(route):
            nonlocal permitted_url
            request = route.request
            if (
                request.method != "GET"
                or request.resource_type != "document"
                or request.url != permitted_url
                or request.frame != page.main_frame
            ):
                await route.abort()
                return
            permitted_url = None
            if not await worker.allowed_destination(request.url):
                await route.abort()
                return
            # An intercepted redirect must never make a second network request.
            response = await route.fetch(max_redirects=0, timeout=8000)
            try:
                if response.status != 200 or not response.headers.get(
                    "content-type", ""
                ).lower().startswith("text/html"):
                    await route.abort()
                else:
                    await route.fulfill(response=response)
            finally:
                await response.dispose()

        await context.route("**/*", route_request)
        page = await context.new_page()
        page.set_default_timeout(5000)

        async def act(name, operation):
            proposal = {"url": origin} if name == "navigate" else {}
            answer = await worker.rpc(
                "action", {"action": {name: proposal}, "url": origin, "confirm": True}
            )
            if not answer.get("ok"):
                raise asyncio.CancelledError("Browser action denied")
            permit = answer["permit"]
            confirmation = asyncio.get_running_loop().create_future()
            worker.pending[permit + ":result"] = confirmation
            try:
                try:
                    await operation()
                    result = {"verified_action": name}
                except Exception:
                    result = {"error": "search_page_drift"}
                emit("action_result", id=permit, result=result)
                async with asyncio.timeout(10):
                    confirmed = await confirmation
                if not confirmed.get("ok"):
                    raise ValueError("Search action failed verification")
            finally:
                worker.pending.pop(permit + ":result", None)

        main = page.locator("main")
        search = main.locator('form input[type="search"][name="q"]')
        button = main.locator('form button[type="submit"]')

        async def form_action():
            if (
                not await main.is_visible()
                or not await main.locator("h1").is_visible()
                or not await search.is_visible()
                or not await button.is_visible()
            ):
                raise ValueError("Search landmarks changed")
            endpoint = await search.evaluate(
                """input => {
                    const form = input.form;
                    const button = form?.querySelector('button[type=submit]');
                    if (!form || !input.closest('main')?.contains(form)
                        || !button || button.form !== form || form.elements.length !== 2
                        || form.method !== 'get'
                        || form.enctype !== 'application/x-www-form-urlencoded'
                        || !['', '_self'].includes(form.target) || input.disabled || input.readOnly
                        || input.value !== '' || input.getAttribute('autocomplete') !== 'off'
                        || button.disabled || button.name
                        || Array.from(button.attributes).some(a => a.name.startsWith('form'))
                        || document.querySelector('base')) return null;
                    return form.action;
                }"""
            )
            if (
                not endpoint
                or not endpoint.startswith(origin + "/")
                or "?" in endpoint
                or "#" in endpoint
            ):
                raise ValueError("Search form changed")
            return endpoint

        async def navigate():
            nonlocal result_url, permitted_url
            permitted_url = start
            await page.goto(start, wait_until="domcontentloaded")
            if page.url != start:
                raise ValueError("Search start changed")
            result_url = (await form_action()) + "?" + urlencode({"q": query})

        async def fill():
            if (
                page.url != start
                or (await form_action()) + "?" + urlencode({"q": query}) != result_url
            ):
                raise ValueError("Search form changed")
            await search.fill(query)
            if await search.input_value() != query:
                raise ValueError("Search input changed")

        async def submit():
            nonlocal permitted_url
            if page.url != start or await search.input_value() != query:
                raise ValueError("Search input changed")
            permitted_url = result_url
            await button.click()
            await page.wait_for_url(result_url, wait_until="domcontentloaded", timeout=8000)

        observed = None

        async def verify():
            nonlocal observed
            if page.url != result_url:
                raise ValueError("Search destination changed")
            heading = main.locator("h1")
            results = main.locator("ul, ol")
            if (
                not await heading.is_visible()
                or query.casefold() not in (await heading.inner_text()).casefold()
                or not await results.is_visible()
            ):
                raise ValueError("Search results changed")
            items = results.locator(":scope > li:visible")
            if not await items.count() or not (await items.first.inner_text()).strip():
                raise ValueError("Search results are empty")
            observed = await results.evaluate("node => node.tagName")
            if args.get("result_list") not in (None, observed):
                raise ValueError("Search result structure changed")

        try:
            for name, operation in (
                ("navigate", navigate),
                ("input_text", fill),
                ("click", submit),
                ("verify_results", verify),
            ):
                await act(name, operation)
        except Exception:
            # Never return page content, addresses, or input in a worker error.
            return {"ok": False, "error": "search_page_drift"}
        return {"ok": True, "target_id": uuid4().hex, "result_list": observed}
    finally:
        await context.close()
