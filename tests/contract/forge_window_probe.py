"""Verify a FORGE-owned Chromium window foreground against the Windows desktop."""

from __future__ import annotations

import asyncio
import json
import logging
import sys
from pathlib import Path


async def main(root: Path) -> None:
    sys.path.insert(0, str(root / "jarvis" / "society" / "browser"))
    from dynamic_search import _visibility
    from playwright.async_api import async_playwright

    profiles = Path(sys.argv[2])
    async with async_playwright() as pw:
        unrelated = await pw.chromium.launch_persistent_context(
            str(profiles / "unrelated"), executable_path=sys.argv[1], headless=False
        )
        try:
            owned = await pw.chromium.launch_persistent_context(
                str(profiles / "owned"), executable_path=sys.argv[1], headless=False
            )
            try:
                page = owned.pages[0]
                session = await owned.new_cdp_session(page)
                visibility = await _visibility(owned, session, True)
                await session.detach()
            finally:
                await owned.close()
            unrelated_page = unrelated.pages[0]
            await unrelated_page.set_content("<h1>Unrelated window survived</h1>")
            assert await unrelated_page.locator("h1").text_content() == "Unrelated window survived"
            print(json.dumps({"visibility": visibility, "unrelated": "alive"}))
        finally:
            await unrelated.close()


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    asyncio.run(main(Path(__file__).resolve().parents[2]))
