"""Bounded, content-free evidence for the live browser learner."""

from __future__ import annotations

from dataclasses import dataclass
from urllib.parse import urlsplit

from .engine import Refused


@dataclass(frozen=True)
class PageShape:
    """Presence of semantic landmarks, with no names, values or page text."""

    main: bool
    heading: bool

    def __post_init__(self) -> None:
        if type(self.main) is not bool or type(self.heading) is not bool:
            raise Refused("invalid_page_shape")


@dataclass(frozen=True)
class Navigation:
    """A same-origin navigation with its destination supplied at runtime."""

    origin: str
    after: PageShape

    def __post_init__(self) -> None:
        if self.origin != _origin(self.origin):
            raise Refused("invalid_browser_origin")
        if not isinstance(self.after, PageShape) or not (self.after.main and self.after.heading):
            raise Refused("missing_page_landmarks")

    @classmethod
    def capture(
        cls,
        *,
        allowed_origin: str,
        requested_url: str,
        final_url: str,
        after: PageShape,
        succeeded: bool,
    ) -> Navigation:
        if succeeded is not True or requested_url != final_url:
            raise Refused("navigation_unverified")
        procedure = cls(allowed_origin, after)
        procedure.bind(requested_url)
        return procedure

    def bind(self, url: str) -> str:
        if _origin(url) != self.origin:
            raise Refused("browser_origin_mismatch")
        return url

    def verify(self, requested_url: str, final_url: str, after: PageShape) -> None:
        self.bind(requested_url)
        if final_url != requested_url or after != self.after:
            raise Refused("browser_navigation_drift")


def _origin(url: str) -> str:
    if (
        not isinstance(url, str)
        or not 0 < len(url) <= 2048
        or any(ord(char) <= 32 or ord(char) >= 127 for char in url)
        or "\\" in url
    ):
        raise Refused("invalid_browser_url")
    try:
        parts = urlsplit(url)
        host, port = parts.hostname, parts.port
    except ValueError:
        raise Refused("invalid_browser_url") from None
    if (
        parts.scheme not in ("http", "https")
        or not host
        or parts.username is not None
        or parts.password is not None
        or "%" in host
    ):
        raise Refused("invalid_browser_url")
    authority = f"[{host}]" if ":" in host else host
    default_port = 443 if parts.scheme == "https" else 80
    if port is not None and port != default_port:
        authority += f":{port}"
    return f"{parts.scheme}://{authority}"
