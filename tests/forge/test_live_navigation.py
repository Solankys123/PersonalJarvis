from dataclasses import asdict

import pytest

from jarvis.forge.engine import Refused
from jarvis.forge.live_navigation import Navigation, PageShape


def test_navigation_uses_ephemeral_parameters():
    origin = "http://127.0.0.1:8765"
    first = origin + "/private-path?q=private-value"
    second = origin + "/other?query=second-value"
    shape = PageShape(main=True, heading=True)
    procedure = Navigation.capture(
        allowed_origin=origin,
        requested_url=first,
        final_url=first,
        after=shape,
        succeeded=True,
    )
    assert asdict(procedure) == {"origin": origin, "after": {"main": True, "heading": True}}
    assert procedure.bind(second) == second
    procedure.verify(second, second, shape)
    with pytest.raises(Refused, match="browser_navigation_drift"):
        procedure.verify(second, second, PageShape(main=False, heading=True))
    with pytest.raises(Refused, match="browser_navigation_drift"):
        procedure.verify(second, origin + "/redirect", shape)


@pytest.mark.parametrize(
    "url",
    [
        "https://other.example/",
        "https://example.test:8443/",
        "https://example.test@other.example/",
        "https://user:secret@example.test/",
        "https://example.test\\@other.example/",
        "https://example.test\n/",
        "file:///index.html",
    ],
)
def test_navigation_refuses_invalid_or_other_origins(url):
    procedure = Navigation("https://example.test", PageShape(main=True, heading=True))
    with pytest.raises(Refused):
        procedure.bind(url)


def test_navigation_requires_successful_observed_destination():
    url = "https://example.test/"
    with pytest.raises(Refused, match="navigation_unverified"):
        Navigation.capture(
            allowed_origin="https://example.test",
            requested_url=url,
            final_url=url,
            after=PageShape(main=True, heading=True),
            succeeded=False,
        )
