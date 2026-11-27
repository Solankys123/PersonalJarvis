from dataclasses import asdict

import pytest

from jarvis.forge.engine import Evidence, Refused, State, Verdict
from jarvis.forge.live_navigation import DynamicSearch, Navigation, PageShape
from jarvis.forge.store import Store


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


def test_promoted_dynamic_search_lookup_after_reopen(tmp_path):
    path = tmp_path / "forge.sqlite"
    store = Store(path)
    origin = "https://example.test"
    steps = ("navigate", "input_text:" + "a" * 64, "click:" + "b" * 64, "verify_results")
    spec = DynamicSearch(origin, steps)

    def evidence(target, length):
        return [
            Evidence(
                target,
                step,
                f"length:{length}",
                State.BLANK,
                State.MATCH,
                Verdict.SUCCESS,
                "verified",
                index,
            )
            for index, step in enumerate(steps, 1)
        ]

    assert store.record("learn", "learn", None, Verdict.SUCCESS, evidence("first", 6), spec) == (
        1,
        Verdict.SUCCESS,
    )
    assert store.promoted_search(origin) is None
    assert store.record(
        "validate", "validate", 1, Verdict.SUCCESS, evidence("second", 4), spec
    ) == (
        1,
        Verdict.SUCCESS,
    )
    store.promote(1, authorized=True)
    reopened = Store(path)
    assert reopened.promoted_search(origin) == 1
    assert reopened.promoted_search("https://other.test") is None
    assert reopened.get(1) == (spec, "promoted")
