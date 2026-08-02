"""Elapsed-time awareness: the coarse duration phrase (unit) and the time
note reaching the model after a real gap (e2e — the suite server runs with
TIME_NOTE_MIN_S=15, see conftest)."""

import time

import pytest
import util

from parlor.server import elapsed_phrase


@pytest.mark.parametrize("seconds,phrase", [
    (5, "a few seconds"),
    (20, "about 20 seconds"),
    (44, "about 40 seconds"),
    (60, "about a minute"),
    (89, "about a minute"),
    (130, "about 2 minutes"),
    (600, "about 10 minutes"),
    (3600, "about an hour"),
    (7200, "about 2 hours"),
])
def test_elapsed_phrase_buckets(seconds, phrase):
    assert elapsed_phrase(seconds) == phrase


def test_time_note_reaches_model(server, session):
    server.require_managed()
    t = session.turn(util.audio("capital_france"))
    assert t.marker == "complete", t
    before = server.log().count("Time note:")

    time.sleep(18)
    t = session.turn(util.audio("how_long_quiet"))
    if t.marker == "incomplete":
        # If smart-turn holds the question, the flush turn carries the note
        # instead (the gap is measured to the same quiet either way).
        t = session.turn({"type": "flush"})
    assert t.marker == "complete", t
    assert server.log().count("Time note:") == before + 1
    # Soft check: the reply engages with the gap rather than deflecting.
    assert any(w in t.text.lower()
               for w in ("second", "minute", "moment", "while")), t.text
