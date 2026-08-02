"""Timers e2e — set by voice (the action decider extracts integer
seconds; there is no server-side duration parsing to unit-test), ring as
a proactive spoken turn, cancel from the UI chip, and ring during listen
mode (timers are the one background event every mode delivers)."""

import util
from util import switch_by_voice


def set_timer_turn(session, fixture="cmd_timer_short", tries=2):
    """Speak a timer request and wait for timer_started; like delegation
    tags, retry with an alternate phrasing when the model confirms without
    emitting the tag (same audio would reproduce the same completion)."""
    for attempt in range(tries):
        name = fixture if attempt == 0 else f"{fixture}_alt"
        t = session.turn(util.audio(name))
        if t.marker == "incomplete":
            t = session.turn({"type": "flush"})
        started = session.wait_for("timer_started", timeout=10)
        if started:
            return t, started
    raise AssertionError(
        f"no timer_started in {tries} tries — last reply {t.text!r}")


def test_timer_set_rings_and_speaks(server, session):
    server.require_managed()
    ack, started = set_timer_turn(session)
    assert 8 <= started["seconds"] <= 12, started
    assert ack.marker == "complete", ack  # the confirmation is spoken
    assert "<" not in ack.text, ack.text  # the tag itself is never spoken
    resolved = session.wait_for("timer_resolved", timeout=30)
    assert resolved and not resolved.get("cancelled"), resolved
    ring = session.collect_turn(60)
    assert ring.marker == "complete", ring  # the ring is SPOKEN
    assert any(w in ring.text.lower()
               for w in ("timer", "done", "up", "ding", "seconds")), ring.text


def test_timer_cancel(server, session):
    server.require_managed()
    _, started = set_timer_turn(session, "cmd_timer_minute")
    session.send({"type": "cancel_timer", "id": started["id"]})
    resolved = session.wait_for("timer_resolved", timeout=10)
    assert resolved and resolved.get("cancelled"), resolved
    # The session keeps working, and no ring ever follows.
    t = session.turn(util.audio("capital_france"))
    assert t.marker == "complete" and "paris" in t.text.lower(), t.text


def test_timer_rings_during_listen_mode(server, session):
    server.require_managed()
    # A minute-long timer: entering listen and one silent turn complete
    # well before it fires, so the ring lands on a quiet floor.
    set_timer_turn(session, "cmd_timer_minute")
    switch_by_voice(session, "cmd_listen", "listen")
    t = session.turn(util.audio("filler_jazz"))
    assert t.marker == "released", t
    resolved = session.wait_for("timer_resolved", timeout=90)
    assert resolved, "timer never rang in listen mode"
    ring = session.collect_turn(60)
    assert ring.marker == "complete", ring  # it spoke, mid-listen
