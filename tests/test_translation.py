"""Translation-mode e2e: voice-command activation, VAD-only segmenting
(no smart-turn holds), translate-don't-answer behavior, and both exits —
the spoken command and the UI escape hatch (set_mode).
"""

import util
from util import switch_by_voice


def test_translation_renders_speech_instead_of_answering(server, session):
    server.require_managed()
    switch_by_voice(session, "cmd_translate", "translate")
    # 'What is the capital of France?' must come back AS the (restated)
    # question — an interpreter renders words. Answering it says 'Paris',
    # which is exactly what must not happen.
    t = session.turn(util.audio("capital_france"))
    assert t.marker == "complete", t
    assert "capital" in t.text.lower()
    assert "paris" not in t.text.lower(), f"answered instead of translating: {t.text!r}"


def test_translation_does_not_hold_incomplete_speech(server, session):
    server.require_managed()
    switch_by_voice(session, "cmd_translate", "translate")
    # This fixture IS held by smart-turn in conversation mode (see
    # test_incomplete_utterance_is_held) — an interpreter must render it
    # on the silence window instead of waiting for a complete thought.
    t = session.turn(util.audio("incomplete_cutoff"))
    assert t.marker == "complete", t
    # The rendering carried the actual words (not just a fallback line):
    # "…I wanted to ask you about…" survives an English→English restate.
    assert "ask" in t.text.lower(), t.text


def test_spoken_command_exits_translation(server, session):
    server.require_managed()
    switch_by_voice(session, "cmd_translate", "translate")
    switch_by_voice(session, "cmd_stop_translate", "conversation")
    # Back to conversation: questions get answers again.
    t = session.turn(util.audio("capital_france"))
    assert t.marker == "complete" and "paris" in t.text.lower()


def test_ui_stop_button_exits_translation(server, session):
    server.require_managed()
    switch_by_voice(session, "cmd_translate", "translate")
    session.send({"type": "set_mode", "mode": "conversation"})
    changed = session.wait_for("mode_changed", timeout=10)
    assert changed and changed.get("mode") == "conversation"
