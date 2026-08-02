"""Just-listen mode e2e: voice-command activation, silent transcribe-only
turns (no replies, no smart-turn holds), the spoken exit — which answers —
and the UI escape hatch."""

import util
from util import switch_by_voice


def test_listen_turns_are_silent(server, session):
    server.require_managed()
    switch_by_voice(session, "cmd_listen", "listen")
    t = session.turn(util.audio("filler_color"))
    assert t.marker == "released", t
    assert t.audio_chunks == 0
    # Transcribed (shown in the UI), just not answered.
    assert "turquoise" in (t.transcription or "").lower(), t.transcription


def test_listen_does_not_hold_incomplete_speech(server, session):
    server.require_managed()
    switch_by_voice(session, "cmd_listen", "listen")
    # A mid-thought cutoff: a scribe transcribes it on the silence window
    # (no smart-turn hold) and stays quiet — the user is still thinking.
    # Deliberately NOT incomplete_cutoff, whose "I wanted to ask you…"
    # reads as addressed to the assistant.
    t = session.turn(util.audio("filler_baking_cutoff"))
    assert t.marker == "released", t
    assert t.audio_chunks == 0


def test_spoken_command_exits_listen_and_conversation_resumes(server, session):
    server.require_managed()
    switch_by_voice(session, "cmd_listen", "listen")
    t = session.turn(util.audio("filler_pet"))
    assert t.marker == "released", t
    # "What do you think about all of that?" is addressed TO the
    # assistant: it must exit AND speak, not get silently transcribed.
    exit_turn = switch_by_voice(session, "cmd_stop_listen", "conversation")
    assert exit_turn.marker == "complete", exit_turn  # the exit turn SPOKE
    # Back to conversation: questions get answers again — the silently
    # stored turns didn't teach the model to stay quiet.
    t = session.turn(util.audio("capital_france"))
    assert t.marker == "complete" and "paris" in t.text.lower(), t.text


def test_ui_stop_button_exits_listen(server, session):
    server.require_managed()
    switch_by_voice(session, "cmd_listen", "listen")
    session.send({"type": "set_mode", "mode": "conversation"})
    changed = session.wait_for("mode_changed", timeout=10)
    assert changed and changed.get("mode") == "conversation"
