"""Unit tests for llama-server discovery and the Gemma 4 build floor —
stub binaries on PATH, no real llama.cpp involved."""

import stat

import pytest

from parlor import llama


def _stub(dir, name: str, script: str) -> None:
    path = dir / name
    path.write_text(f"#!/bin/sh\n{script}\n")
    path.chmod(path.stat().st_mode | stat.S_IXUSR)


@pytest.fixture
def on_path(tmp_path, monkeypatch):
    """A bin dir that is the ONLY place binaries resolve from."""
    monkeypatch.setenv("PATH", str(tmp_path))
    return tmp_path


def test_prefers_llama_server(on_path):
    _stub(on_path, "llama-server", "")
    _stub(on_path, "llama", "")
    assert llama.server_command() == [str(on_path / "llama-server")]


def test_falls_back_to_unified_binary(on_path):
    # llama.cpp's first-party installer ships `llama` with a `serve`
    # subcommand and no `llama-server` at all.
    _stub(on_path, "llama", "")
    assert llama.server_command() == [str(on_path / "llama"), "serve"]


def test_missing_binary_says_how_to_install(on_path):
    with pytest.raises(RuntimeError, match="install llama.cpp"):
        llama.server_command()


def test_old_build_is_refused_with_upgrade_hint(on_path):
    _stub(on_path, "llama-server", 'echo "version: 9000 (deadbee)" >&2')
    with pytest.raises(RuntimeError, match="9000 is too old"):
        llama.check_build(llama.server_command(), 9503)


def test_floor_is_per_model(on_path):
    # b9505 carries e2b/e4b (floor 9503) but not the 12b mmproj (9512).
    _stub(on_path, "llama-server", 'echo "version: 9505 (deadbee)" >&2')
    cmd = llama.server_command()
    llama.check_build(cmd, llama.MIN_BUILD)
    with pytest.raises(RuntimeError, match="needs 9512"):
        llama.check_build(cmd, llama.MODEL_MIN_BUILD["12b"])


def test_version_probed_on_root_binary(on_path):
    # `llama serve --version` may not exit — the probe must ask the root
    # binary, and the failure hint must not say brew (wrong installer).
    _stub(on_path, "llama",
          '[ "$1" = "--version" ] && echo "version: 9000 (cafe)" >&2')
    with pytest.raises(RuntimeError, match="llama.app installer"):
        llama.check_build(llama.server_command(), llama.MIN_BUILD)


def test_unparseable_version_fails_open(on_path):
    # Self-built trees print 'version: 0 (unknown)' — the guard is for
    # stale installs, not custom builds.
    _stub(on_path, "llama-server", 'echo "version: 0 (unknown)" >&2')
    llama.check_build(llama.server_command(), llama.MIN_BUILD)
    _stub(on_path, "llama-server", "echo not a version line")
    llama.check_build(llama.server_command(), llama.MIN_BUILD)
