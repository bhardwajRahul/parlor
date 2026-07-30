"""E2E suite server management.

By default each pytest session spawns its own server (real llama.cpp, TTS,
turn detector) on a dedicated port with a small context window so history
rotation is cheap to trigger, and captures its log for assertions. Set
PARLOR_TEST_URL=ws://host:port/ws to test an already-running server instead
(log- and process-based tests skip).
"""

import os
import subprocess
import sys
import time
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(SRC / "benchmarks"))

import fixtures  # noqa: E402

TEST_PORT = 8821
TEST_LLAMA_PORT = 8822
STARTUP_TIMEOUT_S = 300


@dataclass
class Server:
    url: str
    proc: subprocess.Popen | None = None
    log_path: Path | None = None

    def log(self) -> str:
        return self.log_path.read_text() if self.log_path else ""

    def require_managed(self) -> None:
        if self.proc is None:
            pytest.skip("needs a suite-managed server (unset PARLOR_TEST_URL)")


@pytest.fixture(scope="session")
def server(tmp_path_factory) -> Server:
    fixtures.generate_all()

    external = os.environ.get("PARLOR_TEST_URL")
    if external:
        yield Server(url=external)
        return

    log_path = tmp_path_factory.mktemp("server") / "server.log"
    env = {**os.environ, "PORT": str(TEST_PORT), "LLAMA_PORT": str(TEST_LLAMA_PORT),
           "LLAMA_CTX": "4096"}
    env.pop("LLAMA_SERVER_URL", None)
    with open(log_path, "w") as log:
        proc = subprocess.Popen([sys.executable, "server.py"], cwd=SRC, env=env,
                                stdout=log, stderr=subprocess.STDOUT)

    deadline = time.time() + STARTUP_TIMEOUT_S
    while True:
        if proc.poll() is not None:
            pytest.fail(f"server exited during startup:\n{log_path.read_text()[-3000:]}")
        try:
            urllib.request.urlopen(f"http://127.0.0.1:{TEST_PORT}/", timeout=2)
            break
        except OSError:
            if time.time() > deadline:
                proc.terminate()
                pytest.fail(f"server not ready in {STARTUP_TIMEOUT_S}s:\n"
                            f"{log_path.read_text()[-3000:]}")
            time.sleep(2)

    yield Server(url=f"ws://127.0.0.1:{TEST_PORT}/ws", proc=proc, log_path=log_path)

    proc.terminate()  # lifespan shutdown terminates the spawned llama-server
    try:
        proc.wait(timeout=15)
    except subprocess.TimeoutExpired:
        proc.kill()


@pytest.fixture
def session(server):
    from util import Session

    with Session(server.url) as s:
        yield s
