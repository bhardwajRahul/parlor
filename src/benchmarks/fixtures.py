"""Speech fixtures for the e2e benchmark.

Synthesizes real spoken audio with the local Kokoro TTS backend (so the
Gemma audio encoder gets actual speech, not sine waves), resamples to the
16 kHz the frontend sends, and caches WAVs in benchmarks/fixtures/.

Run directly to (re)generate:  uv run python benchmarks/fixtures.py
"""

import base64
import io
import sys
import wave
from pathlib import Path

import numpy as np

FIXTURES_DIR = Path(__file__).parent / "fixtures"
TARGET_SR = 16000

# name -> (spoken text, keywords expected in the transcript, expected turn kind)
FIXTURES = {
    "capital_france": (
        "What is the capital of France?",
        ["capital", "france"],
        "complete",
    ),
    "long_question": (
        "I have been trying to learn English for a few months now, and I "
        "wonder if you could give me some advice on how to improve my "
        "pronunciation when I speak with other people.",
        ["english", "pronunciation"],
        "complete",
    ),
    # Truncated mid-utterance (see generate_all) so the audio cuts off
    # abruptly, like a real interrupted speaker — TTS of a trailing-off
    # sentence otherwise sounds politely finished.
    "incomplete_cutoff": (
        "So the thing I wanted to ask you about is the weather in Paris "
        "for my trip next week.",
        [],
        "incomplete_short",
    ),
    "thinking_pause": (
        "That's a hard question. Hmm, let me think about it for a moment.",
        [],
        "incomplete_long",
    ),
    "describe_scene": (
        "Can you describe what you can see right now?",
        ["see"],
        "complete",
    ),
}


def _resample_linear(pcm: np.ndarray, src_sr: int, dst_sr: int) -> np.ndarray:
    if src_sr == dst_sr:
        return pcm
    duration = len(pcm) / src_sr
    n_dst = int(duration * dst_sr)
    x_src = np.linspace(0.0, duration, num=len(pcm), endpoint=False)
    x_dst = np.linspace(0.0, duration, num=n_dst, endpoint=False)
    return np.interp(x_dst, x_src, pcm).astype(np.float32)


def _write_wav(path: Path, pcm: np.ndarray, sr: int) -> None:
    pcm_int16 = (pcm * 32767).clip(-32768, 32767).astype(np.int16)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(pcm_int16.tobytes())


def generate_all(force: bool = False) -> None:
    missing = [n for n in FIXTURES if force or not (FIXTURES_DIR / f"{n}.wav").exists()]
    if not missing:
        return
    FIXTURES_DIR.mkdir(exist_ok=True)
    sys.path.insert(0, str(Path(__file__).parent.parent))
    import tts

    backend = tts.load()
    for name in missing:
        text = FIXTURES[name][0]
        pcm = backend.generate(text, speed=1.0)
        pcm = _resample_linear(np.asarray(pcm, dtype=np.float32), backend.sample_rate, TARGET_SR)
        if name == "incomplete_cutoff":
            pcm = pcm[: int(len(pcm) * 0.55)]  # chop mid-utterance
            fade = min(len(pcm), int(0.01 * TARGET_SR))
            pcm[-fade:] *= np.linspace(1.0, 0.0, fade)
        _write_wav(FIXTURES_DIR / f"{name}.wav", pcm, TARGET_SR)
        print(f"fixture {name}: {len(pcm) / TARGET_SR:.1f}s speech")


def load_wav_b64(name: str) -> str:
    return base64.b64encode((FIXTURES_DIR / f"{name}.wav").read_bytes()).decode()


def make_image_b64(width: int = 320, height: int = 240) -> str:
    """A scene the model can plausibly describe: blue sky over a green field
    with a red circle (sun-like) — drawn with PIL, JPEG-encoded like the frontend."""
    from PIL import Image, ImageDraw

    img = Image.new("RGB", (width, height), color=(90, 160, 220))
    draw = ImageDraw.Draw(img)
    draw.rectangle([0, height * 2 // 3, width, height], fill=(70, 150, 70))
    draw.ellipse([width - 110, 20, width - 40, 90], fill=(220, 60, 50))
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=70)
    return base64.b64encode(buf.getvalue()).decode()


if __name__ == "__main__":
    generate_all(force="--force" in sys.argv)
    print("Fixtures ready:", ", ".join(FIXTURES))
