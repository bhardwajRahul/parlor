"""The streaming turn pipeline: message-content helpers, the incremental
response/transcript parser, and run_turn (decode → sentences → TTS, all
pipelined), plus speculative cache priming."""

import asyncio
import base64
import io
import json
import re
import time
import wave

import numpy as np

import llama

TRANSCRIPT_TAG = "###TRANSCRIPT:"
# The model takes liberties with the tag ("### TRANSCRIPT: ..."), so parse
# it tolerantly — a dropped transcript looks like a broken turn to the user.
TRANSCRIPT_TAG_RE = re.compile(r"#{2,}\s*TRANSCRIPT:?\s*", re.IGNORECASE)
SENTENCE_END_RE = re.compile(r"[.!?]+\s")
MAX_OUTPUT_TOKENS = 256

# Appended (inside the final WAV) before the LLM sees the utterance: audio
# that stops abruptly at the VAD cutoff makes the encoder hallucinate a
# confident completion of the last word; a beat of silence fixes it.
TAIL_SILENCE_S = 0.3

# Rough per-part token costs for the context-rotation estimate.
AUDIO_TOKENS_PER_SEC = 32
IMAGE_TOKENS = 300

_DONE = object()


# ── message content ───────────────────────────────────────────────────────

def image_part(b64: str) -> dict:
    return {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64," + b64}}


def audio_part(b64: str) -> dict:
    return {"type": "input_audio", "input_audio": {"data": b64, "format": "wav"}}


def text_part(text: str) -> dict:
    return {"type": "text", "text": text}


def valid_audio(b64: str | None) -> bool:
    """At least ~100ms of 16kHz s16 WAV — llama-server 400s on empty audio,
    and one bad message in history would poison every later request."""
    return bool(b64) and len(b64) * 3 // 4 > 44 + 3200


def user_content(image_b64: str | None, audio_b64s: list[str]) -> list:
    """Media parts of the current user turn, in canonical (cache-stable)
    order: image first, then audio segments oldest-to-newest."""
    parts = [image_part(image_b64)] if image_b64 else []
    parts += [audio_part(b) for b in audio_b64s]
    return parts


def estimate_tokens(messages: list) -> int:
    total = 0
    for m in messages:
        content = m.get("content")
        if isinstance(content, str):
            total += len(content) // 4 + 8
            continue
        for p in content:
            if p["type"] == "text":
                total += len(p["text"]) // 4
            elif p["type"] == "input_audio":
                wav_bytes = len(p["input_audio"]["data"]) * 3 // 4
                total += (wav_bytes // 32000) * AUDIO_TOKENS_PER_SEC  # 16kHz s16
            else:
                total += IMAGE_TOKENS
        total += 8
    return total


def wav_to_float32(b64: str) -> np.ndarray:
    with wave.open(io.BytesIO(base64.b64decode(b64)), "rb") as w:
        pcm = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16)
    return pcm.astype(np.float32) / 32768.0


def pad_tail_silence(b64: str) -> str:
    """Append TAIL_SILENCE_S of silence inside the WAV. Must be in the same
    WAV as the speech — a separate silence part doesn't stop the encoder
    hallucinating a completion of an abruptly-cut last word."""
    with wave.open(io.BytesIO(base64.b64decode(b64)), "rb") as w:
        params = w.getparams()
        frames = w.readframes(w.getnframes())
    frames += b"\x00" * (2 * int(TAIL_SILENCE_S * params.framerate))
    buf = io.BytesIO()
    with wave.open(buf, "wb") as out:
        out.setparams(params)
        out.writeframes(frames)
    return base64.b64encode(buf.getvalue()).decode()


# ── streaming turn parser ─────────────────────────────────────────────────

class StreamParser:
    """Incrementally parses '<response>\\n###TRANSCRIPT: <words>'.

    feed() returns complete response sentences as they become available;
    finalize() returns the trailing partial sentence and the transcript.
    """

    # Hold back enough of the tail to never TTS a partially-arrived tag.
    TAG_HOLDBACK = len(TRANSCRIPT_TAG) + 4

    def __init__(self):
        self.response = ""
        self.transcript = ""
        self._in_transcript = False
        self._emitted = 0

    def feed(self, delta: str) -> list[str]:
        if self._in_transcript:
            self.transcript += delta
            return []

        self.response += delta
        m = TRANSCRIPT_TAG_RE.search(self.response)
        if m:
            self.transcript = self.response[m.end():]
            self.response = self.response[:m.start()]
            self._in_transcript = True
        return self._complete_sentences()

    def _complete_sentences(self) -> list[str]:
        end = len(self.response)
        if not self._in_transcript:
            end = max(self._emitted, end - self.TAG_HOLDBACK)
        # A malformed tag can slip past TRANSCRIPT_TAG_RE — still never
        # speak anything from a "###" onwards.
        hash_pos = self.response.find("###", self._emitted)
        if hash_pos != -1:
            end = min(end, hash_pos)
        sentences = []
        while True:
            m = SENTENCE_END_RE.search(self.response, self._emitted, end)
            if not m:
                break
            sentence = self.response[self._emitted:m.end()].strip()
            self._emitted = m.end()
            if sentence:
                sentences.append(sentence)
        return sentences

    def finalize(self) -> tuple[list[str], str | None]:
        sentences = self.feed("")
        # Cut any (possibly truncated) transcript tag — never speak it.
        tail = re.split(r"#{2,}", self.response[self._emitted:])[0].strip()
        transcript = self.transcript.strip() or None
        return sentences + ([tail] if tail else []), transcript


# ── turn execution ────────────────────────────────────────────────────────

async def run_turn(ws, messages: list, interrupted: asyncio.Event,
                   active: dict, tts_backend) -> str:
    """Stream one model turn: decode → sentences → TTS, all pipelined.
    Returns the raw generated text (stored verbatim in history so the next
    request gets a full prefix-cache hit)."""
    loop = asyncio.get_event_loop()
    t0 = time.time()
    timings: dict = {}

    chunk_q: asyncio.Queue = asyncio.Queue()
    stream = llama.ChatStream(messages, MAX_OUTPUT_TOKENS)
    active["stream"] = stream
    raw = {"text": ""}

    def produce():
        try:
            def on_delta(text):
                raw["text"] += text
                loop.call_soon_threadsafe(chunk_q.put_nowait, text)
            stream.run(on_delta)
            loop.call_soon_threadsafe(chunk_q.put_nowait, _DONE)
        except Exception as e:  # surfaced to the consumer loop
            loop.call_soon_threadsafe(chunk_q.put_nowait, e)

    producer = loop.run_in_executor(None, produce)

    sentence_q: asyncio.Queue = asyncio.Queue()
    audio_state = {"started": False, "first_audio_at": None, "chunks": 0}

    async def tts_worker():
        while True:
            sentence = await sentence_q.get()
            if sentence is _DONE:
                return
            if interrupted.is_set():
                continue  # keep draining
            pcm = await loop.run_in_executor(None, lambda s=sentence: tts_backend.generate(s))
            if interrupted.is_set():
                continue
            if not audio_state["started"]:
                audio_state["started"] = True
                audio_state["first_audio_at"] = time.time()
                await ws.send_text(json.dumps({
                    "type": "audio_start",
                    "sample_rate": tts_backend.sample_rate,
                }))
            pcm_int16 = (pcm * 32767).clip(-32768, 32767).astype(np.int16)
            await ws.send_text(json.dumps({
                "type": "audio_chunk",
                "audio": base64.b64encode(pcm_int16.tobytes()).decode(),
                "index": audio_state["chunks"],
            }))
            audio_state["chunks"] += 1

    tts_task = asyncio.create_task(tts_worker())
    parser = StreamParser()
    tts_started_at = None

    async def dispatch(sentences: list[str]):
        nonlocal tts_started_at
        for sentence in sentences:
            if tts_started_at is None:
                tts_started_at = time.time()
            await ws.send_text(json.dumps({"type": "text_delta", "text": sentence + " "}))
            sentence_q.put_nowait(sentence)

    try:
        while True:
            item = await chunk_q.get()
            if item is _DONE:
                break
            if isinstance(item, Exception):
                raise item
            if "prefill_s" not in timings:
                timings["prefill_s"] = round(time.time() - t0, 3)
            if not interrupted.is_set():
                await dispatch(parser.feed(item))

        tail, transcript = parser.finalize()
        timings["llm_time"] = round(time.time() - t0, 3)
        timings["decode_s"] = round(timings["llm_time"] - timings.get("prefill_s", 0), 3)

        if not interrupted.is_set():
            await dispatch(tail)
    finally:
        active["stream"] = None
        sentence_q.put_nowait(_DONE)
        try:
            await tts_task
        finally:
            # The producer thread must be done before anyone reuses the slot.
            await producer

    if audio_state["first_audio_at"]:
        timings["ttfa_s"] = round(audio_state["first_audio_at"] - t0, 3)
    if tts_started_at:
        timings["tts_time"] = round(time.time() - tts_started_at, 3)

    print(
        f"LLM ({timings['llm_time']:.2f}s, prefill {timings.get('prefill_s')}s) "
        f"heard: {transcript!r} → {parser.response.strip()!r}"
    )

    if interrupted.is_set():
        print("Interrupted mid-turn")
        return raw["text"]

    await ws.send_text(json.dumps({
        "type": "turn_final",
        "transcription": transcript,
        "timings": timings,
        "spoke": audio_state["started"],  # False → client must not wait for audio_end
    }))
    if audio_state["started"]:
        await ws.send_text(json.dumps({
            "type": "audio_end",
            "tts_time": timings.get("tts_time", 0),
        }))
    return raw["text"]


async def prime_cache(messages: list):
    """Fire-and-discard request that pushes a prompt prefix (camera frame,
    speech chunks) through llama-server's cache while the user is talking.
    Content must be media-only appends — a trailing text block would diverge
    the prefix and kill reuse."""
    t0 = time.time()
    try:
        await asyncio.get_event_loop().run_in_executor(
            None, lambda: llama.chat_blocking(messages, max_tokens=1))
        print(f"Primed cache ({time.time() - t0:.2f}s)")
        return True
    except Exception as e:
        print(f"Cache priming failed: {e}")
        return False
