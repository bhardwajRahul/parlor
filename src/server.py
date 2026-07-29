"""Parlor — on-device, real-time multimodal AI (voice + vision)."""

import asyncio
import base64
import json
import os
import re
import time
import traceback
from contextlib import asynccontextmanager
from pathlib import Path

import litert_lm
import numpy as np
import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse

try:  # raised by ws.send_text after the client goes away (uvicorn-specific)
    from uvicorn.protocols.utils import ClientDisconnected
except ImportError:  # pragma: no cover
    class ClientDisconnected(OSError):
        pass

DISCONNECT_ERRORS = (WebSocketDisconnect, ClientDisconnected)

import tts

from dotenv import load_dotenv
load_dotenv()

HF_REPO = "litert-community/gemma-4-E2B-it-litert-lm"
HF_FILENAME = "gemma-4-E2B-it.litertlm"


def resolve_model_path() -> str:
    path = os.environ.get("MODEL_PATH", "")
    if path:
        return path
    from huggingface_hub import hf_hub_download
    print(f"Downloading {HF_REPO}/{HF_FILENAME} (first run only)...")
    return hf_hub_download(repo_id=HF_REPO, filename=HF_FILENAME)


MODEL_PATH = resolve_model_path()

# The reply streams straight into TTS, so the format is: response first
# (sentences are spoken as they decode), transcript last (it never delays
# first audio). Incomplete turns cost only a couple of decode tokens.
SYSTEM_PROMPT = (
    "You are a friendly, conversational AI assistant. The user talks to you "
    "through a microphone and may show you their camera. Your reply is spoken "
    "aloud, so write plain conversational text: 1-4 short sentences, no formatting.\n"
    "\n"
    "Your reply MUST start with exactly one of these words on its own line, "
    "judging the user's speech:\n"
    "- FINISHED — the user completed their thought. Continue with your spoken "
    "response on the next line.\n"
    "- WAIT — the user has not finished: they were cut off mid-sentence or are "
    "pausing to think. Say nothing else and let them continue.\n"
    "\n"
    "If the user sent audio, end your reply with a new line:\n"
    "###TRANSCRIPT: the exact words the user said\n"
    "\n"
    "Examples:\n"
    'User audio: "What\'s your favorite color?"\n'
    "You: FINISHED\n"
    "I really like deep blue. What about you?\n"
    "###TRANSCRIPT: What's your favorite color?\n"
    "\n"
    'User audio with camera image: "Is this too much salt?"\n'
    "You: FINISHED\n"
    "That looks like a good amount for a pot that size.\n"
    "###TRANSCRIPT: Is this too much salt?\n"
    "\n"
    'User audio: "So the thing I wanted to say is"\n'
    "You: WAIT\n"
    "\n"
    'User audio: "Hmm, let me think about that for a second."\n'
    "You: WAIT\n"
)

NUDGE_PROMPT = (
    "(The user went quiet without finishing their thought. Reply FINISHED, then "
    "in one short, warm sentence encourage them to continue. No transcript line.)"
)

# Sent as a tiny standalone turn the moment the user STARTS speaking, so the
# ~274 image tokens are already in the KV cache when the audio arrives —
# cuts camera-turn time-to-first-token by ~75%.
FRAME_PROMPT = (
    "This is the user's current camera view; they are about to speak. "
    "Reply with only the word OK."
)

# Accept close marker variants ("FINISH", trailing colon, markdown wrap) —
# they never appear as the first word of a real response in uppercase.
MARKER_RE = re.compile(r"[\s*_]*(FINISHED|FINISH|WAITING|WAIT)\b[:.]?[\s*_]*")
MARKER_KINDS = {
    "FINISHED": "complete",
    "FINISH": "complete",
    "WAIT": "incomplete_short",
    "WAITING": "incomplete_short",
}
MARKER_HOLDBACK = 10  # chars of non-marker text before we give up waiting
TRANSCRIPT_TAG = "###TRANSCRIPT:"
SENTENCE_END_RE = re.compile(r"[.!?]+\s")
MAX_OUTPUT_TOKENS = 256

_DONE = object()

engine = None
tts_backend = None


def _audio_backend():
    """AUDIO_BACKEND=gpu|cpu, AUDIO_THREADS=<n> (0 = library default).
    Note: the current gemma-4-E2B .litertlm requires cpu for audio."""
    if os.environ.get("AUDIO_BACKEND", "cpu").lower() == "gpu":
        return litert_lm.Backend.GPU()
    threads = int(os.environ.get("AUDIO_THREADS", "0"))
    return litert_lm.Backend.CPU(thread_count=threads or None)


def load_models():
    global engine, tts_backend
    print(f"Loading Gemma 4 E2B from {MODEL_PATH}...")
    engine = litert_lm.Engine(
        MODEL_PATH,
        backend=litert_lm.Backend.GPU(),
        vision_backend=litert_lm.Backend.GPU(),
        audio_backend=_audio_backend(),
    )
    engine.__enter__()
    print("Engine loaded.")

    tts_backend = tts.load()


@asynccontextmanager
async def lifespan(app):
    await asyncio.get_event_loop().run_in_executor(None, load_models)
    yield


app = FastAPI(lifespan=lifespan)


@app.get("/")
async def root():
    return HTMLResponse(content=(Path(__file__).parent / "index.html").read_text())


class StreamParser:
    """Incrementally parses 'FINISHED\\n<response>\\n###TRANSCRIPT: <words>',
    where the first line may instead be a lone WAIT turn marker.

    feed() returns complete response sentences as they become available;
    finalize() returns the trailing partial sentence and the transcript.
    """

    # Hold back enough of the tail to never TTS a partially-arrived tag.
    TAG_HOLDBACK = len(TRANSCRIPT_TAG) + 2

    def __init__(self):
        self.kind = None  # complete / incomplete_short / incomplete_long
        self.response = ""
        self.transcript = ""
        self._pending = ""  # text held until marker-vs-response is decided
        self._in_transcript = False
        self._emitted = 0

    def _decide_kind(self, final: bool) -> tuple[str, str]:
        """Returns (kind, response remainder) — kind '' means keep holding."""
        m = MARKER_RE.match(self._pending)
        if m:
            if not final and m.end() == len(self._pending):
                # "FINISH" may still grow into "FINISHED" — deciding now would
                # leave the trailing "ED" to be spoken as response text.
                return "", ""
            return MARKER_KINDS[m.group(1)], self._pending[m.end():]
        stripped = self._pending.lstrip()
        if not final and len(stripped) < MARKER_HOLDBACK:
            return "", ""
        # Model skipped the marker — treat everything as the response.
        return "complete", stripped

    def feed(self, delta: str, final: bool = False) -> list[str]:
        if self.kind is None:
            self._pending += delta
            kind, remainder = self._decide_kind(final)
            if not kind:
                return []
            self.kind = kind
            if kind != "complete":
                return []
            delta, self._pending = remainder.lstrip(), ""
        elif self.kind != "complete":
            return []  # marker turn: ignore any trailing text

        if self._in_transcript:
            self.transcript += delta
            return []

        self.response += delta
        tag_pos = self.response.find(TRANSCRIPT_TAG)
        if tag_pos != -1:
            self.transcript = self.response[tag_pos + len(TRANSCRIPT_TAG):]
            self.response = self.response[:tag_pos]
            self._in_transcript = True
        return self._complete_sentences()

    def _complete_sentences(self) -> list[str]:
        end = len(self.response)
        if not self._in_transcript:
            end = max(self._emitted, end - self.TAG_HOLDBACK)
        # A malformed transcript tag (e.g. missing colon) never matches
        # TRANSCRIPT_TAG — make sure we still never speak past a "###".
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
        sentences = self.feed("", final=True)
        # Cut any (possibly truncated) transcript tag — never speak it.
        tail = re.split(r"#{2,}", self.response[self._emitted:])[0].strip()
        transcript = self.transcript.strip() or None
        return sentences + ([tail] if tail else []), transcript


async def run_frame_turn(conversation, image_b64: str):
    """Prefill the camera frame while the user is still speaking."""
    t0 = time.time()

    def produce():
        for _ in conversation.send_message_async(
            {"role": "user", "content": [
                {"type": "image", "blob": image_b64},
                {"type": "text", "text": FRAME_PROMPT},
            ]},
            max_output_tokens=4,
        ):
            pass

    try:
        await asyncio.get_event_loop().run_in_executor(None, produce)
        print(f"Frame prefilled ({time.time() - t0:.2f}s)")
        return True
    except Exception as e:
        print(f"Frame prefill failed: {e}")
        return False


async def run_turn(ws: WebSocket, conversation, content: list, interrupted: asyncio.Event):
    """Stream one model turn: decode → sentences → TTS, all pipelined."""
    loop = asyncio.get_event_loop()
    t0 = time.time()
    timings = {}

    chunk_q: asyncio.Queue = asyncio.Queue()

    def produce():
        try:
            for chunk in conversation.send_message_async(
                {"role": "user", "content": content},
                max_output_tokens=MAX_OUTPUT_TOKENS,
            ):
                text = "".join(
                    p.get("text", "") for p in chunk.get("content", [])
                    if isinstance(p, dict)
                )
                if text:
                    loop.call_soon_threadsafe(chunk_q.put_nowait, text)
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

        if parser.kind in ("incomplete_short", "incomplete_long"):
            kind = parser.kind.removeprefix("incomplete_")
            print(f"LLM ({timings['llm_time']:.2f}s) turn incomplete ({kind})")
            if not interrupted.is_set():
                await ws.send_text(json.dumps({"type": "turn_incomplete", "kind": kind, **timings}))
            return

        if not interrupted.is_set():
            await dispatch(tail)
    finally:
        sentence_q.put_nowait(_DONE)
        try:
            await tts_task
        finally:
            # The decode thread writes into `conversation` — it must be done
            # before the caller can ever tear the conversation down.
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
        return

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


def build_content(msg: dict, frame_prefilled: bool = False) -> list:
    content = []
    if msg.get("audio"):
        content.append({"type": "audio", "blob": msg["audio"]})
    if msg.get("image"):
        content.append({"type": "image", "blob": msg["image"]})

    if msg.get("type") == "nudge":
        content.append({"type": "text", "text": NUDGE_PROMPT})
    elif msg.get("audio") and (msg.get("image") or frame_prefilled):
        content.append({"type": "text", "text": "The user just spoke while showing their camera. Start with FINISHED or WAIT, then respond to what they said, referencing what you see if relevant. End with the ###TRANSCRIPT line."})
    elif msg.get("audio"):
        content.append({"type": "text", "text": "The user just spoke to you. Start with FINISHED or WAIT, then respond to what they said. End with the ###TRANSCRIPT line."})
    elif msg.get("image"):
        content.append({"type": "text", "text": "The user is showing you their camera. Describe what you see."})
    else:
        content.append({"type": "text", "text": msg.get("text", "Hello!")})
    return content


@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    await ws.accept()

    conversation = engine.create_conversation(
        messages=[{"role": "system", "content": SYSTEM_PROMPT}],
    )
    conversation.__enter__()

    interrupted = asyncio.Event()
    msg_queue = asyncio.Queue()

    async def receiver():
        try:
            while True:
                raw = await ws.receive_text()
                msg = json.loads(raw)
                if msg.get("type") == "interrupt":
                    # Suppress output but let decode run out: cancel_process()
                    # permanently wedges the conversation in litert-lm 0.14
                    # (the next send_message never returns). Responses are
                    # short, so the extra busy window is <1s.
                    interrupted.set()
                    print("Client interrupted")
                else:
                    await msg_queue.put(msg)
        except DISCONNECT_ERRORS:
            pass
        finally:
            # Always unblock the main loop, even on unexpected errors.
            await msg_queue.put(None)

    recv_task = asyncio.create_task(receiver())

    frame_prefilled = False
    try:
        while True:
            msg = await msg_queue.get()
            if msg is None:
                break
            if msg.get("type") == "frame":
                # Always accept a fresh frame (the client sends at most one
                # per utterance; a re-send replaces a stale one).
                if msg.get("image"):
                    if await run_frame_turn(conversation, msg["image"]):
                        frame_prefilled = True
                    else:
                        await ws.send_text(json.dumps({"type": "frame_failed"}))
                continue
            interrupted.clear()
            try:
                await run_turn(ws, conversation, build_content(msg, frame_prefilled), interrupted)
            except DISCONNECT_ERRORS:
                raise
            except Exception:
                # Keep the session alive and release the client from
                # its 'processing' state.
                traceback.print_exc()
                await ws.send_text(json.dumps({
                    "type": "turn_final", "transcription": None,
                    "timings": {}, "spoke": False,
                }))
            finally:
                frame_prefilled = False
    except DISCONNECT_ERRORS:
        print("Client disconnected")
    finally:
        recv_task.cancel()
        conversation.__exit__(None, None, None)


if __name__ == "__main__":
    port = int(os.environ.get("PORT", "8000"))
    uvicorn.run(app, host="0.0.0.0", port=port)
