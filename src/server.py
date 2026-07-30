"""Parlor — on-device, real-time multimodal AI (voice + vision).

LLM inference runs on llama.cpp (llama-server, spawned as a subprocess —
see llama.py). This server owns the conversation history and re-sends it
every request; llama-server's prefix cache makes that cheap, and it also
enables two speculative tricks (see pipeline.py): the camera frame and the
user's speech (in ~3s chunks) are pushed through cache-priming requests
WHILE the user is still talking, so the final request only pays for the
tail of the utterance.
"""

import asyncio
import json
import os
import sys
import time
import traceback
from contextlib import asynccontextmanager
from pathlib import Path

sys.stdout.reconfigure(line_buffering=True)  # logs stream even when piped

import numpy as np
import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

try:  # raised by ws.send_text after the client goes away (uvicorn-specific)
    from uvicorn.protocols.utils import ClientDisconnected
except ImportError:  # pragma: no cover
    class ClientDisconnected(OSError):
        pass

DISCONNECT_ERRORS = (WebSocketDisconnect, ClientDisconnected)

from dotenv import load_dotenv
load_dotenv()  # before importing llama — it reads its config at import time

import llama
import tts
from pipeline import (estimate_tokens, pad_tail_silence, prime_cache,
                      release_client, run_turn, send_json, text_part,
                      user_content, valid_audio, wav_to_float32)

# Turn completeness is judged by the smart-turn audio classifier before the
# LLM is involved, so the prompt carries no FINISHED/WAIT machinery at all.
# Asking Gemma to judge it instead scores at chance on audio — see
# benchmarks/turnbench.py, which still reproduces those two variants.
SYSTEM_PROMPT = (
    "You are a friendly, conversational AI assistant. The user talks to you "
    "through a microphone and may show you their camera. Your replies are "
    "spoken aloud, so write plain conversational text without formatting."
)

# The transcript line LEADS the reply: transcribing after the response
# turns the transcript into a paraphrase from memory (WER 0.39 vs 0.00 on a
# clean 33-word utterance), and the leading line reaches the client while
# the response still decodes. Costs its decode time (~0.2s short / ~0.7s
# long utterances) before first audio — measured worth it. Grammar-forced
# JSON ({transcript, response}) was also measured: format breaks 1-3/3 on
# degraded audio and 3/3 on chunked — don't go back to structured output.
RESPOND_PROMPT = (
    "Begin your reply with one line: ###TRANSCRIPT: followed by the exact "
    "words the user said in their audio message. Then, on a new line, respond "
    "to them: 1-4 short sentences, spoken aloud.{camera}"
)

# A "flush" turn: the classifier judged the utterance unfinished, the user
# then stayed silent, so answer what we have — the model decides whether it
# is answerable or needs encouragement to continue. Dedicated prompt: with
# a bolted-on suffix the model would sometimes emit the transcript line and
# stop, leaving the turn silent.
FLUSH_PROMPT = (
    "Begin your reply with one line: ###TRANSCRIPT: followed by the exact "
    "words the user said in their audio message. The user paused mid-thought, "
    "so on a new line: if their words feel unfinished, write one short, warm "
    "sentence encouraging them to continue; otherwise respond to them in 1-4 "
    "short sentences, spoken aloud.{camera}"
)

# Rotate history before the llama context fills. Rough token estimates are
# fine here — the guard just needs to fire before generation degrades.
CONTEXT_HEADROOM = 2000

tts_backend = None
detector = None  # smart-turn end-of-turn classifier


def load_models():
    global tts_backend, detector
    llama.start()
    from turn_detector import TurnDetector
    detector = TurnDetector()
    tts_backend = tts.load()


@asynccontextmanager
async def lifespan(app):
    await asyncio.get_event_loop().run_in_executor(None, load_models)
    yield
    llama.stop()


app = FastAPI(lifespan=lifespan)
app.mount("/static", StaticFiles(directory=Path(__file__).parent / "static"), name="static")


@app.get("/")
async def root():
    return HTMLResponse(content=(Path(__file__).parent / "index.html").read_text())


def turn_instruction(msg: dict, has_image: bool, has_audio: bool) -> str:
    if has_audio:
        camera = " Mention what you see on their camera if relevant." if has_image else ""
        prompt = FLUSH_PROMPT if msg.get("type") == "flush" else RESPOND_PROMPT
        return prompt.format(camera=camera)
    if has_image:
        return "The user is showing you their camera. Describe what you see."
    return msg.get("text", "Hello!")


@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    await ws.accept()

    history: list = [{"role": "system", "content": SYSTEM_PROMPT}]

    interrupted = asyncio.Event()
    active = {"stream": None}
    msg_queue = asyncio.Queue()

    async def receiver():
        try:
            while True:
                raw = await ws.receive_text()
                msg = json.loads(raw)
                if msg.get("type") == "interrupt":
                    interrupted.set()
                    stream = active.get("stream")
                    if stream:
                        stream.cancel()  # actually aborts generation
                    print("Client interrupted")
                else:
                    await msg_queue.put(msg)
        except DISCONNECT_ERRORS:
            pass
        finally:
            # Always unblock the main loop, even on unexpected errors.
            await msg_queue.put(None)

    recv_task = asyncio.create_task(receiver())

    frame_image: str | None = None   # camera frame held for the current utterance
    speech_chunks: list[str] = []    # streamed-in speech, already cache-primed
    held_audio: list[str] = []       # incomplete-turn segments awaiting continuation

    async def prime(audio_b64s: list[str]) -> None:
        """Warm the cache for the turn as it stands so far — reads the live
        history and held camera frame, so it must stay in this scope."""
        await prime_cache(history + [
            {"role": "user", "content": user_content(frame_image, audio_b64s)}])

    try:
        while True:
            msg = await msg_queue.get()
            if msg is None:
                break

            # Rotate history before the llama context fills: keep the system
            # prompt and the most recent exchanges.
            if estimate_tokens(history) > llama.CTX - CONTEXT_HEADROOM:
                keep = 1 + max(2, (len(history) - 1) // 2)
                print(f"Context near limit — dropping {len(history) - keep} oldest messages")
                history = [history[0]] + history[-(keep - 1):]

            if msg.get("type") == "frame":
                if msg.get("image"):
                    frame_image = msg["image"]
                    speech_chunks = []
                    await prime(held_audio)
                continue

            if msg.get("type") == "speech_chunk":
                if msg.get("seq") == 0:
                    speech_chunks = []
                if valid_audio(msg.get("audio")):
                    speech_chunks.append(msg["audio"])
                    await prime(held_audio + speech_chunks)
                continue

            interrupted.clear()
            audio_b64s = held_audio + (speech_chunks if msg.get("chunked") else [])
            speech_chunks = []
            if valid_audio(msg.get("audio")):
                audio_b64s.append(msg["audio"])
            image = msg.get("image") or frame_image
            has_audio = bool(audio_b64s)

            if not has_audio and not image and not msg.get("text"):
                # Mic glitch (or a flush with nothing held) produced no usable media.
                await release_client(ws)
                continue

            # From here on any failure (malformed WAV included — valid_audio
            # only checks length) must release the client, not kill the
            # session loop.
            try:
                # The audio classifier judges completeness before the LLM is
                # involved at all. Incomplete → hold the segments (they stay
                # in the next turn's content AND warm in the cache) and wait.
                # A flush turn skips the check: the client waited out the
                # hold and now wants an answer to whatever we have.
                if has_audio and msg.get("type") != "flush":
                    pcm = np.concatenate([wav_to_float32(b) for b in audio_b64s])
                    t0 = time.time()
                    complete, prob = await asyncio.get_event_loop().run_in_executor(
                        None, detector.predict, pcm)
                    decision_s = round(time.time() - t0, 3)
                    if not complete and not interrupted.is_set():
                        held_audio = audio_b64s
                        await prime(held_audio)
                        await send_json(ws, {
                            "type": "turn_incomplete",
                            "decision_s": decision_s, "p_complete": round(prob, 2),
                        })
                        continue

                # Padding the final segment diverges it from its primed bytes
                # on flush turns (≤3s re-prefilled) — the honest-transcript
                # win beats that; the continuation path keeps its cache hits.
                if audio_b64s:
                    audio_b64s[-1] = pad_tail_silence(audio_b64s[-1])
                content = user_content(image, audio_b64s)
                frame_image = None
                held_audio = []

                instruction = turn_instruction(msg, bool(image), has_audio)
                user_msg = {"role": "user", "content": content + [text_part(instruction)]}
                raw_text = await run_turn(ws, history + [user_msg], interrupted, active,
                                          tts_backend, expect_transcript=has_audio)
                # Store the turn verbatim (same bytes → full prefix-cache hit
                # on the next request). Never store a turn the model produced
                # nothing for — a degenerate message poisons all later requests.
                if raw_text.strip():
                    history.append(user_msg)
                    history.append({"role": "assistant", "content": raw_text})
            except DISCONNECT_ERRORS:
                raise
            except Exception:
                traceback.print_exc()  # keep the session alive
                await release_client(ws)
    except DISCONNECT_ERRORS:
        print("Client disconnected")
    finally:
        recv_task.cancel()


if __name__ == "__main__":
    port = int(os.environ.get("PORT", "8000"))
    uvicorn.run(app, host="0.0.0.0", port=port)
