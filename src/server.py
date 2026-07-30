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
import itertools
import json
import os
import sys
import time
import traceback
from concurrent.futures import ThreadPoolExecutor
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
import reasoner
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
    "spoken aloud, so write plain conversational text without formatting. "
    "If an audio message is just your own previous reply playing back "
    "(echo), don't answer it — briefly ask what they'd like to talk about."
)

# Appended to the system prompt only when a reasoner endpoint is
# configured (.env REASONER_*). An XML element, not a ###-style line:
# benchmarks/tagbench.py measured much higher tag recall for it on E4B.
DELEGATE_INSTRUCTION = (
    " You also have a background research assistant with web access. When "
    "the user asks you to search, look up, find, or research something, or "
    "asks about anything current or changing (weather, news, prices, "
    "scores, openings, \"right now\", \"today\"), you MUST hand the task "
    "over instead of answering from memory — your knowledge is stale and a "
    "guess is worse than handing over. To hand over: say one short "
    "sentence telling the user you're on it, then append <delegate>the "
    "task, restated to stand alone</delegate> — never speak or mention "
    "that tag; the result arrives later and you can share it then. "
    "Everything else, answer yourself and don't use the tag."
)

# A finished background task is delivered by the voice model, not read out
# raw: it ties the answer back to the conversation and keeps one voice.
DELIVER_PROMPT = (
    'Your background research assistant just finished the task "{task}". '
    "Its answer:\n{answer}\n\n"
    "Deliver this answer to the user now: tie it back to what they asked "
    "earlier, keep every fact exactly as given, 2-5 short sentences, "
    "plain spoken text."
)
DELIVER_FAILED_PROMPT = (
    'Your background research assistant could not finish the task '
    '"{task}". Briefly tell the user you couldn\'t get that answer, and '
    "that they're welcome to ask again. Don't start new research now."
)
DELIVER_FALLBACK = "Sorry — I couldn't get an answer for that one."

# One session can only usefully consume a few pending research tasks, and
# an off-the-rails reply must not queue an HTTP call per imagined tag.
MAX_PENDING_DELEGATIONS = 3

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

# Spoken when a turn yields no reply at all (models of every size
# occasionally emit only the transcript line) — silence would leave the
# user hanging, and a stored transcript-only reply teaches the model to
# do it again next turn.
FLUSH_FALLBACK = "Take your time — I'm listening."
AUDIO_FALLBACK = "Sorry, I didn't catch that — could you say it again?"

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

# Reasoner calls block for up to REASONER_TIMEOUT — keep them off the
# default executor, which serves the latency-critical path (llama
# streaming, TTS, the turn classifier, cache priming).
REASONER_POOL = ThreadPoolExecutor(max_workers=4, thread_name_prefix="reasoner")


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
    html = (Path(__file__).parent / "index.html").read_text()
    return HTMLResponse(content=html.replace("{{model}}", llama.model_label()))


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

    delegation = reasoner.enabled()
    system = SYSTEM_PROMPT + (DELEGATE_INSTRUCTION if delegation else "")
    control_tags = ("delegate",) if delegation else ()
    history: list = [{"role": "system", "content": system}]

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

    def remember(user_msg: dict, raw_text: str) -> None:
        """Store a finished turn verbatim (same bytes → full prefix-cache hit
        on the next request). A turn the model produced nothing for is never
        stored — a degenerate message poisons all later requests."""
        if raw_text.strip():
            history.append(user_msg)
            history.append({"role": "assistant", "content": raw_text})

    delegation_ids = itertools.count(1)
    delegation_tasks: set[asyncio.Task] = set()
    ready_delegations: list[dict] = []  # finished while the floor was busy

    def floor_busy() -> bool:
        """The user holds the floor: audio held for a continuation, a
        mid-utterance chunk stream, or a just-fired barge-in. A delegation
        result may only be delivered when this is false."""
        return bool(held_audio or speech_chunks or interrupted.is_set())

    def drain_ready() -> None:
        """Requeue one finished delegation whenever the floor frees up —
        called at every point that releases it, because msg_queue only
        wakes for client traffic and a result must not wait for one."""
        if ready_delegations and not floor_busy():
            msg_queue.put_nowait(ready_delegations.pop(0))

    async def run_delegation(task_id: int, task: str) -> None:
        """Background reasoner call; its outcome re-enters the main loop
        through msg_queue so delivery is serialized with real turns."""
        try:
            answer = await asyncio.get_event_loop().run_in_executor(
                REASONER_POOL, reasoner.ask, task)
            # Clamp a verbose answer: it is interpolated into the delivery
            # prompt and stored in history under a small LLAMA_CTX.
            if len(answer) > 1500:
                answer = answer[:1500].rsplit(". ", 1)[0] + "."
            outcome = {"ok": True, "answer": answer}
        except Exception as e:
            print(f"Delegation #{task_id} failed: {e}")
            outcome = {"ok": False, "answer": ""}
        await msg_queue.put({"type": "delegation_done", "id": task_id,
                             "task": task, **outcome})

    async def spawn_delegations(tags: list[tuple[str, str]]) -> None:
        for name, value in tags:
            if name == "DELEGATE" and value.strip():
                if len(delegation_tasks) >= MAX_PENDING_DELEGATIONS:
                    print(f"Delegation skipped (cap {MAX_PENDING_DELEGATIONS}): {value!r}")
                    continue
                task_id = next(delegation_ids)
                print(f"Delegation #{task_id}: {value!r}")
                await send_json(ws, {"type": "delegation_started",
                                     "id": task_id, "task": value})
                t = asyncio.create_task(run_delegation(task_id, value))
                delegation_tasks.add(t)
                t.add_done_callback(delegation_tasks.discard)

    async def deliver_delegation(done: dict) -> None:
        """A server-initiated turn: the result goes into history and the
        voice model speaks it. Failures are delivered too — a delegation
        must never end in silence, which is also why the fallback is the
        reasoner's own answer: if the model's relay yields nothing
        speakable (##-markup relapse, transcript-only reply), TTS speaks
        the answer directly."""
        interrupted.clear()
        await send_json(ws, {"type": "delegation_resolved",
                             "id": done["id"], "ok": done["ok"]})
        prompt = (DELIVER_PROMPT if done["ok"] else DELIVER_FAILED_PROMPT).format(
            task=done["task"], answer=done["answer"])
        user_msg = {"role": "user", "content": [text_part(prompt)]}
        # expect_transcript=True although there is no audio: history trains
        # the model to open every reply with ###TRANSCRIPT:, and it often
        # does so here too (echoing the instruction). The transcript parser
        # consumes that line and streams the real delivery; with False the
        # ##-markup cut would swallow the entire reply (observed live).
        raw_text, tags = await run_turn(ws, history + [user_msg], interrupted,
                                        active, tts_backend,
                                        expect_transcript=True,
                                        control_tags=control_tags,
                                        proactive=True,
                                        fallback=done["answer"] if done["ok"]
                                        else DELIVER_FALLBACK)
        remember(user_msg, raw_text)
        await spawn_delegations(tags)  # a delivery may chain deeper research
        drain_ready()                  # more results may already be waiting

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

            if msg.get("type") == "ready":
                # The client returned to idle listening (e.g. after a false
                # barge-in that never became an utterance). A sticky
                # interrupted flag must not strand queued deliveries — and
                # the client cleared its own frame-discard flag before
                # sending this, so delivering now is safe.
                interrupted.clear()
                drain_ready()
                continue

            if msg.get("type") == "delegation_done":
                if floor_busy():  # deliver at the next idle moment instead
                    ready_delegations.append(msg)
                    continue
                try:
                    await deliver_delegation(msg)
                except DISCONNECT_ERRORS:
                    raise
                except Exception:
                    traceback.print_exc()  # keep the session alive
                    if not msg.get("redelivered"):
                        msg["redelivered"] = True  # one more try at idle
                        ready_delegations.append(msg)
                    await release_client(ws)
                continue

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
            is_flush = msg.get("type") == "flush"

            if not has_audio and not image and not msg.get("text"):
                # Mic glitch (or a flush with nothing held) produced no usable media.
                await release_client(ws)
                drain_ready()  # the floor is provably free here
                continue

            # From here on any failure (malformed WAV included — valid_audio
            # only checks length) must release the client, not kill the
            # session loop.
            p_complete = None  # smart-turn probability, surfaced in the UI
            try:
                # The audio classifier judges completeness before the LLM is
                # involved at all. Incomplete → hold the segments (they stay
                # in the next turn's content AND warm in the cache) and wait.
                # A flush turn skips the check: the client waited out the
                # hold and now wants an answer to whatever we have.
                if has_audio and not is_flush:
                    pcm = np.concatenate([wav_to_float32(b) for b in audio_b64s])
                    t0 = time.time()
                    complete, prob = await asyncio.get_event_loop().run_in_executor(
                        None, detector.predict, pcm)
                    decision_s = round(time.time() - t0, 3)
                    p_complete = round(prob, 2)
                    if not complete and not interrupted.is_set():
                        held_audio = audio_b64s
                        # Release the client BEFORE the (slow) cache priming:
                        # until turn_incomplete arrives it can't capture a
                        # resumed utterance, and the flush timer's live-speech
                        # guard can't see the user talking.
                        await send_json(ws, {
                            "type": "turn_incomplete",
                            "decision_s": decision_s, "p_complete": p_complete,
                        })
                        await prime(held_audio)
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
                raw_text, tags = await run_turn(ws, history + [user_msg], interrupted, active,
                                                tts_backend, expect_transcript=has_audio,
                                                p_complete=p_complete,
                                                control_tags=control_tags,
                                                fallback=FLUSH_FALLBACK if is_flush
                                                else AUDIO_FALLBACK if has_audio else None)
                remember(user_msg, raw_text)
                await spawn_delegations(tags)
            except DISCONNECT_ERRORS:
                raise
            except Exception:
                traceback.print_exc()  # keep the session alive
                await release_client(ws)

            # A result that finished while the floor was busy delivers now.
            drain_ready()
    except DISCONNECT_ERRORS:
        print("Client disconnected")
    finally:
        recv_task.cancel()
        for t in delegation_tasks:
            t.cancel()


if __name__ == "__main__":
    port = int(os.environ.get("PORT", "8000"))
    uvicorn.run(app, host="0.0.0.0", port=port)
