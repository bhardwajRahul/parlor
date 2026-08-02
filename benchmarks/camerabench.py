"""Camera-architecture benchmark: attach the frame every turn (production)
vs fetch it on demand — the "camera as a tool call" question.

Production attaches the client's camera frame to every audio turn and
speculatively prefills it while the user is still talking, so the frame
costs no perceived latency — but it does cost GPU work per frame and
~image-sized context per turn, forever, in history. The alternative is to
treat the camera like a tool: the decoupled action head reports whether
the turn NEEDS the frame, and only then does a request carry it.

    Arch A (attach): production — frame + audio in one request, frame
                     primed during speech. The baseline.
    Arch B (pre):    head judges the audio FIRST (needs_camera field),
                     then the speech request attaches the frame only on
                     "yes". The head's wall time lands BEFORE first audio
                     on EVERY turn — that is its structural price.
    Arch C (post):   production's speak-first shape — the reply streams
                     with no frame, THEN the head decides; on "yes" a
                     follow-up turn with the frame answers for real. Chat
                     turns pay nothing, vision turns pay a second full
                     turn — and the first (blind) reply may hallucinate
                     a scene it never saw, which is judged here too.

Per arch: cam_recall (head wants the frame on vision questions),
cam_misfire (wants it on plain chat), answer_ok (the final spoken reply
names what the fixture scene actually shows), blind_hallucination (C's
first reply invents scene content), and the wall-time components a user
would feel. frame_tokens reports what one JPEG costs in context, from
llama-server's own usage counts.

Measured (e4b, M3 Pro, 2026-08-02 — results/camerabench.json): keep A.
A frame costs 50 context tokens (not the ~300 pipeline.py estimates) and
~600ms of prime GPU hidden under speech; A answers 1.0 at 1.55s median.
B decides perfectly (recall 1.0, misfire 0.0) but its head runs before
speech can start, adding ~2.2s to EVERY turn — vision answers land at
4.2s. C is structurally broken: the blind reply either denies having
eyes or confidently hallucinates ("the round shape is blue", "a picture
of a cat"), the user hears it, and the head then reads that answered
reply as evidence no frame is needed — recall 0.0, answer_ok 0.375. The
speak-first shape that makes timers cheap is exactly what kills
on-demand vision: an unanswerable question never survives to the head.

    uv run python benchmarks/camerabench.py --repeat 2 \
        --out benchmarks/results/camerabench.json

Runs its own llama-server on port 8099 (like archbench); WAVs cache in
fixtures/tagbench/.
"""

import argparse
import http.client
import json
import os
import time
from pathlib import Path

os.environ.setdefault("LLAMA_PORT", "8099")

import fixtures  # noqa: E402
from tagbench import ensure_fixtures  # noqa: E402
from parlor import actions  # noqa: E402
from parlor import llama  # noqa: E402
from parlor import server  # noqa: E402
from parlor.pipeline import audio_part, image_part, text_part  # noqa: E402

# ── cases ─────────────────────────────────────────────────────────────────
# The fixture scene (fixtures.make_image_b64 "day"): blue sky over a green
# field with a red sun-like circle. expected: (needs_camera, answer
# keywords — any-of, checked on the final spoken reply; [] = don't judge).
SCENE_WORDS = ["sky", "field", "sun", "green", "blue", "red", "circle",
               "landscape", "grass"]
CASES = {
    "cam_describe": "Can you describe what you can see right now?",
    "cam_color": "Look at my camera. What color is the round shape?",
    "cam_showing": "What am I showing you right now?",
    "cam_capital": "What is the capital of France?",
    "cam_jazz": "Lately I have been listening to a lot of jazz piano "
                "records in the evening.",
    # Trap: visual language with nothing to look at.
    "cam_sunset_trap": "I saw a beautiful sunset yesterday while walking "
                       "home from the station.",
}
EXPECTED: dict[str, tuple[bool, list[str]]] = {
    "cam_describe": (True, SCENE_WORDS),
    "cam_color": (True, ["red"]),
    "cam_showing": (True, SCENE_WORDS),
    "cam_capital": (False, ["paris"]),
    "cam_jazz": (False, []),
    "cam_sunset_trap": (False, []),
}

SYSTEM = server.SYSTEM_PROMPT + server.CAPABILITY_NOTE + server.RESEARCH_NOTE
RESPOND_CAM = server.RESPOND_PROMPT.format(
    camera=" Mention what you see on their camera if relevant.")
RESPOND_PLAIN = server.RESPOND_PROMPT.format(camera="")

# The production head prompt + schema, extended with one field: does this
# turn need the camera? Bench-local — this is exactly what would move into
# actions.py if an on-demand arch won.
CAM_CLAUSE = (
    " camera: true ONLY if answering requires looking at the user's "
    "camera right now — they ask what you see, or refer to something "
    "they are currently showing, holding, or pointing at; talking ABOUT "
    "sights or scenery is not showing you anything, so camera is false."
)
HEAD_PROMPT = actions._HEAD_COMMON.format(
    mode_clause=actions._MODE_CLAUSES["conversation"]) + CAM_CLAUSE
HEAD_SCHEMA = {
    **actions.HEAD_SCHEMA,
    "properties": {**actions.HEAD_SCHEMA["properties"],
                   "camera": {"type": "boolean"}},
    "required": actions.HEAD_SCHEMA["required"] + ["camera"],
}


def chat(messages: list, *, max_tokens: int = 256,
         temperature: float | None = None,
         json_schema: dict | None = None) -> tuple[str, int]:
    """Direct llama-server call; returns (content, prompt_tokens)."""
    body: dict = {"messages": messages, "max_tokens": max_tokens,
                  "cache_prompt": True,
                  "chat_template_kwargs": {"enable_thinking": False}}
    body["temperature"] = llama.TEMPERATURE if temperature is None else temperature
    if json_schema:
        body["response_format"] = {"type": "json_schema",
                                   "json_schema": {"schema": json_schema}}
    conn = http.client.HTTPConnection(*llama.host_port(), timeout=300)
    conn.request("POST", "/v1/chat/completions", json.dumps(body),
                 {"Content-Type": "application/json"})
    data = json.loads(conn.getresponse().read())
    conn.close()
    if "error" in data:
        raise RuntimeError(f"llama-server: {data['error']}")
    return (data["choices"][0]["message"].get("content") or "",
            (data.get("usage") or {}).get("prompt_tokens") or 0)


def prime(messages: list) -> int:
    """Production's cache priming (runs during user speech, so its wall
    time is GPU cost, not perceived latency). Returns its wall ms."""
    t0 = time.time()
    chat(messages, max_tokens=1)
    return round((time.time() - t0) * 1000)


def run_head(prefix: list) -> tuple[dict, int]:
    t0 = time.time()
    raw, _ = chat(prefix + [{"role": "user", "content": HEAD_PROMPT}],
                  max_tokens=64, temperature=0.0, json_schema=HEAD_SCHEMA)
    ms = round((time.time() - t0) * 1000)
    try:
        return json.loads(raw), ms
    except ValueError:
        return {}, ms


def spoken(reply: str) -> str:
    """Everything after the transcript line — what TTS would say."""
    return reply.split("\n", 1)[1] if "\n" in reply else reply


# Each runner returns (wants_camera, answer_text, blind_text|None, extra).
# answer_text is the reply the user ultimately hears for the question;
# blind_text is C's first no-frame reply on turns the head then escalated.

def run_arch_a(wav: str, img: str) -> tuple[bool, str, None, dict]:
    user = [image_part(img), audio_part(wav), text_part(RESPOND_CAM)]
    prime_ms = prime([{"role": "system", "content": SYSTEM},
                      {"role": "user", "content": [image_part(img), audio_part(wav)]}])
    t0 = time.time()
    reply, pt = chat([{"role": "system", "content": SYSTEM},
                      {"role": "user", "content": user}])
    return True, spoken(reply), None, {
        "turn_ms": round((time.time() - t0) * 1000),
        "prime_ms": prime_ms, "prompt_tokens": pt, "raw": reply}


def run_arch_b(wav: str, img: str) -> tuple[bool, str, None, dict]:
    base = [{"role": "system", "content": SYSTEM}]
    prime_ms = prime(base + [{"role": "user", "content": [audio_part(wav)]}])
    # decide_before shape: head prompt rides the same user message as the
    # audio, extending the primed prefix.
    t0 = time.time()
    raw, _ = chat(base + [{"role": "user",
                           "content": [audio_part(wav), text_part(HEAD_PROMPT)]}],
                  max_tokens=64, temperature=0.0, json_schema=HEAD_SCHEMA)
    head_ms = round((time.time() - t0) * 1000)
    try:
        wants = bool(json.loads(raw).get("camera"))
    except ValueError:
        wants = False
    content = ([image_part(img), audio_part(wav), text_part(RESPOND_CAM)]
               if wants else [audio_part(wav), text_part(RESPOND_PLAIN)])
    t0 = time.time()
    reply, pt = chat(base + [{"role": "user", "content": content}])
    return wants, spoken(reply), None, {
        "head_ms": head_ms, "turn_ms": round((time.time() - t0) * 1000),
        "prime_ms": prime_ms, "prompt_tokens": pt, "raw": reply}


def run_arch_c(wav: str, img: str) -> tuple[bool, str, str | None, dict]:
    base = [{"role": "system", "content": SYSTEM}]
    prime_ms = prime(base + [{"role": "user", "content": [audio_part(wav)]}])
    speech = base + [{"role": "user",
                      "content": [audio_part(wav), text_part(RESPOND_PLAIN)]}]
    t0 = time.time()
    reply, pt = chat(speech)
    turn_ms = round((time.time() - t0) * 1000)
    prefix = speech + [{"role": "assistant", "content": reply}]
    head, head_ms = run_head(prefix)
    wants = bool(head.get("camera"))
    extra = {"turn_ms": turn_ms, "head_ms": head_ms, "prime_ms": prime_ms,
             "prompt_tokens": pt, "raw": reply}
    if not wants:
        return False, spoken(reply), None, extra
    t0 = time.time()
    followup = ("System note (not user audio): here is the user's camera "
                "frame. Now answer their question, looking at it. 1-4 "
                "short sentences, spoken aloud.")
    reply2, pt2 = chat(prefix + [{"role": "user",
                                  "content": [image_part(img), text_part(followup)]}])
    extra.update({"turn2_ms": round((time.time() - t0) * 1000),
                  "prompt_tokens2": pt2, "raw2": reply2})
    return True, spoken(reply2), spoken(reply), extra


RUNNERS = {"A_attach": run_arch_a, "B_pre": run_arch_b, "C_post": run_arch_c}


def judge(name: str, wants: bool, answer: str, blind: str | None) -> dict:
    needs, keywords = EXPECTED[name]
    answer_l = answer.lower()
    return {
        "case": name, "needs_camera": needs, "wants_camera": wants,
        "cam_hit": wants == needs,
        "answer_ok": (any(k in answer_l for k in keywords)
                      if keywords else None),
        # C only: the blind first reply claims scene content it never saw.
        "blind_hallucination": (any(k in blind.lower() for k in SCENE_WORDS)
                               if blind is not None else None),
        "blind": blind,
    }


def measure_frame_tokens(wav: str, img: str) -> int:
    """What one JPEG frame costs in context, from llama's own counts."""
    _, with_img = chat([{"role": "system", "content": SYSTEM},
                        {"role": "user", "content": [image_part(img), audio_part(wav),
                                                     text_part(RESPOND_CAM)]}],
                       max_tokens=1)
    _, without = chat([{"role": "system", "content": SYSTEM},
                       {"role": "user", "content": [audio_part(wav),
                                                    text_part(RESPOND_CAM)]}],
                      max_tokens=1)
    return with_img - without


def score(results: list[dict]) -> dict:
    pos = [r for r in results if r["needs_camera"]]
    neg = [r for r in results if not r["needs_camera"]]
    judged = [r for r in results if r["answer_ok"] is not None]
    out = {
        "cam_recall": round(sum(r["wants_camera"] for r in pos) / len(pos), 3),
        "cam_misfire": round(sum(r["wants_camera"] for r in neg) / len(neg), 3),
        "answer_ok": round(sum(r["answer_ok"] for r in judged) / len(judged), 3),
        "n": len(results),
    }
    blind = [r for r in results if r["blind_hallucination"] is not None]
    if blind:
        out["blind_hallucination"] = round(
            sum(r["blind_hallucination"] for r in blind) / len(blind), 3)
    for key in ("turn_ms", "head_ms", "turn2_ms", "prime_ms"):
        vals = sorted(r[key] for r in results if key in r)
        if vals:
            out[f"{key}_p50"] = vals[len(vals) // 2]
    # What the user waits on a VISION turn, per arch: A = turn (frame was
    # primed); B = head + turn; C = turn1 + head + turn2.
    vis = [r for r in results if r["needs_camera"]]
    waits = [r.get("head_ms", 0) + r.get("turn_ms", 0) + r.get("turn2_ms", 0)
             for r in vis]
    if waits:
        waits.sort()
        out["vision_wait_ms_p50"] = waits[len(waits) // 2]
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repeat", type=int, default=2)
    ap.add_argument("--archs", default="A_attach,B_pre,C_post",
                    help=f"comma-separated subset of {', '.join(RUNNERS)}")
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    wavs = ensure_fixtures(CASES)
    img = fixtures.make_image_b64(scene="day")
    llama.start()
    try:
        out = {"model": llama.MODEL, "speech_temperature": llama.TEMPERATURE,
               "repeat": args.repeat,
               "frame_tokens": measure_frame_tokens(wavs["cam_capital"], img),
               "archs": {}}
        print(f"One frame costs {out['frame_tokens']} context tokens\n")
        for label in args.archs.split(","):
            runner = RUNNERS[label]
            results = []
            for name, wav in wavs.items():
                for _ in range(args.repeat):
                    t0 = time.time()
                    wants, answer, blind, extra = runner(wav, img)
                    r = judge(name, wants, answer, blind)
                    r.update(extra)
                    results.append(r)
                    ok = "✓" if (r["cam_hit"] and r["answer_ok"] is not False) else "✗"
                    print(f"{ok} [{label}] {name}: cam={wants} "
                          f"({time.time() - t0:.1f}s"
                          f"{', head ' + str(r['head_ms']) + 'ms' if 'head_ms' in r else ''})")
            stats = score(results)
            out["archs"][label] = {"stats": stats, "results": results}
            print(f"\n{label}: {stats}\n")
    finally:
        llama.stop()

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(out, indent=2))
        print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
