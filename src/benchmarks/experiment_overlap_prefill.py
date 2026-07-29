"""Experiment: can we prefill the static part of a turn while the user is still speaking?

Premise: a turn carrying a camera frame costs ~1.9s end-to-end today, and the frame is
worth ~0.5s of that. The frontend knows speech has STARTED 1-3s before it knows speech
has ENDED, so anything not derived from the audio could be prefilled in that window,
leaving only audio + decode on the critical path once the user stops talking.

Two mechanisms are tested:

  * Conversation-level (variants A/D/E) — send the camera frame as its own conversation
    turn at speech start, so the next turn only pays for audio. Public API only.
  * Session-level (variant B) — litert_lm.Engine.create_session() + incremental
    run_prefill() calls into one KV cache. Text only; the multimodal path is checked
    and its failure recorded.

Controls matter here: a second turn is cheaper than a first turn for reasons that have
nothing to do with the image, so every overlap variant has a no-image twin, and
conversation.token_count is recorded per turn to show what each turn actually prefilled.

Run:  uv run python benchmarks/experiment_overlap_prefill.py
"""

import argparse
import ctypes
import json
import os
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))

import litert_lm

import fixtures

HF_REPO = "litert-community/gemma-4-E2B-it-litert-lm"
HF_FILENAME = "gemma-4-E2B-it.litertlm"

RUNS = 3
MAX_OUTPUT_TOKENS = 60
PREFILL_TURN_MAX_TOKENS = 4
AUDIO_FIXTURE = "capital_france"  # "What is the capital of France?" -> expect "Paris"
EXPECT_KEYWORD = "paris"

# A stand-in for server.py's SYSTEM_PROMPT — copied rather than imported because
# importing server.py loads the model and the TTS backend as a side effect.
SYSTEM_PROMPT = (
    "You are a friendly, conversational AI assistant. The user talks to you "
    "through a microphone and may show you their camera. Your reply is spoken "
    "aloud, so write plain conversational text: 1-4 short sentences, no formatting.\n"
    "\n"
    "Your reply MUST start with exactly one of these words on its own line, "
    "judging the user's speech:\n"
    "- FINISHED - the user completed their thought. Continue with your spoken "
    "response on the next line.\n"
    "- WAIT - the user has not finished: they were cut off mid-sentence or are "
    "pausing to think. Say nothing else and let them continue.\n"
    "\n"
    "If the user sent audio, end your reply with a new line:\n"
    "###TRANSCRIPT: the exact words the user said\n"
)

TURN_TEXT_WITH_IMAGE = (
    "The user just spoke while showing their camera. Start with FINISHED or WAIT, "
    "then respond to what they said, referencing what you see if relevant. "
    "End with the ###TRANSCRIPT line."
)
TURN_TEXT_AUDIO_ONLY = (
    "The user just spoke to you. Start with FINISHED or WAIT, then respond to what "
    "they said. End with the ###TRANSCRIPT line."
)
# Sent at speech START, when the audio does not exist yet.
FRAME_TURN_TEXT = (
    "This is the user's current camera view. They are about to speak. "
    "Reply with only the word OK."
)
# The no-image control for the frame turn: same shape, same decode, no picture.
CONTROL_TURN_TEXT = (
    "The user is about to speak. Reply with only the word OK."
)

GROUNDING_QUESTION = (
    "What color is the round shape in the upper right of the image? "
    "Answer with a single word."
)
GROUNDING_EXPECTED = "red"

GREEDY = litert_lm.SamplerConfig(top_k=1, top_p=1.0, temperature=0.0, seed=0)


def log(*a):
    print(*a, flush=True)


def resolve_model_path() -> str:
    path = os.environ.get("MODEL_PATH", "")
    if path:
        return path
    from huggingface_hub import hf_hub_download
    return hf_hub_download(repo_id=HF_REPO, filename=HF_FILENAME)


# --- visual token budget: exported by the C library, absent from the Python wrapper ---

def bind_visual_token_budget(lib) -> bool:
    fn = getattr(lib, "litert_lm_conversation_optional_args_set_visual_token_budget", None)
    if fn is None:
        return False
    fn.argtypes = [ctypes.c_void_p, ctypes.c_int]
    fn.restype = None
    return True


def set_visual_budget(conv, budget: int | None):
    """Attach a visual_token_budget to every send_message on this conversation."""
    if budget is None:
        return
    lib = conv._lib

    def build(max_output_tokens):
        ptr = lib.litert_lm_conversation_optional_args_create()
        if max_output_tokens is not None:
            lib.litert_lm_conversation_optional_args_set_max_output_tokens(ptr, max_output_tokens)
        lib.litert_lm_conversation_optional_args_set_visual_token_budget(ptr, budget)
        return ptr

    conv._create_optional_args = build


# --- measurement helpers ---

def send_timed(conv, content: list, max_tokens: int = MAX_OUTPUT_TOKENS) -> dict:
    """Stream one turn; return latency to first token, to completion, and tokens added."""
    before = conv.token_count
    t0 = time.perf_counter()
    ttft = None
    parts = []
    for chunk in conv.send_message_async({"role": "user", "content": content},
                                         max_output_tokens=max_tokens):
        text = "".join(p.get("text", "") for p in chunk.get("content", [])
                       if isinstance(p, dict))
        if text and ttft is None:
            ttft = time.perf_counter() - t0
        parts.append(text)
    total = time.perf_counter() - t0
    return {
        "ttft": ttft,
        "total": total,
        "text": "".join(parts),
        "tokens_before": before,
        "tokens_after": conv.token_count,
    }


def new_conv(engine, budget=None):
    conv = engine.create_conversation(
        messages=[{"role": "system", "content": SYSTEM_PROMPT}],
        sampler_config=GREEDY,
    )
    conv.__enter__()
    set_visual_budget(conv, budget)
    return conv


def summarize(runs: list[dict]) -> dict:
    ttfts = [r["ttft"] for r in runs if r["ttft"] is not None]
    last = runs[-1]
    out = {
        "ttft_mean": round(statistics.mean(ttfts), 3) if ttfts else None,
        "ttft_stdev": round(statistics.stdev(ttfts), 3) if len(ttfts) > 1 else 0.0,
        "total_mean": round(statistics.mean([r["total"] for r in runs]), 3),
        "critical_tokens": last.get("tokens_after", 0) - last.get("tokens_before", 0),
        "context_tokens": last.get("tokens_after"),
        "sample_text": last["text"][:300],
    }
    offline = [r["offline_s"] for r in runs if "offline_s" in r]
    if offline:
        out["offline_mean"] = round(statistics.mean(offline), 3)
        out["offline_tokens"] = last.get("offline_tokens")
    return out


# --- variants -------------------------------------------------------------

def v_monolithic(engine, wav_b64, img_b64, *, with_image: bool, budget=None) -> dict:
    """A / E: everything in one turn on a fresh conversation — what server.py does today."""
    conv = new_conv(engine, budget)
    try:
        content = [{"type": "audio", "blob": wav_b64}]
        if with_image:
            content.append({"type": "image", "blob": img_b64})
        content.append({"type": "text",
                        "text": TURN_TEXT_WITH_IMAGE if with_image else TURN_TEXT_AUDIO_ONLY})
        return send_timed(conv, content)
    finally:
        conv.__exit__(None, None, None)


def v_two_turn(engine, wav_b64, img_b64, *, prefill_image: bool, image_on_critical_path=False,
               budget=None) -> dict:
    """D and its controls.

    prefill_image=True   -> the camera frame goes in the early turn (the real proposal).
    prefill_image=False  -> identical shape, no frame: isolates the plain second-turn effect.
    image_on_critical_path -> frame stays in the timed turn, early turn is text only.
    """
    conv = new_conv(engine, budget)
    try:
        early = []
        if prefill_image:
            early.append({"type": "image", "blob": img_b64})
        early.append({"type": "text",
                      "text": FRAME_TURN_TEXT if prefill_image else CONTROL_TURN_TEXT})

        t0 = time.perf_counter()
        pre = send_timed(conv, early, max_tokens=PREFILL_TURN_MAX_TOKENS)
        offline_s = time.perf_counter() - t0

        # ---- simulated speech end: everything below is on the critical path ----
        content = [{"type": "audio", "blob": wav_b64}]
        if image_on_critical_path:
            content.append({"type": "image", "blob": img_b64})
        content.append({"type": "text",
                        "text": TURN_TEXT_WITH_IMAGE if (prefill_image or image_on_critical_path)
                        else TURN_TEXT_AUDIO_ONLY})
        r = send_timed(conv, content)
        r["offline_s"] = offline_s
        r["offline_tokens"] = pre["tokens_after"] - pre["tokens_before"]
        r["early_reply"] = pre["text"][:40]
        return r
    finally:
        conv.__exit__(None, None, None)


def v_staged_session(engine) -> dict:
    """B: raw Session incremental prefill. Text works; image/audio are probed and reported."""
    from litert_lm._ffi import InputDataType

    TYPES = {"text": InputDataType.TEXT, "image": InputDataType.IMAGE,
             "image_end": InputDataType.IMAGE_END, "audio": InputDataType.AUDIO,
             "audio_end": InputDataType.AUDIO_END}

    def prefill(session, parts):
        lib = session._lib
        arr = (ctypes.c_void_p * len(parts))()
        made = []
        try:
            for i, (kind, payload) in enumerate(parts):
                if payload is None:
                    p = lib.litert_lm_input_data_create(TYPES[kind], None, 0)
                else:
                    b = payload.encode("utf-8") if isinstance(payload, str) else payload
                    p = lib.litert_lm_input_data_create(TYPES[kind], b, len(b))
                if not p:
                    raise RuntimeError(f"input_data_create failed for {kind}")
                made.append(p)
                arr[i] = p
            if lib.litert_lm_session_run_prefill(session._ptr, arr, len(parts)) != 0:
                raise RuntimeError("litert_lm_session_run_prefill returned -1")
        finally:
            for p in made:
                lib.litert_lm_input_data_delete(p)

    def decode(session):
        t0 = time.perf_counter()
        ttft, parts = None, []
        for chunk in session.run_decode_async():
            if ttft is None:
                ttft = time.perf_counter() - t0
            parts.append(chunk.texts[0] if chunk.texts else "")
        return {"ttft": ttft, "total": time.perf_counter() - t0, "text": "".join(parts)}

    def session():
        return engine.create_session(apply_prompt_template=False, sampler_config=GREEDY,
                                     max_output_tokens=MAX_OUTPUT_TOKENS)

    static = f"<|turn>system\n{SYSTEM_PROMPT}<turn|>\n<|turn>user\n"
    late = "What is the capital of France? Answer in one short sentence.<turn|>\n<|turn>model\n"

    out = {}

    s = session()
    try:
        t0 = time.perf_counter()
        prefill(s, [("text", static + late)])
        r = decode(s)
        r["critical_s"] = round(time.perf_counter() - t0, 3)
        out["text_monolithic"] = r
    finally:
        s.close()

    s = session()
    try:
        t0 = time.perf_counter()
        prefill(s, [("text", static)])
        offline = time.perf_counter() - t0
        t1 = time.perf_counter()
        prefill(s, [("text", late)])
        r = decode(s)
        r["critical_s"] = round(time.perf_counter() - t1, 3)
        r["offline_s"] = round(offline, 3)
        out["text_staged"] = r
    finally:
        s.close()

    out["text_identical"] = (out["text_monolithic"]["text"].strip()
                             == out["text_staged"]["text"].strip())

    # Multimodal at the Session layer: expected to fail, record exactly how.
    img = fixtures.make_image_b64()
    import base64
    probes = {
        "image_raw_jpeg": [("text", "<|turn>user\n"), ("image", base64.b64decode(img)),
                           ("image_end", None), ("text", "Describe it.<turn|>\n<|turn>model\n")],
        "audio_raw_wav": [("text", "<|turn>user\n"),
                          ("audio", (Path(__file__).parent / "fixtures" /
                                     f"{AUDIO_FIXTURE}.wav").read_bytes()),
                          ("audio_end", None),
                          ("text", "Answer it.<turn|>\n<|turn>model\n")],
    }
    out["multimodal_probes"] = {}
    for name, parts in probes.items():
        s = session()
        try:
            prefill(s, parts)
            out["multimodal_probes"][name] = "unexpectedly succeeded"
        except Exception as e:
            out["multimodal_probes"][name] = f"{type(e).__name__}: {e}"
        finally:
            s.close()
    return out


def v_multiturn(engine, wav_b64, img_b64, *, overlap: bool, exchanges: int = 3) -> list[dict]:
    """F: steady state. One conversation, several exchanges, frame early or inline.

    First-turn numbers flatter the overlap design because they also move the system
    prompt off the critical path, which only ever happens once. This measures what a
    real session sees on turn 2 and beyond.
    """
    conv = new_conv(engine)
    try:
        out = []
        for _ in range(exchanges):
            offline_s, offline_tok = 0.0, 0
            if overlap:
                t0 = time.perf_counter()
                pre = send_timed(conv, [{"type": "image", "blob": img_b64},
                                        {"type": "text", "text": FRAME_TURN_TEXT}],
                                 max_tokens=PREFILL_TURN_MAX_TOKENS)
                offline_s = time.perf_counter() - t0
                offline_tok = pre["tokens_after"] - pre["tokens_before"]

            content = [{"type": "audio", "blob": wav_b64}]
            if not overlap:
                content.append({"type": "image", "blob": img_b64})
            content.append({"type": "text", "text": TURN_TEXT_WITH_IMAGE})
            r = send_timed(conv, content)
            r["offline_s"] = offline_s
            r["offline_tokens"] = offline_tok
            out.append(r)
        return out
    finally:
        conv.__exit__(None, None, None)


def v_truncation(engine, wav_b64) -> dict:
    """Does a turn cut short by max_output_tokens cost the next turn its cached prefix?

    server.py caps replies at 256 tokens, so a long reply can be truncated mid-sentence.
    If truncation invalidates the KV prefix, the following turn silently gets slower.
    """
    out = {}
    for name, first_max in (("clean_stop", PREFILL_TURN_MAX_TOKENS), ("truncated", 6)):
        conv = new_conv(engine)
        try:
            prompt = (CONTROL_TURN_TEXT if name == "clean_stop"
                      else "Count slowly from one to fifty in words.")
            first = send_timed(conv, [{"type": "text", "text": prompt}], max_tokens=first_max)
            r = send_timed(conv, [{"type": "audio", "blob": wav_b64},
                                  {"type": "text", "text": TURN_TEXT_AUDIO_ONLY}])
            out[name] = {
                "ttft": round(r["ttft"], 3),
                "critical_tokens": r["tokens_after"] - r["tokens_before"],
                "first_turn_tokens": first["tokens_after"] - first["tokens_before"],
                "first_turn_reply": first["text"][:50],
            }
        finally:
            conv.__exit__(None, None, None)
    return out


def v_grounding(engine, img_b64, *, two_turn: bool, budget=None) -> dict:
    """C: does moving the frame into an earlier turn cost visual grounding?"""
    conv = new_conv(engine, budget)
    try:
        if two_turn:
            send_timed(conv, [{"type": "image", "blob": img_b64},
                              {"type": "text", "text": FRAME_TURN_TEXT}],
                       max_tokens=PREFILL_TURN_MAX_TOKENS)
            r = send_timed(conv, [{"type": "text", "text": GROUNDING_QUESTION}], max_tokens=24)
        else:
            r = send_timed(conv, [{"type": "image", "blob": img_b64},
                                  {"type": "text", "text": GROUNDING_QUESTION}], max_tokens=24)
        return r
    finally:
        conv.__exit__(None, None, None)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=int, default=RUNS)
    ap.add_argument("--out", default=str(Path(__file__).parent / "results" / "overlap_prefill.json"))
    args = ap.parse_args()

    fixtures.generate_all()
    wav_b64 = fixtures.load_wav_b64(AUDIO_FIXTURE)
    img_b64 = fixtures.make_image_b64()

    t = time.perf_counter()
    engine = litert_lm.Engine(
        resolve_model_path(),
        backend=litert_lm.Backend.GPU(),
        vision_backend=litert_lm.Backend.GPU(),
        audio_backend=litert_lm.Backend.CPU(),  # this model requires cpu audio
    )
    engine.__enter__()
    log(f"engine loaded in {time.perf_counter() - t:.1f}s")

    has_budget = bind_visual_token_budget(engine._lib)
    log(f"visual_token_budget available: {has_budget}")

    results = {}

    def measure(name, fn):
        log(f"\n--- {name} ---")
        fn()  # warmup
        runs = [fn() for _ in range(args.runs)]
        for i, r in enumerate(runs):
            log(f"  run {i + 1}: ttft={r['ttft']:.3f}s total={r['total']:.3f}s "
                f"prefilled={r['tokens_after'] - r['tokens_before']}tok"
                + (f" (offline {r['offline_s']:.3f}s / {r['offline_tokens']}tok)"
                   if "offline_s" in r else ""))
        results[name] = summarize(runs)
        s = results[name]
        log(f"  => ttft {s['ttft_mean']}s +/-{s['ttft_stdev']} | "
            f"{s['critical_tokens']}tok on critical path | {runs[-1]['text'][:100]!r}")

    measure("A_monolithic_audio_image",
            lambda: v_monolithic(engine, wav_b64, img_b64, with_image=True))
    measure("A_monolithic_audio_only",
            lambda: v_monolithic(engine, wav_b64, img_b64, with_image=False))
    measure("D_overlap_frame_early",
            lambda: v_two_turn(engine, wav_b64, img_b64, prefill_image=True))
    measure("D_control_no_frame",
            lambda: v_two_turn(engine, wav_b64, img_b64, prefill_image=False))
    measure("D_control_frame_late",
            lambda: v_two_turn(engine, wav_b64, img_b64, prefill_image=False,
                               image_on_critical_path=True))

    if has_budget:
        for budget in (256, 64):
            measure(f"E_budget_{budget}_monolithic",
                    lambda b=budget: v_monolithic(engine, wav_b64, img_b64,
                                                  with_image=True, budget=b))
        measure("E_budget_64_overlap",
                lambda: v_two_turn(engine, wav_b64, img_b64, prefill_image=True, budget=64))

    log("\n--- B_session_staged_prefill ---")
    try:
        b = v_staged_session(engine)
        results["B_session_staged_prefill"] = b
        m, st = b["text_monolithic"], b["text_staged"]
        log(f"  text monolithic: critical={m['critical_s']}s -> {m['text'][:70]!r}")
        log(f"  text staged:     critical={st['critical_s']}s offline={st['offline_s']}s "
            f"-> {st['text'][:70]!r}")
        log(f"  identical output: {b['text_identical']}")
        for k, v in b["multimodal_probes"].items():
            log(f"  {k}: {v}")
    except Exception as e:
        log(f"  FAILED: {type(e).__name__}: {e}")
        results["B_session_staged_prefill"] = {"error": f"{type(e).__name__}: {e}"}

    log("\n--- F_multiturn (steady state) ---")
    multiturn = {}
    for label, overlap in (("inline", False), ("overlap", True)):
        v_multiturn(engine, wav_b64, img_b64, overlap=overlap, exchanges=1)  # warmup
        turns = v_multiturn(engine, wav_b64, img_b64, overlap=overlap)
        multiturn[label] = [
            {"ttft": round(t["ttft"], 3),
             "critical_tokens": t["tokens_after"] - t["tokens_before"],
             "context_tokens": t["tokens_after"],
             "offline_s": round(t["offline_s"], 3)}
            for t in turns
        ]
        for i, t in enumerate(multiturn[label]):
            log(f"  {label} exchange {i + 1}: ttft={t['ttft']}s "
                f"crit={t['critical_tokens']}tok context={t['context_tokens']}tok"
                + (f" (offline {t['offline_s']}s)" if t["offline_s"] else ""))
    results["F_multiturn"] = multiturn

    log("\n--- F_truncation_safety ---")
    trunc = v_truncation(engine, wav_b64)
    results["F_truncation_safety"] = trunc
    for k, v in trunc.items():
        log(f"  {k}: next-turn ttft={v['ttft']}s crit={v['critical_tokens']}tok "
            f"(first turn added {v['first_turn_tokens']}tok, reply {v['first_turn_reply']!r})")

    log("\n--- C_grounding ---")
    grounding = {}
    for name, two in [("monolithic", False), ("frame_early", True)]:
        r = v_grounding(engine, img_b64, two_turn=two)
        answer = r["text"].strip()
        grounding[name] = {"answer": answer[:120],
                           "correct": GROUNDING_EXPECTED in answer.lower()}
        log(f"  {name}: correct={grounding[name]['correct']} -> {answer[:80]!r}")
    if has_budget:
        r = v_grounding(engine, img_b64, two_turn=True, budget=64)
        answer = r["text"].strip()
        grounding["frame_early_budget_64"] = {"answer": answer[:120],
                                              "correct": GROUNDING_EXPECTED in answer.lower()}
        log(f"  frame_early_budget_64: correct={grounding['frame_early_budget_64']['correct']} "
            f"-> {answer[:80]!r}")
    results["C_grounding"] = grounding

    log("\n--- C_answer_quality (audio turns) ---")
    quality = {}
    for name in ("A_monolithic_audio_image", "D_overlap_frame_early"):
        text = results[name]["sample_text"]
        low = text.lower()
        quality[name] = {
            "mentions_paris": EXPECT_KEYWORD in low,
            "starts_with_marker": low.lstrip().startswith(("finished", "wait")),
            "has_transcript": "###transcript" in low,
        }
        log(f"  {name}: {quality[name]}")
    results["C_answer_quality"] = quality

    log("\n" + "=" * 82)
    base = results["A_monolithic_audio_image"]["ttft_mean"]
    log(f"{'variant':<30} {'ttft(s)':>8} {'total(s)':>9} {'crit tok':>9} {'vs A':>14}")
    for name, r in results.items():
        if not isinstance(r, dict) or r.get("ttft_mean") is None:
            continue
        d = base - r["ttft_mean"]
        log(f"{name:<30} {r['ttft_mean']:>8.3f} {r['total_mean']:>9.3f} "
            f"{r['critical_tokens']:>9} {d:>+9.3f}s ({d / base * 100:+.0f}%)")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2))
    log(f"\nwrote {out}")

    engine.__exit__(None, None, None)


if __name__ == "__main__":
    main()
