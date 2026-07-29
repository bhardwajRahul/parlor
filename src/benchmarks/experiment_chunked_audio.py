"""Experiment: can we overlap prefill of the user's SPEECH by streaming it in chunks?

Variant D (camera frame prefilled at speech start) works because the frame is complete
the moment speech begins. Speech audio is not — it only exists progressively. The idea
tested here is to send each ~3s segment as its own throwaway turn while the user is still
talking, so that at speech end only the final segment is left to prefill.

The prize is small by construction. Audio prefill scales with duration, and on this
machine a 2.15s utterance costs ~0.55s to first token versus ~0.84s for a 9.45s one, so
the audio-length-dependent component of a long utterance is only ~0.3s. The question is
therefore not just "is it faster" but "is it faster *without* costing transcript
accuracy", because chunk boundaries cut words mid-phoneme.

Variants (mean of RUNS after 1 warmup, greedy):
  A_monolithic        full 9.45s utterance in one turn — today's behaviour.
  B_chunk_arbitrary   split at equal time offsets, N-1 segments prefilled as OK-turns.
  B_chunk_silence     split at the quietest point near each boundary.
  B_chunk_stitch      silence split, plus an instruction telling the model the audio
                      arrived in parts and should be treated as one utterance.

Correctness is the deciding measure: word error rate of the ###TRANSCRIPT against the
known fixture text. Any WER regression versus A means don't ship.

Run:  uv run python benchmarks/experiment_chunked_audio.py
"""

import argparse
import base64
import io
import json
import os
import re
import statistics
import sys
import time
import wave
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np

import litert_lm

import fixtures

HF_REPO = "litert-community/gemma-4-E2B-it-litert-lm"
HF_FILENAME = "gemma-4-E2B-it.litertlm"

RUNS = 3
MAX_OUTPUT_TOKENS = 120
CHUNK_MAX_TOKENS = 4
FIXTURE = "long_question"
TARGET_CHUNK_S = 3.0
SR = 16000

# Copied from server.py rather than imported — importing it loads the model and TTS.
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

RESPOND_TEXT = (
    "The user just spoke to you. Start with FINISHED or WAIT, then respond to what "
    "they said. End with the ###TRANSCRIPT line."
)
CHUNK_TEXT = (
    "The user is still speaking; this is a partial audio segment. "
    "Reply with only the word OK."
)
STITCH_TEXT = (
    "This is the final segment of the user's speech. The earlier audio segments above "
    "are earlier parts of the SAME continuous sentence - words may be cut across the "
    "boundaries. Treat all the segments together as one utterance. "
    "Start with FINISHED or WAIT, then respond. End with the ###TRANSCRIPT line, "
    "giving the exact words of the WHOLE utterance across every segment."
)

TRANSCRIPT_TAG = "###TRANSCRIPT:"
GREEDY = litert_lm.SamplerConfig(top_k=1, top_p=1.0, temperature=0.0, seed=0)
GROUND_TRUTH = fixtures.FIXTURES[FIXTURE][0]
RESPONSE_KEYWORDS = fixtures.FIXTURES[FIXTURE][1]  # ["english", "pronunciation"]


def log(*a):
    print(*a, flush=True)


def resolve_model_path() -> str:
    path = os.environ.get("MODEL_PATH", "")
    if path:
        return path
    from huggingface_hub import hf_hub_download
    return hf_hub_download(repo_id=HF_REPO, filename=HF_FILENAME)


# --- audio helpers ---

def load_pcm(name: str) -> np.ndarray:
    with wave.open(str(fixtures.FIXTURES_DIR / f"{name}.wav"), "rb") as w:
        return np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16)


def pcm_to_wav_b64(pcm: np.ndarray) -> str:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes(pcm.astype(np.int16).tobytes())
    return base64.b64encode(buf.getvalue()).decode()


def split_arbitrary(pcm: np.ndarray, chunk_s: float = TARGET_CHUNK_S) -> list[np.ndarray]:
    n = max(1, round(len(pcm) / (chunk_s * SR)))
    bounds = [round(i * len(pcm) / n) for i in range(n + 1)]
    return [pcm[a:b] for a, b in zip(bounds, bounds[1:])]


def split_at_silence(pcm: np.ndarray, chunk_s: float = TARGET_CHUNK_S,
                     search_s: float = 0.75) -> list[np.ndarray]:
    """Same target boundaries, nudged to the quietest 20ms frame within +/- search_s."""
    frame = int(0.02 * SR)
    n_frames = len(pcm) // frame
    rms = np.array([
        np.sqrt(np.mean(pcm[i * frame:(i + 1) * frame].astype(np.float64) ** 2))
        for i in range(n_frames)
    ])
    n = max(1, round(len(pcm) / (chunk_s * SR)))
    bounds = [0]
    for i in range(1, n):
        target = round(i * len(pcm) / n)
        lo = max(0, (target - int(search_s * SR)) // frame)
        hi = min(n_frames, (target + int(search_s * SR)) // frame)
        if hi <= lo:
            bounds.append(target)
            continue
        bounds.append(int((lo + int(np.argmin(rms[lo:hi]))) * frame))
    bounds.append(len(pcm))
    return [pcm[a:b] for a, b in zip(bounds, bounds[1:])]


# --- scoring ---

def normalize_words(text: str) -> list[str]:
    return re.sub(r"[^a-z0-9' ]+", " ", text.lower()).split()


def wer(reference: str, hypothesis: str) -> float:
    """Word error rate: edit distance over words, normalised by reference length."""
    r, h = normalize_words(reference), normalize_words(hypothesis)
    if not r:
        return 0.0 if not h else 1.0
    prev = list(range(len(h) + 1))
    for i, rw in enumerate(r, 1):
        cur = [i] + [0] * len(h)
        for j, hw in enumerate(h, 1):
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (rw != hw))
        prev = cur
    return prev[len(h)] / len(r)


def parse_reply(text: str) -> dict:
    marker = ""
    m = re.match(r"[\s*_]*(FINISHED|FINISH|WAITING|WAIT)\b[:.]?", text)
    if m:
        marker = m.group(1)
        text = text[m.end():]
    transcript = ""
    idx = text.find(TRANSCRIPT_TAG)
    if idx != -1:
        transcript = text[idx + len(TRANSCRIPT_TAG):].strip()
        text = text[:idx]
    return {"marker": marker, "response": text.strip(), "transcript": transcript}


# --- runner ---

def new_conv(engine):
    conv = engine.create_conversation(
        messages=[{"role": "system", "content": SYSTEM_PROMPT}],
        sampler_config=GREEDY,
    )
    conv.__enter__()
    return conv


def send_timed(conv, content: list, max_tokens: int) -> dict:
    before = conv.token_count
    t0 = time.perf_counter()
    ttft, parts = None, []
    for chunk in conv.send_message_async({"role": "user", "content": content},
                                         max_output_tokens=max_tokens):
        text = "".join(p.get("text", "") for p in chunk.get("content", [])
                       if isinstance(p, dict))
        if text and ttft is None:
            ttft = time.perf_counter() - t0
        parts.append(text)
    return {
        "ttft": ttft if ttft is not None else time.perf_counter() - t0,
        "total": time.perf_counter() - t0,
        "text": "".join(parts),
        "tokens": conv.token_count - before,
        "context": conv.token_count,
    }


def v_monolithic(engine, pcm) -> dict:
    conv = new_conv(engine)
    try:
        r = send_timed(conv, [{"type": "audio", "blob": pcm_to_wav_b64(pcm)},
                              {"type": "text", "text": RESPOND_TEXT}], MAX_OUTPUT_TOKENS)
        r["offline_s"] = 0.0
        r["offline_tokens"] = 0
        r["chunk_times"] = []
        r["total_tokens"] = r["tokens"]
        return r
    finally:
        conv.__exit__(None, None, None)


def v_chunked(engine, pcm, *, splitter, final_text: str) -> dict:
    segments = splitter(pcm)
    conv = new_conv(engine)
    try:
        offline_s, offline_tokens, chunk_times = 0.0, 0, []
        for seg in segments[:-1]:
            t0 = time.perf_counter()
            pre = send_timed(conv, [{"type": "audio", "blob": pcm_to_wav_b64(seg)},
                                    {"type": "text", "text": CHUNK_TEXT}], CHUNK_MAX_TOKENS)
            dt = time.perf_counter() - t0
            chunk_times.append(round(dt, 3))
            offline_s += dt
            offline_tokens += pre["tokens"]

        # ---- speech end: only the final segment is on the critical path ----
        r = send_timed(conv, [{"type": "audio", "blob": pcm_to_wav_b64(segments[-1])},
                              {"type": "text", "text": final_text}], MAX_OUTPUT_TOKENS)
        r["offline_s"] = offline_s
        r["offline_tokens"] = offline_tokens
        r["chunk_times"] = chunk_times
        r["n_segments"] = len(segments)
        r["segment_durations"] = [round(len(s) / SR, 2) for s in segments]
        r["total_tokens"] = offline_tokens + r["tokens"]
        return r
    finally:
        conv.__exit__(None, None, None)


def summarize(runs: list[dict]) -> dict:
    last = runs[-1]
    parsed = parse_reply(last["text"])
    return {
        "ttft_mean": round(statistics.mean([r["ttft"] for r in runs]), 3),
        "ttft_stdev": round(statistics.stdev([r["ttft"] for r in runs]), 3) if len(runs) > 1 else 0.0,
        "total_mean": round(statistics.mean([r["total"] for r in runs]), 3),
        "offline_mean": round(statistics.mean([r["offline_s"] for r in runs]), 3),
        "max_chunk_turn_s": max([max(r["chunk_times"]) for r in runs if r["chunk_times"]],
                                default=0.0),
        "critical_tokens": last["tokens"],
        "total_tokens": last["total_tokens"],
        "context_tokens": last["context"],
        "n_segments": last.get("n_segments", 1),
        "segment_durations": last.get("segment_durations", []),
        "marker": parsed["marker"],
        "response": parsed["response"][:200],
        "transcript": parsed["transcript"][:300],
        "wer": round(wer(GROUND_TRUTH, parsed["transcript"]), 3),
        "response_keywords_hit": sum(k in parsed["response"].lower()
                                     for k in RESPONSE_KEYWORDS),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=int, default=RUNS)
    ap.add_argument("--out", default=str(Path(__file__).parent / "results" / "chunked_audio.json"))
    args = ap.parse_args()

    fixtures.generate_all()
    pcm = load_pcm(FIXTURE)
    log(f"fixture {FIXTURE}: {len(pcm) / SR:.2f}s")
    log(f"ground truth: {GROUND_TRUTH!r}")
    log(f"arbitrary split: {[round(len(s) / SR, 2) for s in split_arbitrary(pcm)]}")
    log(f"silence split:   {[round(len(s) / SR, 2) for s in split_at_silence(pcm)]}")

    t = time.perf_counter()
    engine = litert_lm.Engine(
        resolve_model_path(),
        backend=litert_lm.Backend.GPU(),
        vision_backend=litert_lm.Backend.GPU(),
        audio_backend=litert_lm.Backend.CPU(),  # this model requires cpu audio
    )
    engine.__enter__()
    log(f"engine loaded in {time.perf_counter() - t:.1f}s")

    results = {}

    def measure(name, fn):
        log(f"\n--- {name} ---")
        fn()  # warmup
        runs = [fn() for _ in range(args.runs)]
        for i, r in enumerate(runs):
            log(f"  run {i + 1}: ttft={r['ttft']:.3f}s total={r['total']:.3f}s "
                f"crit={r['tokens']}tok"
                + (f" offline={r['offline_s']:.3f}s/{r['offline_tokens']}tok "
                   f"chunks={r['chunk_times']}" if r["chunk_times"] else ""))
        results[name] = summarize(runs)
        s = results[name]
        log(f"  => ttft {s['ttft_mean']}s +/-{s['ttft_stdev']} | WER {s['wer']} | "
            f"marker {s['marker']!r} | {s['total_tokens']}tok total")
        log(f"     transcript: {s['transcript']!r}")
        log(f"     response:   {s['response'][:120]!r}")

    measure("A_monolithic", lambda: v_monolithic(engine, pcm))
    measure("B_chunk_arbitrary",
            lambda: v_chunked(engine, pcm, splitter=split_arbitrary, final_text=RESPOND_TEXT))
    measure("B_chunk_silence",
            lambda: v_chunked(engine, pcm, splitter=split_at_silence, final_text=RESPOND_TEXT))
    measure("B_chunk_stitch",
            lambda: v_chunked(engine, pcm, splitter=split_at_silence, final_text=STITCH_TEXT))

    log("\n" + "=" * 92)
    base = results["A_monolithic"]
    log(f"{'variant':<22} {'ttft':>7} {'saved':>8} {'WER':>6} {'crit tok':>9} "
        f"{'all tok':>8} {'offline':>8}")
    for name, r in results.items():
        d = base["ttft_mean"] - r["ttft_mean"]
        log(f"{name:<22} {r['ttft_mean']:>7.3f} {d:>+7.3f}s {r['wer']:>6.3f} "
            f"{r['critical_tokens']:>9} {r['total_tokens']:>8} {r['offline_mean']:>7.3f}s")

    log("\nVerdict inputs:")
    log(f"  baseline WER            : {base['wer']}")
    best = min((r for n, r in results.items() if n != "A_monolithic"),
               key=lambda r: r["ttft_mean"])
    log(f"  best chunked WER        : {best['wer']}")
    log(f"  best time saved         : {base['ttft_mean'] - best['ttft_mean']:+.3f}s")
    log(f"  extra KV tokens/utterance: {best['total_tokens'] - base['total_tokens']:+d}")
    log(f"  slowest chunk turn      : {best['max_chunk_turn_s']}s (must stay under "
        f"{TARGET_CHUNK_S}s to keep up with speech)")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2))
    log(f"\nwrote {out}")

    engine.__exit__(None, None, None)


if __name__ == "__main__":
    main()
