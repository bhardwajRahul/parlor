"""Experiment: overlapped speech prefill on llama.cpp, where chunks stay INSIDE one turn.

On litert-lm, streaming speech as separate conversation turns destroyed the transcript
(WER 0.77-0.79) because the model would not treat a previous audio turn as part of the
current utterance. That was a property of CLOSED turns, not of chunked audio. llama.cpp
allows many input_audio blocks inside a single open user turn, and llama-server does
prefix caching over media chunks, so the same overlap becomes expressible without ever
closing the turn.

Variants (mean of RUNS after 1 warmup, greedy, fresh cache each time):
  A_monolithic     one input_audio block with the whole utterance + instruction.
  B_chunked_noreuse  three input_audio blocks + instruction in ONE request, no warmups.
                     Isolates whether chunking by itself costs accuracy or time.
  C_overlap_arbitrary  chunks warmed during "speech" (audio-only requests, max_tokens=1),
                     then the final request adds the last chunk + instruction. Split at
                     equal time offsets.
  C_overlap_silence  same, split at the quietest point near each boundary.

Cache evidence comes from llama-server's own timings: prompt_n is what it actually had
to evaluate, cache_n is what it reused.

Requires llama-server already running (see LLAMA_URL). Start it with:
  llama-server -m <gemma-4-E2B-it-Q5_K_M.gguf> --mmproj <mmproj-BF16.gguf> \
      --port 8323 -ngl 99 -c 8192 -np 1 --temp 0 --cache-ram 0

--cache-ram 0 matters: it disables llama.cpp's cross-request prompt-state restore, which
would otherwise make a repeated baseline look free. In-slot prefix reuse still works.

Run:  uv run python benchmarks/experiment_overlap_llamacpp.py
"""

import argparse
import base64
import io
import json
import re
import statistics
import sys
import time
import urllib.request
import wave
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import numpy as np

import fixtures

LLAMA_URL = "http://127.0.0.1:8323/v1/chat/completions"
RUNS = 3
SR = 16000
TARGET_CHUNK_S = 3.0
MAX_OUTPUT_TOKENS = 120
FIXTURE = "long_question"

SYSTEM_PROMPT = (
    "You are a friendly, conversational AI assistant. The user talks to you "
    "through a microphone. Your reply is spoken aloud, so write plain "
    "conversational text: 1-4 short sentences, no formatting.\n"
    "\n"
    "Your reply MUST start with exactly one of these words on its own line, "
    "judging the user's speech:\n"
    "- FINISHED - the user completed their thought. Continue with your spoken "
    "response on the next line.\n"
    "- WAIT - the user has not finished.\n"
    "\n"
    "If the user sent audio, end your reply with a new line:\n"
    "###TRANSCRIPT: the exact words the user said\n"
)
RESPOND_TEXT = (
    "The user just spoke to you. Start with FINISHED or WAIT, then respond to what "
    "they said. End with the ###TRANSCRIPT line."
)

TRANSCRIPT_TAG = "###TRANSCRIPT:"
GROUND_TRUTH = fixtures.FIXTURES[FIXTURE][0]
RESPONSE_KEYWORDS = fixtures.FIXTURES[FIXTURE][1]


def log(*a):
    print(*a, flush=True)


# --- audio ---

def load_pcm(name: str) -> np.ndarray:
    with wave.open(str(fixtures.FIXTURES_DIR / f"{name}.wav"), "rb") as w:
        return np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16)


def dither(pcm: np.ndarray, seed: int) -> np.ndarray:
    """+/-1 LSB noise (-90 dBFS, inaudible) so each run's bytes are unique.

    Without this, llama-server can match a previous run's media hashes and report a
    baseline that is really a cache hit.
    """
    rng = np.random.default_rng(seed)
    return np.clip(pcm.astype(np.int32) + rng.integers(-1, 2, size=len(pcm)),
                   -32768, 32767).astype(np.int16)


def wav_b64(pcm: np.ndarray) -> str:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes(pcm.astype(np.int16).tobytes())
    return base64.b64encode(buf.getvalue()).decode()


def split_arbitrary(pcm: np.ndarray, chunk_s: float = TARGET_CHUNK_S) -> list[np.ndarray]:
    n = max(1, round(len(pcm) / (chunk_s * SR)))
    b = [round(i * len(pcm) / n) for i in range(n + 1)]
    return [pcm[x:y] for x, y in zip(b, b[1:])]


def split_at_silence(pcm: np.ndarray, chunk_s: float = TARGET_CHUNK_S,
                     search_s: float = 0.75) -> list[np.ndarray]:
    frame = int(0.02 * SR)
    nf = len(pcm) // frame
    rms = np.array([np.sqrt(np.mean(pcm[i * frame:(i + 1) * frame].astype(np.float64) ** 2))
                    for i in range(nf)])
    n = max(1, round(len(pcm) / (chunk_s * SR)))
    bounds = [0]
    for i in range(1, n):
        target = round(i * len(pcm) / n)
        lo = max(0, (target - int(search_s * SR)) // frame)
        hi = min(nf, (target + int(search_s * SR)) // frame)
        bounds.append(target if hi <= lo else int((lo + int(np.argmin(rms[lo:hi]))) * frame))
    bounds.append(len(pcm))
    return [pcm[x:y] for x, y in zip(bounds, bounds[1:])]


# --- scoring ---

def normalize_words(t: str) -> list[str]:
    return re.sub(r"[^a-z0-9' ]+", " ", t.lower()).split()


def wer(ref: str, hyp: str) -> float:
    r, h = normalize_words(ref), normalize_words(hyp)
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
        marker, text = m.group(1), text[m.end():]
    transcript = ""
    i = text.find(TRANSCRIPT_TAG)
    if i != -1:
        transcript, text = text[i + len(TRANSCRIPT_TAG):].strip(), text[:i]
    return {"marker": marker, "response": text.strip(), "transcript": transcript}


# --- llama-server client ---

def call(blocks: list, *, max_tokens: int, system: bool = True, stream: bool = False) -> dict:
    messages = []
    if system:
        messages.append({"role": "system", "content": SYSTEM_PROMPT})
    messages.append({"role": "user", "content": blocks})
    body = {
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": 0,
        "top_k": 1,
        "cache_prompt": True,
        "chat_template_kwargs": {"enable_thinking": False},  # else ~160 thought tokens
        "stream": stream,
        "timings_per_token": True,
    }
    req = urllib.request.Request(LLAMA_URL, data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    t0 = time.perf_counter()
    if not stream:
        r = json.load(urllib.request.urlopen(req, timeout=600))
        tm = r.get("timings", {})
        return {"ttft": time.perf_counter() - t0, "total": time.perf_counter() - t0,
                "text": r["choices"][0]["message"]["content"],
                "prompt_n": tm.get("prompt_n"), "cache_n": tm.get("cache_n"),
                "prompt_ms": tm.get("prompt_ms")}

    ttft, parts, timings = None, [], {}
    with urllib.request.urlopen(req, timeout=600) as resp:
        for raw in resp:
            line = raw.decode().strip()
            if not line.startswith("data: "):
                continue
            payload = line[6:]
            if payload == "[DONE]":
                break
            obj = json.loads(payload)
            if obj.get("timings"):
                timings = obj["timings"]
            for ch in obj.get("choices", []):
                piece = (ch.get("delta") or {}).get("content") or ""
                if piece:
                    if ttft is None:
                        ttft = time.perf_counter() - t0
                    parts.append(piece)
    return {"ttft": ttft if ttft is not None else time.perf_counter() - t0,
            "total": time.perf_counter() - t0, "text": "".join(parts),
            "prompt_n": timings.get("prompt_n"), "cache_n": timings.get("cache_n"),
            "prompt_ms": timings.get("prompt_ms")}


def bust_cache(tag: str):
    """Leave the slot holding unrelated tokens so the next request starts cold."""
    call([{"type": "text", "text": f"Say ok. filler {tag} " * 20}], max_tokens=1, system=False)


def audio_block(pcm: np.ndarray) -> dict:
    return {"type": "input_audio", "input_audio": {"data": wav_b64(pcm), "format": "wav"}}


# --- variants ---

def v_monolithic(pcm: np.ndarray, seed: int) -> dict:
    bust_cache(f"mono{seed}")
    r = call([audio_block(pcm), {"type": "text", "text": RESPOND_TEXT}],
             max_tokens=MAX_OUTPUT_TOKENS, stream=True)
    r["offline_s"] = 0.0
    r["warm_calls"] = []
    return r


def v_chunked_single_request(pcm: np.ndarray, seed: int, splitter) -> dict:
    bust_cache(f"chunk{seed}")
    segs = splitter(pcm)
    blocks = [audio_block(s) for s in segs] + [{"type": "text", "text": RESPOND_TEXT}]
    r = call(blocks, max_tokens=MAX_OUTPUT_TOKENS, stream=True)
    r["offline_s"] = 0.0
    r["warm_calls"] = []
    r["n_segments"] = len(segs)
    return r


def v_overlap(pcm: np.ndarray, seed: int, splitter) -> dict:
    """The real proposal: warm each chunk as it 'arrives', then finish at speech end."""
    bust_cache(f"ovl{seed}")
    segs = splitter(pcm)
    warm = []
    offline = 0.0
    # Audio-only warmups. A trailing text block here would break prefix reuse,
    # because the text lands before the next chunk and diverges the prefix.
    for i in range(len(segs) - 1):
        t0 = time.perf_counter()
        w = call([audio_block(s) for s in segs[:i + 1]], max_tokens=1)
        dt = time.perf_counter() - t0
        offline += dt
        warm.append({"wall": round(dt, 3), "prompt_n": w["prompt_n"], "cache_n": w["cache_n"]})

    # ---- speech end ----
    r = call([audio_block(s) for s in segs] + [{"type": "text", "text": RESPOND_TEXT}],
             max_tokens=MAX_OUTPUT_TOKENS, stream=True)
    r["offline_s"] = offline
    r["warm_calls"] = warm
    r["n_segments"] = len(segs)
    return r


def summarize(runs: list[dict]) -> dict:
    last = runs[-1]
    p = parse_reply(last["text"])
    wers = [wer(GROUND_TRUTH, parse_reply(r["text"])["transcript"]) for r in runs]
    return {
        "ttft_mean": round(statistics.mean([r["ttft"] for r in runs]), 3),
        "ttft_stdev": round(statistics.stdev([r["ttft"] for r in runs]), 3) if len(runs) > 1 else 0.0,
        "total_mean": round(statistics.mean([r["total"] for r in runs]), 3),
        "prompt_n": last["prompt_n"],
        "cache_n": last["cache_n"],
        "prompt_ms_mean": round(statistics.mean([r["prompt_ms"] or 0 for r in runs]), 1),
        "offline_mean": round(statistics.mean([r["offline_s"] for r in runs]), 3),
        "max_warm_call_s": max([w["wall"] for r in runs for w in r["warm_calls"]], default=0.0),
        "n_segments": last.get("n_segments", 1),
        "wer_mean": round(statistics.mean(wers), 3),
        "wer_max": round(max(wers), 3),
        "marker": p["marker"],
        "transcript": p["transcript"][:300],
        "response": p["response"][:200],
        "response_keywords_hit": sum(k in p["response"].lower() for k in RESPONSE_KEYWORDS),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=int, default=RUNS)
    ap.add_argument("--url", default=LLAMA_URL)
    ap.add_argument("--out", default=str(Path(__file__).parent / "results" / "overlap_llamacpp.json"))
    args = ap.parse_args()
    globals()["LLAMA_URL"] = args.url

    fixtures.generate_all()
    pcm = load_pcm(FIXTURE)
    log(f"fixture {FIXTURE}: {len(pcm) / SR:.2f}s, ground truth {len(normalize_words(GROUND_TRUTH))} words")
    log(f"arbitrary split {[round(len(s) / SR, 2) for s in split_arbitrary(pcm)]}, "
        f"silence split {[round(len(s) / SR, 2) for s in split_at_silence(pcm)]}")

    results = {}

    def measure(name, fn):
        log(f"\n--- {name} ---")
        fn(0)  # warmup run, discarded
        runs = []
        for i in range(args.runs):
            r = fn(i + 1)
            runs.append(r)
            w = (f" warm={[c['wall'] for c in r['warm_calls']]} "
                 f"cache_n={[c['cache_n'] for c in r['warm_calls']]}") if r["warm_calls"] else ""
            log(f"  run {i + 1}: ttft={r['ttft']:.3f}s total={r['total']:.3f}s "
                f"prompt_n={r['prompt_n']} cache_n={r['cache_n']} "
                f"prefill={r['prompt_ms']:.0f}ms{w}")
        results[name] = summarize(runs)
        s = results[name]
        log(f"  => ttft {s['ttft_mean']}s +/-{s['ttft_stdev']} | WER {s['wer_mean']} "
            f"(max {s['wer_max']}) | evaluated {s['prompt_n']}tok, reused {s['cache_n']}tok")
        log(f"     transcript: {s['transcript']!r}")

    measure("A_monolithic", lambda seed: v_monolithic(dither(pcm, seed), seed))
    measure("B_chunked_1req_arbitrary",
            lambda seed: v_chunked_single_request(dither(pcm, seed), seed, split_arbitrary))
    measure("C_overlap_arbitrary",
            lambda seed: v_overlap(dither(pcm, seed), seed, split_arbitrary))
    measure("C_overlap_silence",
            lambda seed: v_overlap(dither(pcm, seed), seed, split_at_silence))

    log("\n" + "=" * 100)
    base = results["A_monolithic"]
    log(f"{'variant':<28} {'ttft':>7} {'saved':>8} {'WER':>6} {'eval tok':>9} "
        f"{'reused':>7} {'prefill ms':>11} {'offline':>8}")
    for name, r in results.items():
        d = base["ttft_mean"] - r["ttft_mean"]
        log(f"{name:<28} {r['ttft_mean']:>7.3f} {d:>+7.3f}s {r['wer_mean']:>6.3f} "
            f"{r['prompt_n']:>9} {r['cache_n']:>7} {r['prompt_ms_mean']:>10.1f} "
            f"{r['offline_mean']:>7.3f}s")

    best = min((r for n, r in results.items() if n.startswith("C_")),
               key=lambda r: r["ttft_mean"])
    log("\nVerdict inputs:")
    log(f"  baseline ttft / WER      : {base['ttft_mean']}s / {base['wer_mean']}")
    log(f"  best overlap ttft / WER  : {best['ttft_mean']}s / {best['wer_mean']}")
    log(f"  saved                    : {base['ttft_mean'] - best['ttft_mean']:+.3f}s")
    log(f"  litert stitch bar to beat: +0.166s at WER 0.029")
    log(f"  slowest warm call        : {best['max_warm_call_s']}s (must stay under "
        f"{TARGET_CHUNK_S}s to keep pace with speech)")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2))
    log(f"\nwrote {out}")


if __name__ == "__main__":
    main()
