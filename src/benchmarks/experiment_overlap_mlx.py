"""Experiment: overlapped speech prefill on MLX (mlx-vlm), companion to the llama.cpp run.

mlx-vlm 0.6.8 ships Automatic Prefix Caching (APC): block-level KV reuse keyed by a
chained hash, with an ``extra_hash`` carrying multimodal content. On paper that is
exactly the machinery needed to prefill speech chunks during the utterance and reuse
them at speech end.

There is a catch visible in the source. ``apc.media_safe_prefix_min`` requires any
reused prefix to contain EVERY media placeholder token, so that the suffix can be
re-embedded as text only:

    "APC restore paths consume full prompt-level image/video feature tensors. Until
     media-feature slicing is model-aware, restored prefixes must include every
     media placeholder token so the suffix can be embedded as text-only."

Overlapped speech is the opposite shape: the reused prefix is chunks 1..N-1 and the
suffix still contains chunk N. For images that guard would refuse the reuse outright.
For audio it does not fire at all, because ``multimodal_token_ids_from_config`` only
collects image_token_id / image_token_index / video_token_id / video_token_index and
never audio_token_id (258881 in this model's config). So APC believes an audio suffix
is plain text.

This script asks the empirical question that follows: does APC then reuse the audio
prefix correctly, or does it silently corrupt the utterance? Correctness is judged by
word error rate of the transcript against the known fixture text.

MUST run under the mlx venv interpreter, not the project venv:
  /Users/fikrikarim/.claude/jobs/6a53b319/tmp/mlxtest/bin/python \
      benchmarks/experiment_overlap_mlx.py
"""

import argparse
import json
import os
import re
import statistics
import sys
import tempfile
import time
import wave
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import numpy as np

MODEL = "mlx-community/gemma-4-e2b-it-4bit"
SR = 16000
TARGET_CHUNK_S = 3.0
RUNS = 3
MAX_TOKENS = 120
FIXTURE = "long_question"
FIXTURES_DIR = Path(__file__).parent / "fixtures"

GROUND_TRUTH = (
    "I have been trying to learn English for a few months now, and I "
    "wonder if you could give me some advice on how to improve my "
    "pronunciation when I speak with other people."
)
RESPOND_TEXT = (
    "The user just spoke to you. Respond briefly to what they said, then end with a "
    "new line:\n###TRANSCRIPT: the exact words the user said"
)
TRANSCRIPT_TAG = "###TRANSCRIPT:"


def log(*a):
    print(*a, flush=True)


def load_pcm(name: str) -> np.ndarray:
    with wave.open(str(FIXTURES_DIR / f"{name}.wav"), "rb") as w:
        return np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16)


def write_wav(path: Path, pcm: np.ndarray):
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes(pcm.astype(np.int16).tobytes())


def split_arbitrary(pcm, chunk_s=TARGET_CHUNK_S):
    n = max(1, round(len(pcm) / (chunk_s * SR)))
    b = [round(i * len(pcm) / n) for i in range(n + 1)]
    return [pcm[x:y] for x, y in zip(b, b[1:])]


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


def transcript_of(text: str) -> str:
    i = text.find(TRANSCRIPT_TAG)
    return text[i + len(TRANSCRIPT_TAG):].strip() if i != -1 else ""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=int, default=RUNS)
    ap.add_argument("--out", default=str(Path(__file__).parent / "results" / "overlap_mlx.json"))
    args = ap.parse_args()

    os.environ.setdefault("APC_ENABLED", "1")

    import mlx.core as mx
    from mlx_vlm import apply_chat_template, load
    from mlx_vlm import apc as _apc
    from mlx_vlm.generate import stream_generate

    results = {"api_findings": {}}

    pcm = load_pcm(FIXTURE)
    segs = split_arbitrary(pcm)
    tmp = Path(tempfile.mkdtemp(prefix="mlxaudio_"))
    full_path = tmp / "full.wav"
    write_wav(full_path, pcm)
    seg_paths = []
    for i, s in enumerate(segs):
        p = tmp / f"c{i}.wav"
        write_wav(p, s)
        seg_paths.append(p)
    log(f"fixture {len(pcm) / SR:.2f}s -> {len(segs)} chunks "
        f"{[round(len(s) / SR, 2) for s in segs]}")

    t = time.perf_counter()
    model, processor = load(MODEL)
    log(f"model loaded in {time.perf_counter() - t:.1f}s")
    config = model.config

    # --- what does APC think counts as media? ---
    media_ids = _apc.multimodal_token_ids_from_config(config)
    audio_tok = getattr(config, "audio_token_id", None)
    results["api_findings"]["apc_media_token_ids"] = sorted(int(i) for i in media_ids)
    results["api_findings"]["audio_token_id"] = audio_tok
    results["api_findings"]["audio_token_treated_as_media"] = (
        audio_tok is not None and int(audio_tok) in media_ids
    )
    log(f"APC media token ids: {sorted(media_ids)}; audio_token_id={audio_tok}; "
        f"audio counted as media: {results['api_findings']['audio_token_treated_as_media']}")

    def run(audio_paths, n_audios, instruction, apc_manager=None, max_tokens=MAX_TOKENS):
        prompt = apply_chat_template(processor, config, instruction,
                                     num_images=0, num_audios=n_audios)
        kwargs = {"max_tokens": max_tokens, "temperature": 0.0}
        if apc_manager is not None:
            kwargs["apc_manager"] = apc_manager
        t0 = time.perf_counter()
        ttft, parts = None, []
        for chunk in stream_generate(model, processor, prompt,
                                     audio=[str(p) for p in audio_paths], **kwargs):
            piece = getattr(chunk, "text", "") or ""
            if piece and ttft is None:
                ttft = time.perf_counter() - t0
            parts.append(piece)
        return {"ttft": ttft or (time.perf_counter() - t0),
                "total": time.perf_counter() - t0, "text": "".join(parts)}

    def measure(name, fn, runs=args.runs):
        log(f"\n--- {name} ---")
        try:
            fn()  # warmup
            rs = []
            for i in range(runs):
                r = fn()
                rs.append(r)
                tr = transcript_of(r["text"])
                log(f"  run {i + 1}: ttft={r['ttft']:.3f}s total={r['total']:.3f}s "
                    f"WER={wer(GROUND_TRUTH, tr):.3f}")
            wers = [wer(GROUND_TRUTH, transcript_of(r["text"])) for r in rs]
            results[name] = {
                "ttft_mean": round(statistics.mean([r["ttft"] for r in rs]), 3),
                "total_mean": round(statistics.mean([r["total"] for r in rs]), 3),
                "wer_mean": round(statistics.mean(wers), 3),
                "wer_max": round(max(wers), 3),
                "transcript": transcript_of(rs[-1]["text"])[:300],
                "sample": rs[-1]["text"][:250],
            }
            s = results[name]
            log(f"  => ttft {s['ttft_mean']}s | WER {s['wer_mean']}")
            log(f"     transcript: {s['transcript']!r}")
        except Exception as e:
            import traceback
            results[name] = {"error": f"{type(e).__name__}: {e}",
                             "traceback": traceback.format_exc()[-1200:]}
            log(f"  FAILED: {type(e).__name__}: {e}")
            log(traceback.format_exc()[-800:])

    measure("A_monolithic", lambda: run([full_path], 1, RESPOND_TEXT))
    measure("B_chunked_one_request", lambda: run(seg_paths, len(segs), RESPOND_TEXT))

    # --- the actual overlap attempt: warm chunks under a shared APC manager ---
    log("\n--- C_overlap_apc ---")
    try:
        mgr = _apc.from_env() or _apc.APCManager()
        log(f"  APCManager: block_size={mgr.block_size} num_blocks={mgr.num_blocks}")

        def overlap():
            # "during speech": prefill growing audio prefixes, discard output
            for i in range(len(segs) - 1):
                run(seg_paths[:i + 1], i + 1, "Wait.", apc_manager=mgr, max_tokens=1)
            # "speech end"
            return run(seg_paths, len(segs), RESPOND_TEXT, apc_manager=mgr)

        measure("C_overlap_apc", overlap, runs=args.runs)
        stats = mgr.stats.snapshot(mgr.num_blocks, mgr.block_size)
        results["api_findings"]["apc_stats"] = {
            k: v for k, v in stats.items() if isinstance(v, (int, float, str))
        }
        log(f"  APC stats: {results['api_findings']['apc_stats']}")
    except Exception as e:
        import traceback
        results["C_overlap_apc"] = {"error": f"{type(e).__name__}: {e}",
                                    "traceback": traceback.format_exc()[-1200:]}
        log(f"  C_overlap_apc FAILED: {type(e).__name__}: {e}")
        log(traceback.format_exc()[-800:])

    log("\n" + "=" * 80)
    log(f"{'variant':<26} {'ttft':>8} {'WER':>7}")
    for name, r in results.items():
        if isinstance(r, dict) and "ttft_mean" in r:
            log(f"{name:<26} {r['ttft_mean']:>8.3f} {r['wer_mean']:>7.3f}")
        elif isinstance(r, dict) and "error" in r:
            log(f"{name:<26} {'ERROR':>8}  {r['error'][:60]}")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2))
    log(f"\nwrote {out}")


if __name__ == "__main__":
    main()
