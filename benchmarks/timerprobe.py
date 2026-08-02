"""Does the model run a timer from elapsed time alone? (experiment)

Parlor has no timer machinery at this commit: the user asks for a
twenty-second timer (the model can only promise), waits out the deadline,
then speaks again. The per-turn time note (TIME_NOTE_MIN_S) tells the
model how much quiet preceded the return — does it spontaneously announce
that the timer is done?

A turn-based system can never announce into silence (nothing wakes the
model), so this measures the best case: the user happens to speak after
the deadline. The result informs the real feature — a <timer> control tag
plus a server-scheduled proactive turn.

Start the server with production sampling and a low note threshold:
  TEMPERATURE=0.7 TIME_NOTE_MIN_S=10 uv run parlor
Then:
  uv run python benchmarks/timerprobe.py --trials 6
"""

import argparse
import asyncio
import base64
import contextlib
import json
import re
import time
from pathlib import Path

import fixtures
from bench import SERVER_URL, connect, run_turn

PROBE_DIR = fixtures.FIXTURES_DIR / "timerprobe"
UTTERANCES = {
    "probe_set_timer": "Set a timer for twenty seconds, please.",
    "probe_return": "Okay, I'm back now. How's it going?",
}

# Explicit timer announcement vs. mere awareness that time passed — the
# gap between the two rates is the finding.
ANNOUNCE_RE = re.compile(
    r"\btimer\b|time'?s? (?:is )?up|went off|twenty seconds|20 seconds", re.I)
TIME_AWARE_RE = re.compile(
    r"\bseconds?\b|\bminutes?\b|\ba while\b|welcome back", re.I)


def ensure_fixtures() -> dict[str, str]:
    PROBE_DIR.mkdir(parents=True, exist_ok=True)
    missing = [n for n in UTTERANCES if not (PROBE_DIR / f"{n}.wav").exists()]
    if missing:
        from parlor import tts
        backend = tts.load()
        for name in missing:
            pcm = fixtures._synthesize(backend, UTTERANCES[name])
            fixtures._write_wav(PROBE_DIR / f"{name}.wav", pcm, fixtures.TARGET_SR)
            print(f"fixture {name}: {len(pcm) / fixtures.TARGET_SR:.1f}s speech")
    return {n: base64.b64encode((PROBE_DIR / f"{n}.wav").read_bytes()).decode()
            for n in UTTERANCES}


async def speak(ws, payload: dict) -> dict:
    r = await run_turn(ws, payload, timeout=60)
    if (r["marker"] or "").startswith("incomplete"):
        r = await run_turn(ws, {"type": "flush"}, timeout=60)
    return r


async def trial(url: str, wavs: dict, wait_s: float) -> dict:
    async with connect(url) as ws:
        ack = await speak(ws, {"audio": wavs["probe_set_timer"]})
        await asyncio.sleep(wait_s)
        back = await speak(ws, {"audio": wavs["probe_return"]})
    return {
        "ack": ack["text"].strip(),
        "reply": back["text"].strip(),
        "promised": bool(ANNOUNCE_RE.search(ack["text"])),
        "announced": bool(ANNOUNCE_RE.search(back["text"])),
        "time_aware": bool(TIME_AWARE_RE.search(back["text"])),
    }


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default=SERVER_URL)
    ap.add_argument("--trials", type=int, default=6)
    ap.add_argument("--wait", type=float, default=30.0,
                    help="idle seconds between setting the timer and returning")
    ap.add_argument("--out", default=None, help="write results JSON here")
    args = ap.parse_args()

    wavs = ensure_fixtures()
    results = []
    for i in range(args.trials):
        r = await trial(args.url, wavs, args.wait)
        results.append(r)
        print(f"trial {i + 1}: promised={r['promised']} "
              f"announced={r['announced']} time_aware={r['time_aware']}")
        print(f"  ack:   {r['ack']!r}")
        print(f"  reply: {r['reply']!r}")

    n = len(results)
    announced = sum(r["announced"] for r in results)
    aware = sum(r["time_aware"] for r in results)
    print(f"\n== timerprobe: announced {announced}/{n}, "
          f"time-aware {aware}/{n} (wait {args.wait:.0f}s) ==")
    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps({
            "wait_s": args.wait,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "trials": results,
        }, indent=2))
        print(f"wrote {out_path}")


if __name__ == "__main__":
    with contextlib.suppress(KeyboardInterrupt):
        asyncio.run(main())
