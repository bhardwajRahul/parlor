"""Control-tag emission benchmark — when a reply should carry an action
tag, which syntax does the model emit more reliably: a '###DELEGATE: task'
line (matching the existing ###TRANSCRIPT machinery) or an XML
'<delegate>task</delegate>' element?

Real synthesized speech goes through the production turn shape (audio part
+ transcript-first RESPOND_PROMPT) against a live llama-server, with the
delegation instruction appended to the system prompt in each syntax.
Scored per syntax:

    recall    — well-formed tag on delegate-worthy turns (higher = better)
    misfire   — any tag on ordinary turns (lower = better)
    malformed — tag-ish output the strict extractor rejects
    leaked    — tag fragments that would reach TTS and be spoken aloud

    uv run python benchmarks/tagbench.py --repeat 2 \
        --out benchmarks/results/tagbench-e4b.json

--suite timer benches server.py's TIMER_INSTRUCTION verbatim instead:
<timer> recall/misfire plus parse_ok (does server.parse_timer resolve the
duration the model emitted, including spelled-out numbers).

Runs its own llama-server on port 8099 (like turnbench), so a dev server
on 8081 can keep running. Fixture WAVs cache in fixtures/tagbench/.
"""

import argparse
import base64
import json
import os
import re
import time
from pathlib import Path

os.environ.setdefault("LLAMA_PORT", "8099")

import fixtures  # noqa: E402
from parlor import llama  # noqa: E402
from parlor.pipeline import StreamParser, audio_part, text_part  # noqa: E402

CACHE_DIR = Path(__file__).parent / "fixtures" / "tagbench"

# Utterances that should trigger a delegation (need web search / current
# data / deep research) and ordinary turns that must not.
DELEGATE_CASES = {
    "weather_tokyo": "What's the weather in Tokyo right now?",
    "pizza_rome": "Can you find the best pizza places in Rome for my trip?",
    "bitcoin_price": "Look up the current price of Bitcoin for me.",
    "phone_reviews": "I want to buy a new phone. Can you research the latest "
                     "reviews and tell me which one is best?",
    "flight_deals": "Can you search for cheap flights from Jakarta to Tokyo "
                    "next month?",
    "news_today": "What are the biggest news stories today?",
}
PLAIN_CASES = {
    "capital_france": "What is the capital of France?",
    "how_are_you": "Hey, how are you doing today?",
    "tell_joke": "Tell me a joke about cats.",
    "long_day": "I had a really long day at work and I just want to chat.",
    "english_advice": "Could you give me some advice on how to improve my "
                      "English pronunciation?",
    "what_is_rain": "Why does it rain more in the tropics?",
}

# Timer-tag cases (--suite timer, benched against server.TIMER_INSTRUCTION
# verbatim): positives ask for a timer — including spelled-out durations,
# which also exercise server.parse_timer — negatives mention durations
# without asking for one.
TIMER_CASES = {
    "pasta_three": "Set a timer for three minutes for the pasta.",
    "ten_minute": "Can you give me a ten minute timer?",
    "remind_mom": "Remind me in five minutes to call my mom back.",
    "twenty_sec": "Set a 20 second timer.",
    "oven_fortyfive": "Set a timer for forty-five seconds to check the oven.",
    "laundry_hour": "Please set a one hour timer for the laundry.",
}
TIMER_PLAIN_CASES = {
    "ran_faster": "Last time I ran three minutes faster than my friend.",
    "walk_station": "It usually takes about ten minutes to walk to the station.",
    "movie_long": "That movie was almost three hours long.",
    "microwave": "The timer on my old microwave is broken, it's so annoying.",
    "capital_france": "What is the capital of France?",
    "how_are_you": "Hey, how are you doing today?",
}

INSTRUCTION_COMMON = (
    "\n\nYou can hand a task to a powerful background research assistant "
    "that has web access. When the user asks for something that needs "
    "current information, web search, or deep research you cannot do "
    "yourself, briefly acknowledge the request in one short spoken "
    "sentence, and end your reply with exactly one line like this:\n"
    "{example}\n"
    "The task must be self-contained. For anything you can answer "
    "yourself, reply normally and never write {name}."
)

SYNTAXES = {
    "hash": {
        "example": "###DELEGATE: <the task>",
        "name": "###DELEGATE",
        # loose: did it attempt a tag at all; strict: would production parse
        # it. No bare 'DELEGATE:' alternative — that matches markup-free
        # prose and would score misfires asymmetrically against hash.
        "loose": re.compile(r"#{2,}\s*DELEGATE", re.IGNORECASE),
        "strict": re.compile(r"#{2,}[ \t]*DELEGATE[ \t]*:[ \t]*(?P<task>[^\n]+)",
                             re.IGNORECASE),
    },
    "xml": {
        "example": "<delegate>the task</delegate>",
        "name": "<delegate>",
        "loose": re.compile(r"<\s*/?\s*delegate|delegate\s*>", re.IGNORECASE),
        "strict": re.compile(r"<delegate>\s*(?P<task>[^<]+?)\s*</delegate>",
                             re.IGNORECASE),
    },
}


# Mirrors server.py's production prompts (imported so drift is impossible).
def production_prompts():
    from parlor import server
    return (server.SYSTEM_PROMPT, server.RESPOND_PROMPT.format(camera=""),
            server.DELEGATE_INSTRUCTION, server.TIMER_INSTRUCTION)


def ensure_fixtures(cases: dict[str, str]) -> dict[str, str]:
    """name -> WAV base64, synthesizing any missing clips."""
    missing = [n for n in cases if not (CACHE_DIR / f"{n}.wav").exists()]
    if missing:
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        from parlor import tts
        backend = tts.load()
        for name in missing:
            pcm = fixtures._synthesize(backend, cases[name])
            fixtures._write_wav(CACHE_DIR / f"{name}.wav", pcm, fixtures.TARGET_SR)
            print(f"fixture {name}: {len(pcm) / fixtures.TARGET_SR:.1f}s speech")
    return {n: base64.b64encode((CACHE_DIR / f"{n}.wav").read_bytes()).decode()
            for n in cases}


def parse_production(raw: str, tag: str) -> tuple[str, list]:
    """One reply through the real production parser: (what TTS would say,
    the control tags it extracted)."""
    p = StreamParser(expect_transcript=True, control_tags=(tag,))
    spoken = p.feed(raw)
    tail, _ = p.finalize()
    return " ".join(spoken + tail), p.tags


def spoken_text(raw: str, syntax: str) -> str:
    """What TTS would say: the XML path uses the real production parser
    (which extracts <delegate> elements); the hash path simulates a
    ###DELEGATE:-line parser by regex."""
    if syntax == "xml":
        return parse_production(raw, "delegate")[0]
    text = SYNTAXES["hash"]["strict"].sub("", raw)
    m = re.search(r"#{2,}[ \t]*TRANSCRIPT[ \t]*:[^\n]*\n?", text, re.IGNORECASE)
    if m:
        text = text[m.end():]
    return re.split(r"#{2,}", text)[0].strip()


def judge(raw: str, syntax: str, expects_tag: bool) -> dict:
    s = SYNTAXES[syntax]
    strict = s["strict"].search(raw)
    loose = bool(s["loose"].search(raw))
    spoken = spoken_text(raw, syntax)
    # Any tag residue in what gets spoken is the worst failure mode.
    leaked = "delegate" in spoken.lower() or "##" in spoken or "<" in spoken
    return {
        "expects_tag": expects_tag,
        "well_formed": bool(strict),
        "attempted": loose,
        "malformed": loose and not strict,
        "leaked": leaked,
        "task": strict.group("task").strip() if strict else None,
        "raw": raw,
    }


def judge_timer(raw: str, expects_tag: bool) -> dict:
    """Timer-suite judge: the production parser extracts the tag, and
    parse_ok reports whether server.parse_timer can resolve its duration.
    'timer' is legitimately SPOKEN in confirmations ("your three-minute
    timer is set") — unlike the delegate suites, only markup counts as a
    leak here."""
    from parlor.server import parse_timer
    spoken, tags = parse_production(raw, "timer")
    values = [v for n, v in tags if n == "TIMER"]
    attempted = bool(re.search(r"<\s*/?\s*timer\b", raw, re.IGNORECASE))
    return {
        "expects_tag": expects_tag,
        "well_formed": bool(values),
        "attempted": attempted,
        "malformed": attempted and not values,
        "leaked": "<" in spoken or "##" in spoken,
        "parse_ok": bool(values) and parse_timer(values[0])[0] is not None,
        "task": values[0] if values else None,
        "raw": raw,
    }


def score(results: list[dict]) -> dict:
    pos = [r for r in results if r["expects_tag"]]
    neg = [r for r in results if not r["expects_tag"]]
    out = {
        "recall": round(sum(r["well_formed"] for r in pos) / len(pos), 3) if pos else None,
        "misfire": round(sum(r["attempted"] for r in neg) / len(neg), 3) if neg else None,
        "malformed": sum(r["malformed"] for r in results),
        "leaked": sum(r["leaked"] for r in results),
        "n": len(results),
    }
    if any("parse_ok" in r for r in results):
        out["parse_ok"] = (round(sum(bool(r.get("parse_ok")) for r in pos)
                                 / len(pos), 3) if pos else None)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repeat", type=int, default=1)
    ap.add_argument("--syntaxes", default="hash,xml")
    ap.add_argument("--production", action="store_true",
                    help="bench server.py's DELEGATE_INSTRUCTION verbatim "
                         "(the xml syntax it uses) instead of the symmetric "
                         "A/B instruction — the regression guard for prompt "
                         "changes")
    ap.add_argument("--suite", choices=("delegate", "timer"), default="delegate",
                    help="'timer' benches server.py's TIMER_INSTRUCTION "
                         "verbatim: recall/misfire of <timer> plus parse_ok "
                         "(does parse_timer resolve the emitted duration)")
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    system, respond, delegate_instruction, timer_instruction = production_prompts()
    if args.suite == "timer":
        positive_cases = TIMER_CASES
        wavs = ensure_fixtures(TIMER_CASES | TIMER_PLAIN_CASES)
        variants = [("timer-production", system + timer_instruction, "timer")]
    else:
        positive_cases = DELEGATE_CASES
        wavs = ensure_fixtures(DELEGATE_CASES | PLAIN_CASES)
        if args.production:
            variants = [("production", system + delegate_instruction, "xml")]
        else:
            variants = [(s, system + INSTRUCTION_COMMON.format(
                            example=SYNTAXES[s]["example"], name=SYNTAXES[s]["name"]), s)
                        for s in args.syntaxes.split(",")]
    llama.start()
    try:
        out = {"model": llama.MODEL, "temperature": llama.TEMPERATURE,
               "repeat": args.repeat, "syntaxes": {}}
        for label, sys_prompt, syntax in variants:
            results = []
            for name, wav in wavs.items():
                expects = name in positive_cases
                for _ in range(args.repeat):
                    t0 = time.time()
                    raw = llama.chat_blocking(
                        [{"role": "system", "content": sys_prompt},
                         {"role": "user", "content": [audio_part(wav),
                                                      text_part(respond)]}],
                        max_tokens=256)
                    r = (judge_timer(raw, expects) if syntax == "timer"
                         else judge(raw, syntax, expects)) | {"case": name}
                    results.append(r)
                    verdict = ("tag" if r["well_formed"] else
                               "MALFORMED" if r["malformed"] else "plain")
                    flag = " LEAKED" if r["leaked"] else ""
                    if r["well_formed"] and r.get("parse_ok") is False:
                        flag += " PARSE_FAIL"
                    ok = "✓" if (r["well_formed"] == expects and not r["malformed"]
                                 and not r["leaked"]) else "✗"
                    print(f"{ok} [{label}] {name}: {verdict}{flag} "
                          f"({time.time() - t0:.1f}s)")
            stats = score(results)
            out["syntaxes"][label] = {"stats": stats, "results": results}
            print(f"\n{label}: {stats}\n")
    finally:
        llama.stop()

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(out, indent=2))
        print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
