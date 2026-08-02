"""Action-architecture benchmark: in-band control tags (the retired
baseline) vs a decoupled JSON action head (a second request over the
cached prefix).

The question: should actions (timer, mode switch, research delegation)
ride the speech stream as tags the TagFilter must excise — or be decided
by a separate grammar-forced JSON request that can never be spoken, runs
at temperature 0 regardless of speech temperature, and sees the model's
own reply as evidence?

    Arch A (tags):  the retired tag prompts + MODE_SUFFIX (vendored in
                    legacy_tags); reply parsed by the vendored TagFilter;
                    tag values through legacy_tags.parse_timer. The
                    historical baseline.
    Arch B (head):  speech request with production's capability notes and
                    NO tag machinery; then a second request — same cached
                    prefix + the reply + a decider instruction — forced
                    to a flat JSON schema at temp 0. The model itself
                    converts "twenty minutes" to 1200 seconds. What ships
                    today (src/parlor/actions.py).

Per arch: recall (action detected on positives), misfire (action on
negatives), value_ok (exact seconds / exact mode / usable task), leak
(markup reaching TTS — structurally impossible for B), and for B the
wall time of the head request (it runs while TTS plays, so it is GPU
cost, not perceived latency).

    uv run python benchmarks/archbench.py --repeat 2 \
        --out benchmarks/results/archbench.json

Runs its own llama-server on port 8099 (like tagbench); WAVs cache in
fixtures/tagbench/.
"""

import argparse
import base64
import http.client
import json
import os
import time
from pathlib import Path

os.environ.setdefault("LLAMA_PORT", "8099")

import fixtures  # noqa: E402
import legacy_tags  # noqa: E402
from tagbench import TIMER_CASES, TIMER_PLAIN_CASES, ensure_fixtures  # noqa: E402
from parlor import actions  # noqa: E402
from parlor import llama  # noqa: E402
from parlor.pipeline import audio_part, text_part  # noqa: E402
from parlor import server  # noqa: E402

# ── cases ─────────────────────────────────────────────────────────────────
# expected action per case: ("timer", seconds) | ("mode", name) |
# ("research", None) | None. Timer/delegate cases reuse tagbench's; mode
# cases are new, with trap negatives that MENTION translating/listening.
DELEGATE_CASES = {
    "pizza_rome": "Can you search the web and find the best pizza place in "
                  "Rome right now?",
    "bitcoin_price": "Look up the current price of Bitcoin for me.",
    "news_today": "What are the biggest news stories today?",
}
MODE_CASES = {
    "arch_mode_translate": "From now on, please translate everything I say "
                           "into English.",
    "arch_mode_listen": "Please just listen quietly for a while and don't "
                        "respond. I want to think out loud.",
    # Live-session failures (2026-08-02): "stay silent" phrasings missed
    # the listen clause entirely, and the "for one minute" variant fired
    # a 60s TIMER that rang into the wanted silence.
    "arch_mode_silent": "Can you stay silent for a while?",
    "arch_mode_silent_minute": "Can you stay silent for one minute?",
}
MODE_TRAP_CASES = {
    "arch_trap_jazz": "Lately I have been listening to a lot of jazz piano "
                      "records in the evening.",
    "arch_trap_french": "How do you say good morning in French?",
}

EXPECTED: dict[str, tuple | None] = {
    "pasta_three": ("timer", 180), "ten_minute": ("timer", 600),
    "remind_mom": ("timer", 300), "twenty_sec": ("timer", 20),
    "oven_fortyfive": ("timer", 45), "laundry_hour": ("timer", 3600),
    **{n: None for n in TIMER_PLAIN_CASES},
    **{n: ("research", None) for n in DELEGATE_CASES},
    "arch_mode_translate": ("mode", "translate"),
    "arch_mode_listen": ("mode", "listen"),
    "arch_mode_silent": ("mode", "listen"),
    "arch_mode_silent_minute": ("mode", "listen"),
    "arch_trap_jazz": None, "arch_trap_french": None,
}
ALL_CASES = (TIMER_CASES | TIMER_PLAIN_CASES | DELEGATE_CASES
             | MODE_CASES | MODE_TRAP_CASES)

# ── arch A: retired in-band tags (vendored in legacy_tags) ───────────────
A_SYSTEM = (server.SYSTEM_PROMPT + legacy_tags.DELEGATE_INSTRUCTION
            + legacy_tags.TIMER_INSTRUCTION)
A_RESPOND = server.RESPOND_PROMPT.format(camera="") + legacy_tags.MODE_SUFFIX

# ── arch B: pure speech + JSON action head (production) ──────────────────
# Production's own prompts, imported so drift is impossible: the speech
# prompt knows the capabilities EXIST (so acks sound natural and never
# contradict the head) but carries zero tag syntax.
B_SYSTEM = server.SYSTEM_PROMPT + server.CAPABILITY_NOTE + server.RESEARCH_NOTE
B_RESPOND = server.RESPOND_PROMPT.format(camera="")

# Production's head prompt and schema (actions.py), in conversation mode —
# every bench case runs in conversation mode.
HEAD_PROMPT = actions._HEAD_COMMON.format(
    mode_clause=actions._MODE_CLAUSES["conversation"])
HEAD_SCHEMA = actions.HEAD_SCHEMA

# The 1-token gate: a strictly easier question over the same cached
# prefix. A "no" skips the full head (~1.4s saved on chat turns); a gate
# miss silently skips a real action, so its recall must measure 1.0
# before the cascade ships. Kept local to this benchmark: production
# dropped the gate after this measurement — it said yes on nearly every
# turn, so it saved nothing.
GATE_PROMPT = (
    "System note (not user audio): did the user's last audio message ask "
    "the assistant to DO something — set a timer or timed reminder, "
    "switch how the assistant behaves (translate everything, just "
    "listen, back to normal), or search, look up, or find out something "
    "current like weather, news, or prices? Merely mentioning a "
    "duration or topic is not asking. Answer yes or no."
)
# Object-wrapped on purpose: a top-level bare-string enum generates EMPTY
# output when the context contains audio (llama.cpp edge, reproduced), and
# pretty-printing means the budget must fit '{ "answer": "yes" }'.
GATE_SCHEMA = {"type": "object",
               "properties": {"answer": {"type": "string",
                                         "enum": ["yes", "no"]}},
               "required": ["answer"]}


# The fast head: hand-written GBNF instead of a json_schema. Absent
# fields are OMITTED and "nothing asked" is a two-token "{}" — the
# no-action gate collapsed into the head's first token. Short keys and
# forbidden whitespace because even grammar-forced skeleton tokens each
# cost a forward pass.
FAST_GRAMMAR = r'''
root ::= "{}" | "{" timer "}" | "{" mode "}" | "{" res "}" | "{" timer "," mode "}" | "{" timer "," res "}" | "{" mode "," res "}" | "{" timer "," mode "," res "}"
timer ::= "\"seconds\": " num ", \"l\": \"" str "\""
mode ::= "\"m\": \"" ("translate" | "listen" | "conversation") "\""
res ::= "\"r\": \"" str "\""
num ::= [1-9] [0-9]?  [0-9]? [0-9]? [0-9]?
str ::= [^"\\\x00-\x1f]  [^"\\\x00-\x1f]*
'''

FAST_HEAD_PROMPT = (
    "System note (not user audio): report as compact JSON what the user "
    "just asked the assistant to DO. Omit anything they didn't ask for; "
    "output {} if nothing. \"seconds\": timer or reminder duration "
    "converted to seconds (two minutes → 120), with \"l\": a "
    "two-or-three-word label. \"m\": mode they asked to "
    "switch to (\"translate\" everything they say / \"listen\" quietly "
    "without responding / \"conversation\" = back to normal; the session "
    "is in normal conversation now). \"r\": a research, search, or "
    "look-up task, or any question about something current or changing "
    "(weather, news, prices, scores) — restated to stand alone; stable "
    "general knowledge is not research. Merely mentioning a duration or "
    "topic is not asking."
)


def chat(messages: list, *, max_tokens: int = 256, temperature: float | None = None,
         json_schema: dict | None = None, grammar: str | None = None) -> str:
    """Direct llama-server call with per-request temperature and an
    optional enforced JSON schema (llama.cpp compiles it to a grammar)."""
    body: dict = {"messages": messages, "max_tokens": max_tokens,
                  "cache_prompt": True,
                  "chat_template_kwargs": {"enable_thinking": False}}
    body["temperature"] = llama.TEMPERATURE if temperature is None else temperature
    if json_schema:
        body["response_format"] = {"type": "json_schema",
                                   "json_schema": {"schema": json_schema}}
    if grammar:
        body["grammar"] = grammar
    conn = http.client.HTTPConnection(*llama.host_port(), timeout=300)
    conn.request("POST", "/v1/chat/completions", json.dumps(body),
                 {"Content-Type": "application/json"})
    data = json.loads(conn.getresponse().read())
    conn.close()
    if "error" in data:
        raise RuntimeError(f"llama-server: {data['error']}")
    return data["choices"][0]["message"].get("content") or ""


# ── judging ───────────────────────────────────────────────────────────────

def judge_actions(acts: dict, expected: tuple | None, spoken: str) -> dict:
    """acts: {'timer': (seconds, label)|None, 'mode': str|None,
    'research': str|None} — the arch-neutral decision."""
    fired = [k for k, v in acts.items() if v]
    detected = bool(fired)
    value_ok = False
    if expected is None:
        value_ok = not detected
    elif expected[0] == "timer" and acts.get("timer"):
        value_ok = acts["timer"][0] == expected[1]
    elif expected[0] == "mode" and acts.get("mode"):
        value_ok = acts["mode"] == expected[1]
    elif expected[0] == "research" and acts.get("research"):
        value_ok = len(acts["research"].split()) >= 3  # server's guard
    return {
        "expects_action": expected is not None,
        "detected": detected,
        "recall_hit": expected is not None and detected and value_ok,
        "misfire": expected is None and detected,
        "value_ok": value_ok,
        "leaked": "<" in spoken or "##" in spoken,
        "fired": fired,
    }


def run_arch_a(wav: str) -> tuple[dict, str, dict]:
    raw = chat([{"role": "system", "content": A_SYSTEM},
                {"role": "user", "content": [audio_part(wav), text_part(A_RESPOND)]}])
    spoken, tags = legacy_tags.parse_tagged_reply(raw, ("delegate", "mode", "timer"))
    acts: dict = {"timer": None, "mode": None, "research": None}
    for name, value in tags:
        if name == "TIMER":
            seconds, label = legacy_tags.parse_timer(value)
            if seconds:
                acts["timer"] = (seconds, label)
        elif name == "MODE" and value.strip().lower() in ("translate", "listen",
                                                          "conversation"):
            acts["mode"] = value.strip().lower()
        elif name == "DELEGATE":
            acts["research"] = value.strip()
    return acts, spoken, {"raw": raw}


def run_arch_b(wav: str) -> tuple[dict, str, dict]:
    speech_msgs = [{"role": "system", "content": B_SYSTEM},
                   {"role": "user", "content": [audio_part(wav),
                                                text_part(B_RESPOND)]}]
    reply = chat(speech_msgs)
    t0 = time.time()
    head_raw = chat(speech_msgs + [{"role": "assistant", "content": reply},
                                   {"role": "user", "content": HEAD_PROMPT}],
                    max_tokens=64, temperature=0.0, json_schema=HEAD_SCHEMA)
    head_ms = round((time.time() - t0) * 1000)
    try:
        head = json.loads(head_raw)
    except ValueError:
        head = {}
    acts: dict = {"timer": None, "mode": None, "research": None}
    secs = head.get("timer_seconds") or 0
    if 0 < secs <= server.MAX_TIMER_S:
        acts["timer"] = (secs, head.get("timer_label", ""))
    # "conversation" while already in conversation is the no-op the real
    # server's switch_mode would make of it (every bench case runs in
    # conversation mode) — state-reporting, not an action.
    if head.get("mode") in ("translate", "listen"):
        acts["mode"] = head["mode"]
    if (head.get("research_task") or "").strip():
        acts["research"] = head["research_task"].strip()
    # B's speakable text is the whole reply minus the transcript line —
    # there is nothing to excise, which is the point.
    spoken = reply.split("\n", 1)[1] if "\n" in reply else reply
    return acts, spoken, {"raw": reply, "head": head_raw, "head_ms": head_ms}


def run_arch_b_fast(wav: str) -> tuple[dict, str, dict]:
    """The fast head: omit-absent compact GBNF ('{}' when nothing asked)
    + a tightened instruction. Same decision, a fraction of the tokens."""
    speech_msgs = [{"role": "system", "content": B_SYSTEM},
                   {"role": "user", "content": [audio_part(wav),
                                                text_part(B_RESPOND)]}]
    reply = chat(speech_msgs)
    t0 = time.time()
    head_raw = chat(speech_msgs + [{"role": "assistant", "content": reply},
                                   {"role": "user", "content": FAST_HEAD_PROMPT}],
                    max_tokens=64, temperature=0.0, grammar=FAST_GRAMMAR)
    head_ms = round((time.time() - t0) * 1000)
    try:
        head = json.loads(head_raw)
    except ValueError:
        head = {}
    acts: dict = {"timer": None, "mode": None, "research": None}
    secs = head.get("seconds") or 0
    if 0 < secs <= server.MAX_TIMER_S:
        acts["timer"] = (secs, head.get("l", ""))
    if head.get("m") in ("translate", "listen"):
        acts["mode"] = head["m"]
    if (head.get("r") or "").strip():
        acts["research"] = head["r"].strip()
    spoken = reply.split("\n", 1)[1] if "\n" in reply else reply
    return acts, spoken, {"raw": reply, "head": head_raw, "head_ms": head_ms}


def run_arch_b_gated(wav: str) -> tuple[dict, str, dict]:
    """The cascade: speech → 1-token gate → full head only on 'yes'."""
    speech_msgs = [{"role": "system", "content": B_SYSTEM},
                   {"role": "user", "content": [audio_part(wav),
                                                text_part(B_RESPOND)]}]
    reply = chat(speech_msgs)
    tail = [{"role": "assistant", "content": reply}]
    t0 = time.time()
    verdict = chat(speech_msgs + tail + [{"role": "user", "content": GATE_PROMPT}],
                   max_tokens=16, temperature=0.0, json_schema=GATE_SCHEMA)
    gate_ms = round((time.time() - t0) * 1000)
    try:
        gate_yes = json.loads(verdict).get("answer") == "yes"
    except ValueError:
        gate_yes = True  # unparseable gate → fail open into the full head
    acts: dict = {"timer": None, "mode": None, "research": None}
    extra: dict = {"raw": reply, "gate": verdict, "gate_ms": gate_ms,
                   "gate_yes": gate_yes}
    if gate_yes:
        t0 = time.time()
        head_raw = chat(speech_msgs + tail + [{"role": "user", "content": HEAD_PROMPT}],
                        max_tokens=64, temperature=0.0, json_schema=HEAD_SCHEMA)
        extra["head"] = head_raw
        extra["head_ms"] = round((time.time() - t0) * 1000)
        try:
            head = json.loads(head_raw)
        except ValueError:
            head = {}
        secs = head.get("timer_seconds") or 0
        if 0 < secs <= server.MAX_TIMER_S:
            acts["timer"] = (secs, head.get("timer_label", ""))
        if head.get("mode") in ("translate", "listen"):
            acts["mode"] = head["mode"]
        if (head.get("research_task") or "").strip():
            acts["research"] = head["research_task"].strip()
    spoken = reply.split("\n", 1)[1] if "\n" in reply else reply
    return acts, spoken, extra


def score(results: list[dict]) -> dict:
    pos = [r for r in results if r["expects_action"]]
    neg = [r for r in results if not r["expects_action"]]
    head_ms = [r["head_ms"] for r in results if "head_ms" in r]
    out = {
        "recall": round(sum(r["recall_hit"] for r in pos) / len(pos), 3),
        "misfire": round(sum(r["misfire"] for r in neg) / len(neg), 3),
        "leaked": sum(r["leaked"] for r in results),
        "n": len(results),
    }
    if head_ms:
        head_ms.sort()
        out["head_ms_p50"] = head_ms[len(head_ms) // 2]
        out["head_ms_max"] = head_ms[-1]
    if any("gate_yes" in r for r in results):
        # gate_recall must be 1.0 to ship the cascade: a gate miss on a
        # positive silently skips a real action. A gate "yes" on a
        # negative only costs one head call — report, don't fail.
        out["gate_recall"] = round(sum(r["gate_yes"] for r in pos) / len(pos), 3)
        out["gate_yes_on_neg"] = round(sum(r["gate_yes"] for r in neg) / len(neg), 3)
        gms = sorted(r["gate_ms"] for r in results)
        out["gate_ms_p50"] = gms[len(gms) // 2]
    return out


RUNNERS = {"A_tags": run_arch_a, "B_head": run_arch_b,
           "B_fast": run_arch_b_fast,
           "B_gated": run_arch_b_gated}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repeat", type=int, default=2)
    ap.add_argument("--archs", default="A_tags,B_head",
                    help=f"comma-separated subset of {', '.join(RUNNERS)}")
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    wavs = ensure_fixtures(ALL_CASES)
    llama.start()
    try:
        out = {"model": llama.MODEL, "speech_temperature": llama.TEMPERATURE,
               "repeat": args.repeat, "archs": {}}
        for label, runner in [(a, RUNNERS[a]) for a in args.archs.split(",")]:
            results = []
            for name, wav in wavs.items():
                for _ in range(args.repeat):
                    t0 = time.time()
                    acts, spoken, extra = runner(wav)
                    r = judge_actions(acts, EXPECTED[name], spoken)
                    r.update({"case": name, **extra})
                    results.append(r)
                    ok = "✓" if (r["value_ok"] and not r["leaked"]) else "✗"
                    print(f"{ok} [{label}] {name}: {r['fired'] or 'none'}"
                          f"{' LEAKED' if r['leaked'] else ''}"
                          f" ({time.time() - t0:.1f}s"
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
