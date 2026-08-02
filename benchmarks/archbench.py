"""Action-architecture benchmark: in-band control tags (production) vs a
decoupled JSON action head (a second request over the cached prefix).

The question: should actions (timer, mode switch, research delegation)
ride the speech stream as tags the TagFilter must excise — or be decided
by a separate grammar-forced JSON request that can never be spoken, runs
at temperature 0 regardless of speech temperature, and sees the model's
own reply as evidence?

    Arch A (tags):  production system prompt + MODE_SUFFIX; reply parsed
                    by the real StreamParser; tag values through
                    server.parse_timer. What ships today.
    Arch B (head):  speech request with a light capability note and NO
                    tag machinery; then a second request — same cached
                    prefix + the reply + a decider instruction — forced
                    to a flat JSON schema at temp 0. The model itself
                    converts "twenty minutes" to 1200 seconds.

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
from tagbench import TIMER_CASES, TIMER_PLAIN_CASES, ensure_fixtures  # noqa: E402
from parlor import llama  # noqa: E402
from parlor.pipeline import StreamParser, audio_part, text_part  # noqa: E402
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
    "arch_trap_jazz": None, "arch_trap_french": None,
}
ALL_CASES = (TIMER_CASES | TIMER_PLAIN_CASES | DELEGATE_CASES
             | MODE_CASES | MODE_TRAP_CASES)

# ── arch A: production in-band tags ──────────────────────────────────────
A_SYSTEM = (server.SYSTEM_PROMPT + server.DELEGATE_INSTRUCTION
            + server.TIMER_INSTRUCTION)
A_RESPOND = server.RESPOND_PROMPT.format(camera="") + server.MODE_SUFFIX

# ── arch B: pure speech + JSON action head ───────────────────────────────
# The speech prompt knows the capabilities EXIST (so acks sound natural
# and never contradict the head) but carries zero tag syntax.
B_SYSTEM = (server.SYSTEM_PROMPT
            + " You can also set countdown timers, hand research tasks to a "
              "background assistant with web access, translate everything "
              "the user says from now on, or just listen quietly without "
              "responding. When the user asks for one of these, confirm "
              "briefly and naturally — the system takes care of making it "
              "happen.")
B_RESPOND = server.RESPOND_PROMPT.format(camera="")

HEAD_PROMPT = (
    "System note (not user audio): you are the action decider. From the "
    "user's last audio message (the assistant's reply above may help), "
    "report what the user asked the assistant to DO, as JSON. "
    "timer_seconds: the countdown duration in seconds if they asked for a "
    "timer or a timed reminder, else 0. timer_label: a two-or-three-word "
    "label for it, else empty. mode: the mode they asked to SWITCH TO — "
    "'translate' (translate everything from now on), 'listen' (just "
    "listen quietly, don't respond), or 'conversation' (back to normal) — "
    "the session is already in normal conversation, so mode is 'none' "
    "unless they asked to change it. research_task: if they asked to "
    "search, look up, or research something, or asked about anything "
    "current or changing (weather, news, prices, scores, openings, "
    "\"right now\", \"today\"), the task restated to stand alone, else "
    "empty. A duration or topic merely mentioned in passing is NOT a "
    "request: report an action only when the user asked for it."
)
HEAD_SCHEMA = {
    "type": "object",
    "properties": {
        "timer_seconds": {"type": "integer"},
        "timer_label": {"type": "string"},
        "mode": {"type": "string",
                 "enum": ["none", "translate", "listen", "conversation"]},
        "research_task": {"type": "string"},
    },
    "required": ["timer_seconds", "timer_label", "mode", "research_task"],
}


def chat(messages: list, *, max_tokens: int = 256, temperature: float | None = None,
         json_schema: dict | None = None) -> str:
    """Direct llama-server call with per-request temperature and an
    optional enforced JSON schema (llama.cpp compiles it to a grammar)."""
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
    return data["choices"][0]["message"].get("content") or ""


# ── judging ───────────────────────────────────────────────────────────────

def judge_actions(actions: dict, expected: tuple | None, spoken: str) -> dict:
    """actions: {'timer': (seconds, label)|None, 'mode': str|None,
    'research': str|None} — the arch-neutral decision."""
    fired = [k for k, v in actions.items() if v]
    detected = bool(fired)
    value_ok = False
    if expected is None:
        value_ok = not detected
    elif expected[0] == "timer" and actions.get("timer"):
        value_ok = actions["timer"][0] == expected[1]
    elif expected[0] == "mode" and actions.get("mode"):
        value_ok = actions["mode"] == expected[1]
    elif expected[0] == "research" and actions.get("research"):
        value_ok = len(actions["research"].split()) >= 3  # server's guard
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
    p = StreamParser(expect_transcript=True,
                     control_tags=("delegate", "mode", "timer"))
    spoken = p.feed(raw)
    tail, _ = p.finalize()
    actions: dict = {"timer": None, "mode": None, "research": None}
    for name, value in p.tags:
        if name == "TIMER":
            seconds, label = server.parse_timer(value)
            if seconds:
                actions["timer"] = (seconds, label)
        elif name == "MODE" and value.strip().lower() in ("translate", "listen",
                                                          "conversation"):
            actions["mode"] = value.strip().lower()
        elif name == "DELEGATE":
            actions["research"] = value.strip()
    return actions, " ".join(spoken + tail), {"raw": raw}


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
    actions: dict = {"timer": None, "mode": None, "research": None}
    secs = head.get("timer_seconds") or 0
    if 0 < secs <= server.MAX_TIMER_S:
        actions["timer"] = (secs, head.get("timer_label", ""))
    # "conversation" while already in conversation is the no-op the real
    # server's switch_mode would make of it (every bench case runs in
    # conversation mode) — state-reporting, not an action.
    if head.get("mode") in ("translate", "listen"):
        actions["mode"] = head["mode"]
    if (head.get("research_task") or "").strip():
        actions["research"] = head["research_task"].strip()
    # B's speakable text is the whole reply minus the transcript line —
    # there is nothing to excise, which is the point.
    spoken = reply.split("\n", 1)[1] if "\n" in reply else reply
    return actions, spoken, {"raw": reply, "head": head_raw, "head_ms": head_ms}


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
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repeat", type=int, default=2)
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    wavs = ensure_fixtures(ALL_CASES)
    llama.start()
    try:
        out = {"model": llama.MODEL, "speech_temperature": llama.TEMPERATURE,
               "repeat": args.repeat, "archs": {}}
        for label, runner in (("A_tags", run_arch_a), ("B_head", run_arch_b)):
            results = []
            for name, wav in wavs.items():
                for _ in range(args.repeat):
                    t0 = time.time()
                    actions, spoken, extra = runner(wav)
                    r = judge_actions(actions, EXPECTED[name], spoken)
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
