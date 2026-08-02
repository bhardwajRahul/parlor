"""The retired in-band control-tag architecture, vendored for benchmarks.

Production replaced in-band tags with the decoupled JSON action head
(src/parlor/actions.py) after benchmarks/archbench.py measured the head
at recall 1.0 vs 0.955 for tags — with the tag miss being a spoken
promise the server never keeps ("I will be quiet", no tag). Like
turnbench keeps its losing FINISHED/WAIT prompts, this module keeps the
losing architecture reproducible: tagbench.py still measures it, and
archbench.py scores it as the baseline.

Contents are verbatim from the last production commit that shipped them:
the tag instructions, the TagFilter stream excisor, and the English
word-number duration parser the head made obsolete (the head outputs
integer seconds directly, in any language).
"""

import re

DELEGATE_INSTRUCTION = (
    " You also have a background research assistant with web access. When "
    "the user asks you to search, look up, find, or research something, or "
    "asks about anything current or changing (weather, news, prices, "
    "scores, openings, \"right now\", \"today\"), you MUST hand the task "
    "over instead of answering from memory — your knowledge is stale and a "
    "guess is worse than handing over. Sports, elections, and rankings "
    "count as current. To hand over: say one short "
    "sentence telling the user you're on it, then append <delegate>the "
    "task, restated to stand alone</delegate> — never speak or mention "
    "that tag; the result arrives later and you can share it then. "
    "Everything else, answer yourself and don't use the tag."
)

TIMER_INSTRUCTION = (
    " You can also set countdown timers — but ONLY the timer tag actually "
    "sets one; a spoken promise alone sets nothing and the user would "
    "wait forever. When the user asks for a timer, or to be reminded "
    "after some amount of time, you MUST end your reply with the tag. "
    "Say one short confirmation sentence, then append "
    "<timer>the duration | a two-or-three-word label</timer>. Example "
    "reply: Three minutes — I'll let you know. "
    "<timer>3 minutes | pasta</timer> — never speak or mention the tag "
    "itself. The system tells you when it goes off, and you announce it "
    "to the user then. If they give no duration, ask for one instead. "
    "Never use the tag for anything else."
)

MODE_SUFFIX = (
    " If the audio asks you to translate everything they say from now on, "
    "confirm briefly and end with <mode>translate</mode>. If it asks you "
    "to just listen quietly for a while and not respond, confirm briefly "
    "and end with <mode>listen</mode>."
)


class TagFilter:
    """Streams response text through, extracting complete
    '<name>value</name>' control elements into .tags — the production
    excisor of the tag era, verbatim."""

    def __init__(self, names: tuple[str, ...]):
        alt = "|".join(names)  # names are plain lowercase words
        self._re = re.compile(
            rf"<\s*(?P<name>{alt})\s*>\s*(?P<value>.*?)\s*<\s*/\s*(?P=name)\s*>",
            re.IGNORECASE | re.DOTALL)
        self._unclosed_re = re.compile(
            rf"^<\s*(?P<name>{alt})\s*>\s*(?P<value>\S.*?)\s*$",
            re.IGNORECASE | re.DOTALL)
        self._open_re = re.compile(rf"^<\s*(?:{alt})\s*>", re.IGNORECASE)
        self._miss_re = re.compile(rf"^<\s*/?\s*(?:{alt})\b", re.IGNORECASE)
        self._forming_re = re.compile(r"^<[\s/]*([a-zA-Z]*)\s*$")
        self._names = [n.lower() for n in names]
        self._held = ""
        self._dead = False
        self.tags: list[tuple[str, str]] = []  # (NAME, value), stream order

    def feed(self, delta: str) -> str:
        if self._dead:
            return ""
        buf = self._held + delta
        self._held = ""
        out = []
        while buf:
            lt = buf.find("<")
            if lt == -1:
                out.append(buf)
                break
            out.append(buf[:lt])
            m = self._re.match(buf, lt)
            if m:
                self.tags.append((m.group("name").upper(), m.group("value").strip()))
                out.append("\n")  # keep a boundary where the element sat
                buf = buf[m.end():]
                continue
            rest = buf[lt:]
            forming = self._forming_re.match(rest)
            if self._open_re.match(rest) or (
                    forming and any(n.startswith(forming.group(1).lower())
                                    for n in self._names)):
                self._held = rest  # a clean element may still complete
                break
            if self._miss_re.match(rest):
                self._dead = True  # markup, not speech — nothing more is spoken
                break
            out.append("<")  # literal '<', not ours
            buf = buf[lt + 1:]
        return "".join(out)

    def finalize(self) -> None:
        m = self._unclosed_re.match(self._held)
        if m and not self._dead:
            self.tags.append((m.group("name").upper(), m.group("value").strip()))
        self._held = ""


TRANSCRIPT_TAG_RE = re.compile(r"#{2,}[ \t]*TRANSCRIPT[ \t]*:[ \t]*", re.IGNORECASE)


def parse_tagged_reply(raw: str, names: tuple[str, ...]) -> tuple[str, list]:
    """Whole-reply (batch) reconstruction of the tag-era pipeline:
    (spoken text, extracted tags). Transcript line consumed, tags excised
    by the TagFilter, ##-markup cut terminal — enough for scoring; the
    delta-boundary behavior the old unit suite pinned is not reproduced."""
    m = TRANSCRIPT_TAG_RE.search(raw)
    body = raw[m.end():] if m else raw
    if m:
        body = body.lstrip()
        newline = body.find("\n")
        body = body[newline + 1:] if newline != -1 else ""
    f = TagFilter(names)
    spoken = f.feed(body)
    f.finalize()
    return re.split(r"#{2,}", spoken)[0].strip(), f.tags


_WORD_NUMS = {
    "a": 1, "an": 1, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
    "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10, "eleven": 11,
    "twelve": 12, "thirteen": 13, "fourteen": 14, "fifteen": 15,
    "sixteen": 16, "seventeen": 17, "eighteen": 18, "nineteen": 19,
    "twenty": 20, "thirty": 30, "forty": 40, "fifty": 50, "sixty": 60,
    "seventy": 70, "eighty": 80, "ninety": 90,
}
_UNIT_S = {"h": 3600, "m": 60, "s": 1}
TIMER_VALUE_RE = re.compile(
    r"(?P<num>\d+(?:\.\d+)?|[a-z]+(?:-[a-z]+)?)\s*"
    r"(?P<unit>hours?|hrs?|minutes?|mins?|seconds?|secs?)\b",
    re.IGNORECASE)


def _word_number(raw: str) -> int:
    if raw in _WORD_NUMS:
        return _WORD_NUMS[raw]
    tens, _, units = raw.partition("-")
    t, u = _WORD_NUMS.get(tens, 0), _WORD_NUMS.get(units, 0)
    return t + u if t >= 20 and t % 10 == 0 and 0 < u < 10 else 0


def _duration_s(text: str) -> tuple[float, str] | None:
    m = re.search(r"\bhalf an hour\b", text, re.IGNORECASE)
    if m:
        return 1800.0, (text[:m.start()] + text[m.end():]).strip()
    for m in TIMER_VALUE_RE.finditer(text):
        raw = m.group("num").lower()
        try:
            num = float(raw)
        except ValueError:
            num = _word_number(raw)
        if num > 0:
            rest = (text[:m.start()] + text[m.end():]).strip()
            rest = re.sub(r"^(?:for|in)\b\s*(?:the\b\s*)?", "", rest,
                          flags=re.IGNORECASE)
            return num * _UNIT_S[m.group("unit")[0].lower()], rest
    return None


def parse_timer(value: str) -> tuple[float | None, str]:
    """'3 minutes | pasta' → (180.0, 'pasta') — the tag-era duration parser."""
    segments = [s.strip() for s in value.split("|")]
    for i, seg in enumerate(segments):
        parsed = _duration_s(seg)
        if parsed is None:
            continue
        seconds, rest = parsed
        others = [s for j, s in enumerate(segments) if j != i and s]
        label = " ".join(others) if others else rest
        return seconds, re.sub(r"^\W+|\W+$", "", label)
    return None, ""
