# Changelog

## [2.0.0] — 2026-08-03

Parlor v2 is a near-complete rebuild of the pipeline toward one goal: GPT-Live-class conversation, fully on-device. Inference moved from LiteRT-LM to llama.cpp, turn-taking moved from LLM judgment to a dedicated audio classifier, and actions moved out of the speech stream into a grammar-forced JSON head — each decision measured before it was made (see `benchmarks/`).

### New features

- **Background research.** "Find the best pizza in Rome right now" hands the task to a frontier model on any OpenAI-compatible endpoint (OpenRouter + web search by default) while the conversation continues; the answer is woven back in as a spoken delivery at the next idle moment. Off unless `REASONER_API_KEY` is set — without it Parlor stays fully on-device.
- **Timers.** "Set a timer for three minutes for the pasta" — the server owns the clock (a turn-based model can't ring into silence; `benchmarks/timerprobe.py`), the model announces the ring in any mode, and a countdown chip with a cancel button tracks it.
- **Live translation mode.** "Translate everything I say into English" turns Parlor into a consecutive interpreter: each utterance rendered after a short silence, no conversational replies, until you say "stop translating" or hit the stop chip.
- **Just-listen mode.** "Just listen for a while, I want to think out loud" makes it a silent scribe: every utterance transcribed on screen, nothing spoken back, until you address it again.
- **A sense of time.** The model is told how much quiet preceded a turn, how long research took, and when the session started — so "how long was I gone?" gets a real answer.
- **Turn-taking you can see.** While an utterance is held mid-thought, its bubble shows the classifier's confidence ("sounds unfinished (12%) — still listening").

### Engine and architecture

- **llama.cpp backend**, replacing LiteRT-LM. Runs Google's official Gemma 4 QAT q4_0 GGUFs through a spawned llama-server and its prefix cache, which unlocks speech-overlap prefill: audio chunks and the camera frame are pushed through the cache *while you're still talking*, so long questions start answering almost as fast as short ones. Barge-in aborts generation server-side. Startup discovers both `llama-server` and the newer unified `llama` binary, and enforces the build floor Gemma 4 audio needs (b9503+, b9512 for `MODEL=12b`) with install-method-aware upgrade hints.
- **Model switch.** `MODEL=e2b|e4b|12b` picks the Gemma 4 size; the default is now E4B (noticeably better answers, ~1.8x E2B latency, ~6 GB RAM).
- **smart-turn-v3 turn detection**, replacing LLM FINISHED/WAIT judgment. `benchmarks/turnbench.py` on labelled human speech: the 8M-parameter classifier scores 0.96 accuracy at 19ms; every Gemma variant is at or near chance while costing 0.6–3.6s. Incomplete utterances are held and merged into the next turn; a false hold is answered anyway after 2.5s of continued silence.
- **Decoupled JSON action head**, replacing in-band control tags. The spoken reply is now pure speech; timers, mode switches, and research requests are decided by a second grammar-forced JSON request over the same prompt cache, hidden under TTS playback. `benchmarks/archbench.py`: head recall 1.0 vs tags 0.955 — and the tags' miss is a spoken promise the server never keeps, the worst failure a voice assistant has. Markup leaking into TTS is now structurally impossible. Client and wire protocol unchanged.
- **Transcript-first streaming.** The reply opens with a transcript of what was heard (leading beats trailing: WER 0.00 vs 0.39 on long utterances), shown on screen ~0.3s in, then is spoken sentence-by-sentence while still generating.
- **Context.** 32k window, with rotation driven by real `prompt_tokens` from llama-server instead of an estimate, dropping whole exchanges — no more silent truncation-induced "forgetting".

### Latency (Apple M3 Pro, `MODEL=e2b`, end of utterance → first audio)

| Turn | v1.0.0 | v2.0.0 |
| --- | --- | --- |
| Short question (~2s speech) | ~1.5s | ~0.7s |
| Long question (~9s speech) | ~2.9s | ~1.3s |
| Short question + camera | ~1.9s | ~0.8s |

### Reliability

- Non-speech audio (a breath, a cough, room noise) can no longer become invented user words, answer itself, or switch modes — the prompts sanction "(no speech)" and annotation-shaped transcripts are rejected.
- Instruction text can never be displayed as the user's words or spoken aloud; research deliveries quote the reasoner's answer verbatim instead of paraphrasing away the key fact.
- Research results and timer rings are delivered only in playback gaps — the assistant never interrupts itself mid-sentence.
- Barge-in actually fires on a live mic (sliding-window voicing gate instead of a consecutive-frames counter that consonants kept resetting).
- 300ms of tail padding recovers VAD-clipped word endings that made the audio encoder hallucinate confident wrong completions.
- Only actions that actually fired enter history, so a dropped action can't teach the model a state the server isn't in.
- A mic-glitch WAV can no longer poison history and 400 every later request; served on localhost so the page gets the secure context mic and camera access requires.

### Project and developer experience

- Standard uv src layout: `uv sync && uv run parlor` is the whole quick start.
- End-to-end test suite (`uv run pytest`, ~87 tests) spawns the real server — llama.cpp, TTS, turn detector — and drives it over WebSocket with synthesized speech, including degraded audio (clipped endings, noise, other voices) reproducing live-mic failure modes.
- The decisions above stay reproducible in `benchmarks/`: `bench` (latency), `turnbench`, `archbench`, `tagbench`, `camerabench`, `timerprobe`, with retired architectures vendored so baselines re-run against future models.
- README rewritten for the v2 architecture; the full configuration inventory moved to `docs/configuration.md`.

### Breaking changes

- The LiteRT-LM backend is gone. llama.cpp b9503+ is required (`brew install llama.cpp` on macOS).
- `TURN_MODE` is removed — the smart-turn classifier is the only turn-taking path.
- The repo is reorganized into a `src/parlor` package; run it with `uv run parlor` instead of invoking `server.py` directly.
- The default model is now E4B (~6 GB RAM); set `MODEL=e2b` for the previous footprint.

## [1.0.0] — 2026-04-07

Initial release: on-device voice-and-vision conversations with Gemma via LiteRT-LM, Kokoro text-to-speech, and a browser client over WebSocket.

[2.0.0]: https://github.com/fikrikarim/parlor/compare/v1.0.0...v2.0.0
[1.0.0]: https://github.com/fikrikarim/parlor/releases/tag/v1.0.0
