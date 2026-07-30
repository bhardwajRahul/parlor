# Handoff: latency rebuild + llama.cpp port

Branch: `perf-latency`. This doc is the state of the world, what is verified,
and what still needs human testing. Delete this file before merging.

## What changed (commit order tells the story)

1. **E2E benchmark harness** (`src/benchmarks/`) — real synthesized speech
   fixtures, latency suite, JSON results, `compare.py`.
2. **Streaming pipeline** — the `respond_to_user` tool call (which silently
   cost a second full inference round-trip per turn) replaced by streamed
   decoding: response sentences go to TTS while the model still generates,
   transcript is produced last. VAD silence cutoff 600ms → 200ms.
3. **Speculative prefill** — the camera frame is processed while the user is
   still talking; after the llama.cpp port, the speech itself also streams in
   ~3s chunks through the prompt cache.
4. **llama.cpp port** — litert-lm fully replaced by a spawned `llama-server`
   (official Google QAT q4_0 GGUF + mmproj). The server owns conversation
   history; prefix caching makes re-sending it cheap. Real barge-in abort.
   litert-lm can be restored by reverting two commits (`2284193`, `37caaaf`).
5. **Turn detection** — judgment belongs to pipecat's smart-turn-v3.2 audio
   classifier (~20ms CPU), and the LLM prompt carries no format instructions
   at all. The inline-marker and separate-request variants were measured
   (`benchmarks/turnbench.py`) at chance accuracy on E2B, E4B *and* 12B, so
   only the classifier path remains.
6. **Live-session quality fixes** (after quality regressions vs main were
   reproduced with degraded fixtures):
   - Audio that stops abruptly at the VAD cutoff makes the llama.cpp audio
     encoder hallucinate a confident completion of the last word ("capital
     of France?" clipped 180ms → "capital of the United States", answered
     Washington). 300ms of silence appended INSIDE the final WAV fully
     recovers the transcript. This was the main "transcript seems off /
     answer seems wrong" failure mode at the 200ms VAD cutoff.
   - False `incomplete` holds (e.g. other voices) left finished questions
     unanswered. The canned nudge is replaced by a **flush**: after 2.5s of
     continued silence the client asks the server to answer the held audio;
     the model itself either answers or asks the user to continue.
   - `temperature` 0 → 0.7: responses no longer repeat verbatim;
     transcripts stay WER 0.0 on the clean fixtures.
   - The transcript tag is parsed tolerantly (`### TRANSCRIPT:` with a
     space was silently dropping correct transcripts).
   - The transcript line now LEADS the reply. Measured (3 reps/cell, temp
     0.7): trailing transcripts paraphrase long utterances (WER 0.39 on a
     clean 33-word question vs 0.00 leading), and leading fully recovers
     clipped endings. Grammar-forced JSON `{transcript, response}` (≈
     main's tool call) was also measured: format breaks 1-3/3 on degraded
     audio and 3/3 on chunked — structured output stays rejected. An
     XML-style `<transcript>…</transcript>` scored identically to
     `###TRANSCRIPT:` (24/24 format-intact each, same WER), so the hash
     tag stays: its parser is unit-tested across delta boundaries and it
     costs ~4 fewer tokens per turn. Cost of leading: the transcript's
     decode time (~0.2s short / ~0.7s long) before first audio; the
     client shows the heard words as soon as the line arrives.
7. **Automated e2e suite** (`src/tests/`, `uv run pytest`) — covers
   turn-taking, chunked overlap, camera grounding/freshness, transcript WER
   (clean + degraded audio), memory, robustness (glitches, interrupts,
   queued turns, rotation, llama-server death). 19 tests, ~1 min + model
   load. `PARLOR_TEST_URL` runs it against a live server.

8. **Model size switch** — `MODEL=e2b|e4b|12b` picks among Google's QAT
   q4_0 GGUFs (E4B default — live testing found its answers noticeably
   better and the ~1.8x latency still ~3x faster than baseline). The llama.cpp chat template was verified
   current: the GGUF embeds the 2026-07-09 canonical template
   byte-identical to the upstream tool-calling fix
   (google/gemma-4-E2B-it#35), and llama-server ≥b10150 applies it via
   jinja by default — nothing to update.

## Measured (M3 Pro, `benchmarks/results/`)

End of utterance → first audio heard; add ~200ms VAD on top. Baseline is the
pre-session litert build; "Now" includes the leading transcript's decode
time (`transcript_first.json` vs `after_quality_fixes.json` isolates that
cost: +0.1s short, +0.7-0.8s long — bought back as transcript accuracy, and
total turn time on long questions actually improved since TTS overlaps the
remaining decode). E4B (`e4b_latency.json`): ~1.8x E2B — a viable quality
fallback, still ~3x faster than baseline.

| Turn                        | Baseline | Now (E2B) |
| --------------------------- | -------- | --------- |
| Short question              | 1.52s    | ~0.7s     |
| Short + camera              | 1.94s    | ~0.8s     |
| Long question (9.4s speech) | 2.91s    | ~1.3-1.4s |
| Long + camera               | 2.98s    | ~1.5s     |

Reproduce: `uv run server.py`, then
`uv run python benchmarks/bench.py --label X --out benchmarks/results/X.json`.

## Manual test checklist (what pytest cannot judge)

Run with logs captured: `uv run python server.py 2>&1 | tee /tmp/parlor.log`.
Hard-refresh the browser (Cmd+Shift+R) after every server restart.

### A. Turn-taking feel (real prosody, real mic)

- [ ] Finish a sentence cleanly → response starts in well under a second.
- [ ] Trail off mid-sentence ("So what I wanted to ask is…") → stays quiet,
      log shows `p(complete)` near 0; ~2.5s later the flush turn arrives and
      the model asks you to continue (not a canned line).
- [ ] "Hmm, let me think about that…" with genuine hesitation tone → stays
      quiet. **Watch the `p(complete)` values** — if your speaking style
      lands on the wrong side, the 0.5 threshold in `turn_detector.py` is
      the tuning knob.
- [ ] Continue after an incomplete pause → the eventual answer accounts for
      BOTH parts, and the transcript shows the whole thing.
- [ ] Natural fast back-and-forth feels right at 200ms VAD; if it clips
      you, `redemptionMs` in static/app.js.

### B. Echo and barge-in (browser + speakers)

- [ ] Speakers at normal volume: never answers its own voice (check
      `heard:` lines for its own phrasings).
- [ ] Speakers loud: same. (Two layers: the 250ms sustained-speech gate
      and the echo rule in the prompt. Chrome's AEC route is deliberately
      OFF — on macOS it engages system voice processing that colors the
      TTS voice per turn and suppresses the user's mic during playback,
      which killed barge-in. Headphones sidestep echo entirely.)
- [ ] Deliberate barge-in mid-reply: stops within a beat and handles what
      you said next.
- [ ] Barge-in within the first ~800ms of it speaking is intentionally
      ignored (echo grace period) — confirm that feels okay.

### C. Quality (the original complaint — judge vs main by feel)

- [ ] Transcripts of YOUR real voice are accurate, including the last word
      of each utterance (the padding fix targets exactly this).
- [ ] Responses feel at least as good as main (E4B is the default now;
      `MODEL=12b` is the next step up if quality still lacks, `MODEL=e2b`
      the fast fallback).
- [ ] **Multilingual** (the Bule-AI use case): speak Indonesian or another
      language → understanding and transcript quality. Kokoro voice
      `af_heart` is English; non-English TTS output is a known gap.
      smart-turn-v3 is trained on 23 languages, but verify turn-taking
      feels right in non-English speech.

### D. Platforms (untested — needs hardware)

- [ ] **Linux**: llama.cpp installed manually (no brew); TTS falls back to
      kokoro-onnx; `onnxruntime` and vad-web CDN paths. Entirely unverified.
- [ ] Safari / Firefox: vad-web and audio autoplay policies behave
      differently. Chrome is the only tested browser.
- [ ] Lower-RAM Macs: model + mmproj ≈ 4GB + TTS; 8GB machines are dubious.

## Known limitations / accepted trade-offs

- Image-only turns sometimes invent a "transcript" of the instruction text
  (cosmetic; real clients always send audio).
- llama-server output goes to DEVNULL; un-silence in `llama.py` when
  debugging.
- llama.cpp marks Gemma audio input "experimental stage".
- Context-size guard uses a token *estimate* (`estimate_tokens`), not exact
  counts.
- The `thinking_pause` fixture cannot test hesitation: TTS gives it
  finished-sounding prosody, which the acoustic classifier correctly reads
  as complete (skipped test documents this). Judge from live speech.

## Upstream issues worth filing

- **litert-lm**: `cancel_process()` permanently wedges the Conversation or
  Session it is called on (engine survives; next send never returns).
- **mlx-vlm**: (1) more than one audio segment per turn crashes the feature
  extractor (paths never decoded); (2) `multimodal_token_ids_from_config`
  omits `audio_token_id`, so the prompt-cache media guard silently fails to
  fire for audio — a corruption bug once multi-audio works.

## Experiment record

`src/benchmarks/experiment_*.py` + commit messages document every dead end
with numbers: session-level litert prefill (infeasible), chunked audio
across closed turns (destroys transcript), MLX overlap (blocked upstream),
LLM turn markers (all five prompt variants), visual_token_budget
(hallucinates). Read these before re-attempting any of them.
