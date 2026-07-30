# Parlor

On-device, real-time multimodal AI. Have natural voice and vision conversations with an AI that runs entirely on your machine.

Parlor uses [Gemma 4 E2B](https://huggingface.co/google/gemma-4-E2B-it) for understanding speech and vision, and [Kokoro](https://huggingface.co/hexgrad/Kokoro-82M) for text-to-speech. You talk, show your camera, and it talks back, all locally.

https://github.com/user-attachments/assets/cb0ffb2e-f84f-48e7-872c-c5f7b5c6d51f

> **Research preview.** This is an early experiment. Expect rough edges and bugs.

# Why?

I'm [self-hosting a totally free voice AI](https://www.fikrikarim.com/bule-ai-initial-release/) on my home server to help people learn speaking English. It has hundreds of monthly active users, and I've been thinking about how to keep it free while making it sustainable.

The obvious answer: run everything on-device, eliminating any server cost. Six months ago I needed an RTX 5090 to run just the voice models in real-time.

Google just released a super capable small model that I can run on my M3 Pro in real-time, with vision too! Sure you can't do agentic coding with this, but it is a game-changer for people learning a new language. Imagine a few years from now that people can run this locally on their phones. They can point their camera at objects and talk about them. And this model is multi-lingual, so people can always fallback to their native language if they want. This is essentially what OpenAI demoed a few years ago.

## How it works

```
Browser (mic + camera)
    │
    │  WebSocket (audio PCM + JPEG frames)
    ▼
FastAPI server
    ├── Gemma 4 E2B via llama.cpp (QAT q4_0)  →  understands speech + vision
    └── Kokoro TTS (MLX on Mac, ONNX on Linux)  →  speaks back
    │
    │  WebSocket (streamed audio chunks)
    ▼
Browser (playback + transcript)
```

- **Voice Activity Detection** in the browser ([Silero VAD](https://github.com/ricky0123/vad)). Hands-free, no push-to-talk, with a short 200ms silence cutoff for fast turn-taking.
- **Turn-completeness filtering.** Pipecat's [smart-turn-v3](https://huggingface.co/pipecat-ai/smart-turn-v3) audio classifier (~20ms on CPU) judges whether you finished your thought before the LLM answers — if you were cut off or paused to think, it stays quiet and lets you continue. If you then stay silent, the held audio is flushed to the model, which either answers it or warmly asks you to finish.
- **Streaming decode → TTS.** The response is spoken sentence-by-sentence while the model is still generating, and the transcript is generated last so it never delays audio.
- **Speculative prefill during speech.** The camera frame is sent the moment you start speaking, and your speech itself streams to the server in ~3s chunks — both are pushed through llama.cpp's prompt cache while you're still talking, so at the end of a long question almost everything is already processed.
- **Barge-in.** Interrupt the AI mid-sentence by speaking; generation is aborted server-side.

## Requirements

- Python 3.12+
- [llama.cpp](https://github.com/ggml-org/llama.cpp) (`brew install llama.cpp` on macOS)
- macOS with Apple Silicon, or Linux with a supported GPU
- ~4 GB free RAM for the model

## Quick start

```bash
git clone https://github.com/fikrikarim/parlor.git
cd parlor

# Install uv and llama.cpp if you don't have them
curl -LsSf https://astral.sh/uv/install.sh | sh
brew install llama.cpp

cd src
uv sync
uv run server.py
```

Open [http://localhost:8000](http://localhost:8000), grant camera and microphone access, and start talking.

Models are downloaded automatically on first run (~4 GB for Gemma 4 E2B QAT + its multimodal projector, plus TTS models).

## Configuration

| Variable           | Default                        | Description                                    |
| ------------------ | ------------------------------ | ---------------------------------------------- |
| `MODEL_PATH`       | auto-download from HuggingFace | Path to a local Gemma 4 `.gguf` file           |
| `MMPROJ_PATH`      | auto-download from HuggingFace | Path to the matching `mmproj` `.gguf` (audio + vision encoders) |
| `PORT`             | `8000`                         | Server port                                    |
| `TEMPERATURE`      | `0.7`                          | Sampling temperature (0 = deterministic)       |
| `LLAMA_CTX`        | `16384`                        | llama.cpp context size. The server drops the oldest exchanges shortly before it fills |
| `LLAMA_PORT`       | `8081`                         | Port for the spawned llama-server              |
| `LLAMA_SERVER_URL` | (spawn our own)                | Use an already-running llama-server instead    |

## Performance (Apple M3 Pro)

Measured from end of utterance to first audio heard (add ~200ms of VAD silence detection on top). The camera frame and the speech itself are prefilled while you're still speaking, so long questions and vision add very little to the critical path:

| Turn                                  | First audio | Turn complete |
| ------------------------------------- | ----------- | ------------- |
| Short question (~2s speech)           | ~0.6-0.7s   | ~0.7s         |
| Short question + camera               | ~0.7-0.8s   | ~0.8s         |
| Long question (~9s speech), streamed  | ~0.6-0.7s   | ~1.7-1.8s     |
| Long question + camera                | ~0.8-1.0s   | ~1.9-2.1s     |

Reproduce with the end-to-end benchmark (real spoken audio, synthesized locally). Run it before and after a change to see the impact:

```bash
uv run server.py                 # terminal 1
uv run python benchmarks/bench.py --label before --out benchmarks/results/before.json   # terminal 2
# ...make changes, restart the server...
uv run python benchmarks/bench.py --label after --out benchmarks/results/after.json
uv run python benchmarks/compare.py benchmarks/results/before.json benchmarks/results/after.json
```

## Testing

An end-to-end suite spawns the real server (llama.cpp, TTS, turn detector)
and drives it over WebSocket with synthesized speech — including degraded
audio (clipped word endings, noise, other voices) that reproduces live-mic
failure modes:

```bash
uv run pytest            # ~1 minute + model load
```

Set `PARLOR_TEST_URL=ws://localhost:8000/ws` to run it against an
already-running server. Browser-only behavior (echo cancellation, VAD feel,
multilingual speech) still needs the manual checklist in `HANDOFF.md`.

## Project structure

```
src/
├── server.py              # FastAPI app + per-connection conversation loop
├── llama.py               # llama-server lifecycle + chat API client
├── pipeline.py            # Streaming turn pipeline (decode → sentences → TTS)
├── turn_detector.py       # smart-turn-v3 end-of-turn classifier
├── whisper_features.py    # Log-mel features for the turn detector
├── tts.py                 # Platform-aware TTS (MLX on Mac, ONNX on Linux)
├── index.html             # Frontend markup
├── static/                # Frontend styles + app logic (VAD, camera, playback)
├── tests/                 # End-to-end test suite (uv run pytest)
└── benchmarks/
    ├── bench.py           # End-to-end latency benchmark
    ├── fixtures.py        # Spoken-audio fixtures (synthesized locally)
    ├── compare.py         # Diff two benchmark result files
    ├── turnbench.py       # Turn-detection accuracy benchmark
    └── experiment_*.py    # Recorded dead ends (see commit messages)
```

## Acknowledgments

- [Gemma 4](https://ai.google.dev/gemma) by Google DeepMind
- [llama.cpp](https://github.com/ggml-org/llama.cpp) by Georgi Gerganov and contributors
- [Kokoro](https://huggingface.co/hexgrad/Kokoro-82M) TTS by Hexgrad
- [Silero VAD](https://github.com/snakers4/silero-vad) for browser voice activity detection
- [smart-turn-v3](https://huggingface.co/pipecat-ai/smart-turn-v3) end-of-turn detection by Pipecat

## License

[Apache 2.0](LICENSE)
