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
    ├── Gemma 4 E2B via LiteRT-LM (GPU)  →  understands speech + vision
    └── Kokoro TTS (MLX on Mac, ONNX on Linux)  →  speaks back
    │
    │  WebSocket (streamed audio chunks)
    ▼
Browser (playback + transcript)
```

- **Voice Activity Detection** in the browser ([Silero VAD](https://github.com/ricky0123/vad)). Hands-free, no push-to-talk, with a short 200ms silence cutoff for fast turn-taking.
- **Turn-completeness filtering.** Gemma judges every utterance (`FINISHED` / `WAIT`) before answering — if you were cut off mid-sentence or paused to think, it stays quiet and lets you continue, then gently nudges you if you go silent. (Same idea as [Pipecat's incomplete-turn filtering](https://docs.pipecat.ai/api-reference/server/utilities/turn-management/filter-incomplete-turns).)
- **Streaming decode → TTS.** The response is spoken sentence-by-sentence while the model is still generating, and the transcript is generated last so it never delays audio.
- **Speculative frame prefill.** The camera frame is sent the moment you start speaking, so Gemma digests the image (~274 tokens) while you're still talking — vision costs almost nothing by the time you finish.
- **Barge-in.** Interrupt the AI mid-sentence by speaking; generation is cancelled server-side.

## Requirements

- Python 3.12+
- macOS with Apple Silicon, or Linux with a supported GPU
- ~3 GB free RAM for the model

## Quick start

```bash
git clone https://github.com/fikrikarim/parlor.git
cd parlor

# Install uv if you don't have it
curl -LsSf https://astral.sh/uv/install.sh | sh

cd src
uv sync
uv run server.py
```

Open [http://localhost:8000](http://localhost:8000), grant camera and microphone access, and start talking.

Models are downloaded automatically on first run (~2.6 GB for Gemma 4 E2B, plus TTS models).

## Configuration

| Variable         | Default                        | Description                                    |
| ---------------- | ------------------------------ | ---------------------------------------------- |
| `MODEL_PATH`     | auto-download from HuggingFace | Path to a local `gemma-4-E2B-it.litertlm` file |
| `PORT`           | `8000`                         | Server port                                    |
| `MAX_NUM_TOKENS` | `32768`                        | KV-cache size (context window). Lower it to save ~0.5 GB RAM. The server starts a fresh conversation shortly before the cache fills |
| `AUDIO_BACKEND`  | `cpu`                          | Audio encoder backend (the current model file only supports `cpu`) |
| `AUDIO_THREADS`  | library default                | CPU threads for the audio encoder              |

## Performance (Apple M3 Pro)

Measured from end of utterance to first audio heard (add ~200ms of VAD silence detection on top). The camera frame is prefilled while you're still speaking, so vision adds almost nothing to the critical path:

| Turn                             | First audio | Turn complete |
| -------------------------------- | ----------- | ------------- |
| Short question (~2s speech)      | ~0.9s       | ~1.2s         |
| Short question + camera          | ~0.6s       | ~0.9s         |
| Long question (~9s speech)       | ~1.1s       | ~2.2s         |
| Long question + camera           | ~1.1s       | ~2.2s         |

Reproduce with the end-to-end benchmark (real spoken audio, synthesized locally). Run it before and after a change to see the impact:

```bash
uv run server.py                 # terminal 1
uv run python benchmarks/bench.py --label before --out benchmarks/results/before.json   # terminal 2
# ...make changes, restart the server...
uv run python benchmarks/bench.py --label after --out benchmarks/results/after.json
uv run python benchmarks/compare.py benchmarks/results/before.json benchmarks/results/after.json
```

## Project structure

```
src/
├── server.py              # FastAPI WebSocket server + Gemma 4 inference
├── tts.py                 # Platform-aware TTS (MLX on Mac, ONNX on Linux)
├── index.html             # Frontend UI (VAD, camera, audio playback)
├── pyproject.toml         # Dependencies
└── benchmarks/
    ├── bench.py           # End-to-end perf + correctness benchmark
    ├── fixtures.py        # Spoken-audio test fixtures (synthesized locally)
    ├── compare.py         # Diff two benchmark result files
    └── benchmark_tts.py   # TTS backend comparison
```

## Acknowledgments

- [Gemma 4](https://ai.google.dev/gemma) by Google DeepMind
- [LiteRT-LM](https://github.com/google-ai-edge/LiteRT-LM) by Google AI Edge
- [Kokoro](https://huggingface.co/hexgrad/Kokoro-82M) TTS by Hexgrad
- [Silero VAD](https://github.com/snakers4/silero-vad) for browser voice activity detection

## License

[Apache 2.0](LICENSE)
