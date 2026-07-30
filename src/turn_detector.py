"""End-of-turn detection with the smart-turn-v3 ONNX classifier.

A small audio-native model (from Pipecat, BSD-2) trained specifically to
judge whether a speaker has finished their turn — far more reliable at this
than prompting a 2B LLM, and it answers in tens of milliseconds on CPU.
Inference code adapted from pipecat.audio.turn.smart_turn.local_smart_turn_v3.
"""

import time

import numpy as np
import onnxruntime as ort

from whisper_features import compute_whisper_log_mel_features

HF_REPO = "pipecat-ai/smart-turn-v3"
HF_FILE = "smart-turn-v3.2-cpu.onnx"
SAMPLE_RATE = 16000
WINDOW_SECONDS = 8  # the model judges the last 8 seconds of the utterance


class TurnDetector:
    def __init__(self, model_path: str | None = None):
        if not model_path:
            from huggingface_hub import hf_hub_download
            try:
                model_path = hf_hub_download(HF_REPO, HF_FILE)
            except Exception:  # offline — use the local cache
                model_path = hf_hub_download(HF_REPO, HF_FILE, local_files_only=True)

        so = ort.SessionOptions()
        so.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        so.inter_op_num_threads = 1
        so.intra_op_num_threads = 2
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self._session = ort.InferenceSession(model_path, sess_options=so)

        # Warmup: the first run pays graph initialization.
        self.predict(np.zeros(SAMPLE_RATE, dtype=np.float32))
        print("Turn detector loaded (smart-turn-v3.2).")

    def predict(self, audio: np.ndarray) -> tuple[bool, float]:
        """(complete, probability) for 16kHz float32 mono audio."""
        t0 = time.time()
        max_samples = WINDOW_SECONDS * SAMPLE_RATE
        if len(audio) > max_samples:
            audio = audio[-max_samples:]
        elif len(audio) < max_samples:
            audio = np.pad(audio, (max_samples - len(audio), 0))

        log_mel = compute_whisper_log_mel_features(audio, do_normalize=True)
        outputs = self._session.run(None, {"input_features": np.expand_dims(log_mel, 0)})
        probability = outputs[0][0].item()
        elapsed_ms = (time.time() - t0) * 1000
        print(f"Turn detector: p(complete)={probability:.2f} ({elapsed_ms:.0f}ms)")
        return probability > 0.5, probability
