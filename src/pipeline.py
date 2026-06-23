"""
End-to-end Tamil speech -> dubbing-ready annotation pipeline.

    audio  ->  Whisper ASR (transcript, optional timestamps)
           ->  prosody-based emotion + speaking-style tags
           ->  JSON ready for expressive-TTS / dubbing

The ASR model defaults to your fine-tuned checkpoint on the Hub but falls back
to the multilingual base Whisper so the pipeline runs before training finishes.
"""
from __future__ import annotations

import os
from typing import Optional

import numpy as np

from prosody import analyze_waveform, extract_prosody, style_from_prosody, emotion_from_prosody

DEFAULT_MODEL = os.environ.get("ASR_MODEL", "openai/whisper-small")
TARGET_SR = 16000


class TamilSpeechPipeline:
    def __init__(self, model_id: str = DEFAULT_MODEL, device: Optional[str] = None):
        import torch
        from transformers import pipeline as hf_pipeline

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        self.model_id = model_id
        self.asr = hf_pipeline(
            "automatic-speech-recognition",
            model=model_id,
            device=0 if device == "cuda" else -1,
            chunk_length_s=30,
            return_timestamps=True,
            generate_kwargs={"language": "tamil", "task": "transcribe"},
        )

    def _load_audio(self, audio) -> tuple[np.ndarray, int]:
        """Accept a filepath, or a (sr, ndarray)/(ndarray, sr) tuple (Gradio)."""
        if isinstance(audio, str):
            import soundfile as sf

            y, sr = sf.read(audio, dtype="float32")
            return y, sr
        if isinstance(audio, tuple) and len(audio) == 2:
            a, b = audio
            if isinstance(a, np.ndarray):
                y, sr = a, b
            else:
                sr, y = a, b
            y = y.astype(np.float32)
            if np.issubdtype(y.dtype, np.integer):
                y = y / 32768.0
            return y, int(sr)
        raise ValueError(f"Unsupported audio input: {type(audio)}")

    def transcribe(self, audio) -> dict:
        return self.asr(audio if isinstance(audio, str) else self._gradio_to_asr(audio))

    @staticmethod
    def _gradio_to_asr(audio):
        sr, y = (audio[1], audio[0]) if isinstance(audio[0], np.ndarray) else audio
        y = np.asarray(y, dtype=np.float32)
        if y.max() > 1.0 or y.min() < -1.0:
            y = y / 32768.0
        return {"array": y, "sampling_rate": int(sr)}

    def __call__(self, audio) -> dict:
        y, sr = self._load_audio(audio)

        # 1) ASR (transcript + chunk timestamps)
        asr_input = audio if isinstance(audio, str) else {"array": y, "sampling_rate": sr}
        asr_out = self.asr(asr_input)
        transcript = asr_out["text"].strip()
        chunks = asr_out.get("chunks", []) or []

        # 2) clip-level prosody / emotion / style
        clip = analyze_waveform(y, sr)

        # 3) per-segment tags (reuse Whisper chunk boundaries)
        segments = []
        for i, ch in enumerate(chunks):
            ts = ch.get("timestamp", (None, None)) or (None, None)
            start, end = ts
            seg = {"id": i, "start": start, "end": end, "text": ch.get("text", "").strip()}
            if start is not None and end is not None and end > start:
                a, b = int(start * sr), int(min(end, len(y) / sr) * sr)
                if b - a > sr // 4:  # need >=0.25s to estimate prosody
                    p = extract_prosody(y[a:b], sr)
                    seg["style"] = style_from_prosody(p)
                    seg.update({k: v for k, v in emotion_from_prosody(p).items()
                                if k in ("emotion", "emotion_conf")})
            seg["speaker"] = "SPEAKER_1"  # mock single-speaker label; swap for diarization
            segments.append(seg)

        return {
            "transcript": transcript,
            "emotion": clip["emotion"],
            "emotion_conf": clip["emotion_conf"],
            "style": clip["style"],
            "language": "ta",
            "model": self.model_id,
            "prosody": clip["prosody"],
            "segments": segments,
            "_note": clip["_note"],
        }


if __name__ == "__main__":
    import json
    import sys

    if len(sys.argv) < 2:
        print("usage: python pipeline.py <audio_file> [model_id]")
        raise SystemExit(1)
    model = sys.argv[2] if len(sys.argv) > 2 else DEFAULT_MODEL
    pipe = TamilSpeechPipeline(model)
    print(json.dumps(pipe(sys.argv[1]), ensure_ascii=False, indent=2))
