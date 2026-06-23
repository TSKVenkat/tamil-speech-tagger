"""
Prosody-based emotion & speaking-style tagging for Tamil speech.

This module derives *style* descriptors (pace, energy, pitch movement) directly
from the acoustic signal, and maps them to a coarse *emotion* guess using a
transparent arousal/valence heuristic. It needs NO training and NO labels,
which is why it is reliable for a weekend demo. The emotion output is a weak
(heuristic) label — see `EMOTION_DISCLAIMER` — and is meant as a starting point
for expressive-TTS annotation, not ground truth.

Dependencies: numpy, librosa, soundfile.
"""
from __future__ import annotations

import dataclasses
from typing import Optional

import numpy as np

EMOTION_DISCLAIMER = (
    "Emotion is a heuristic estimate from prosody (arousal/valence proxies), "
    "not a trained classifier. Treat as a weak label."
)

TARGET_SR = 16000


@dataclasses.dataclass
class ProsodyFeatures:
    duration_s: float
    rms_mean: float          # overall loudness
    rms_db: float            # loudness in dBFS-ish scale
    f0_mean_hz: float        # mean voiced pitch
    f0_std_hz: float         # pitch variation (expressiveness proxy)
    voiced_ratio: float      # fraction of frames that are voiced
    speech_rate_sps: float   # syllable-nuclei per second (pace proxy)
    tempo_bpm: float


def _to_mono_16k(y: np.ndarray, sr: int) -> tuple[np.ndarray, int]:
    import librosa

    if y.ndim > 1:
        y = np.mean(y, axis=1)
    if sr != TARGET_SR:
        y = librosa.resample(y, orig_sr=sr, target_sr=TARGET_SR)
        sr = TARGET_SR
    # peak-normalise to avoid scale-dependent RMS
    peak = np.max(np.abs(y)) + 1e-9
    y = y / peak
    return y.astype(np.float32), sr


def _estimate_syllable_rate(y: np.ndarray, sr: int) -> float:
    """Count energy-onset peaks as a crude syllable-nuclei rate (Hz)."""
    import librosa

    onset_env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=256)
    peaks = librosa.util.peak_pick(
        onset_env, pre_max=3, post_max=3, pre_avg=3, post_avg=5,
        delta=0.2, wait=10,
    )
    dur = max(len(y) / sr, 1e-6)
    return float(len(peaks) / dur)


def extract_prosody(y: np.ndarray, sr: int) -> ProsodyFeatures:
    """Extract prosodic features from a waveform array."""
    import librosa

    y, sr = _to_mono_16k(y, sr)
    dur = float(len(y) / sr)

    rms = librosa.feature.rms(y=y, frame_length=1024, hop_length=256)[0]
    rms_mean = float(np.mean(rms))
    rms_db = float(20.0 * np.log10(rms_mean + 1e-9))

    # Pitch via probabilistic YIN (robust on speech, voiced/unvoiced aware)
    try:
        f0, voiced_flag, _ = librosa.pyin(
            y, fmin=70, fmax=400, sr=sr, frame_length=1024,
        )
    except Exception:
        f0 = np.full(1, np.nan)
        voiced_flag = np.zeros(1, dtype=bool)
    voiced = f0[~np.isnan(f0)]
    f0_mean = float(np.mean(voiced)) if voiced.size else 0.0
    f0_std = float(np.std(voiced)) if voiced.size else 0.0
    voiced_ratio = float(np.mean(voiced_flag)) if voiced_flag.size else 0.0

    speech_rate = _estimate_syllable_rate(y, sr)
    try:
        tempo = float(librosa.feature.tempo(y=y, sr=sr)[0])
    except Exception:
        tempo = 0.0

    return ProsodyFeatures(
        duration_s=round(dur, 3),
        rms_mean=round(rms_mean, 5),
        rms_db=round(rms_db, 2),
        f0_mean_hz=round(f0_mean, 1),
        f0_std_hz=round(f0_std, 1),
        voiced_ratio=round(voiced_ratio, 3),
        speech_rate_sps=round(speech_rate, 2),
        tempo_bpm=round(tempo, 1),
    )


def _bucket(value: float, low: float, high: float, labels: tuple[str, str, str]) -> str:
    if value < low:
        return labels[0]
    if value > high:
        return labels[2]
    return labels[1]


def style_from_prosody(p: ProsodyFeatures) -> dict:
    """Map raw prosody to human-readable style descriptors."""
    pace = _bucket(p.speech_rate_sps, 2.6, 4.2, ("slow", "medium", "fast"))
    energy = _bucket(p.rms_db, -28.0, -18.0, ("low", "medium", "high"))
    pitch_var = _bucket(p.f0_std_hz, 20.0, 45.0, ("flat", "moderate", "dynamic"))

    energetic = (energy == "high") and (pace in ("medium", "fast"))
    overall = "energetic" if energetic else ("calm" if energy == "low" else "neutral")
    return {
        "pace": pace,
        "energy": energy,
        "pitch_variation": pitch_var,
        "overall": overall,
    }


def emotion_from_prosody(p: ProsodyFeatures) -> dict:
    """
    Heuristic emotion from arousal (energy + pace + pitch movement) and a
    rough valence proxy (mean pitch height vs. speaker-agnostic midpoint).
    Returns a label plus a soft confidence in [0,1].
    """
    # Arousal proxy: z-ish combination, squashed to [0,1]
    arousal = (
        0.5 * np.clip((p.rms_db + 30.0) / 14.0, 0, 1)
        + 0.3 * np.clip((p.speech_rate_sps - 2.0) / 3.0, 0, 1)
        + 0.2 * np.clip(p.f0_std_hz / 60.0, 0, 1)
    )
    # Valence proxy: higher, more variable pitch leans positive
    valence = 0.6 * np.clip((p.f0_mean_hz - 120.0) / 120.0, 0, 1) + 0.4 * np.clip(
        p.f0_std_hz / 60.0, 0, 1
    )

    if arousal >= 0.6 and valence < 0.45:
        label = "angry"
    elif arousal >= 0.6 and valence >= 0.45:
        label = "happy"
    elif arousal < 0.4 and valence < 0.45:
        label = "sad"
    elif arousal < 0.4 and valence >= 0.45:
        label = "calm"
    else:
        label = "neutral"

    # confidence = distance from the decision boundaries, bounded
    conf = float(np.clip(abs(arousal - 0.5) + abs(valence - 0.5) + 0.3, 0.3, 0.95))
    return {
        "emotion": label,
        "emotion_conf": round(conf, 2),
        "arousal": round(float(arousal), 3),
        "valence": round(float(valence), 3),
        "_note": EMOTION_DISCLAIMER,
    }


def analyze_waveform(y: np.ndarray, sr: int) -> dict:
    p = extract_prosody(y, sr)
    return {
        "prosody": dataclasses.asdict(p),
        "style": style_from_prosody(p),
        **emotion_from_prosody(p),
    }


def analyze_file(path: str) -> dict:
    import soundfile as sf

    y, sr = sf.read(path, dtype="float32")
    return analyze_waveform(y, sr)
