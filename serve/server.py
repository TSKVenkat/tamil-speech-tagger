"""
Kural — dedicated FastAPI backend.
Fast local ASR via faster-whisper (CTranslate2 int8) + prosody emotion/style.
Serves the static UI at / and the analysis endpoint at /api/analyze.
"""
import os
import sys
import time
import tempfile

import librosa
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from faster_whisper import WhisperModel

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "src"))
from prosody import (  # noqa: E402
    analyze_waveform, extract_prosody, style_from_prosody, emotion_from_prosody,
)

MODEL_DIR = os.environ.get("CT2_MODEL", "/tmp/sarvam-svc/ct2-model")
SR = 16000

print(f"[kural] loading model from {MODEL_DIR} ...", flush=True)
model = WhisperModel(MODEL_DIR, device="cpu", compute_type="int8",
                     cpu_threads=os.cpu_count() or 4)
print("[kural] model ready", flush=True)

app = FastAPI(title="Kural")


@app.get("/api/health")
def health():
    return {"ok": True, "model": MODEL_DIR}


@app.post("/api/analyze")
async def analyze(audio: UploadFile = File(...)):
    t0 = time.time()
    raw = await audio.read()
    suffix = os.path.splitext(audio.filename or "")[1] or ".bin"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as fh:
        fh.write(raw)
        path = fh.name
    try:
        # librosa+ffmpeg decode handles webm/opus (mic), wav, mp3, ... -> 16k mono
        y, sr = librosa.load(path, sr=SR, mono=True)
    except Exception as e:
        return JSONResponse({"error": f"could not decode audio: {e}"}, status_code=400)
    finally:
        os.unlink(path)

    if y.size < SR // 4:
        return JSONResponse({"error": "audio too short (need >0.25s)"}, status_code=400)

    segments, _info = model.transcribe(
        y, language="ta", task="transcribe", beam_size=1, vad_filter=True,
    )

    segs, parts = [], []
    for i, s in enumerate(segments):
        parts.append(s.text)
        seg = {"id": i, "start": round(s.start, 2), "end": round(s.end, 2),
               "text": s.text.strip(), "speaker": "SPEAKER_1"}
        a, b = int(s.start * sr), int(min(s.end, len(y) / sr) * sr)
        if b - a > sr // 4:
            p = extract_prosody(y[a:b], sr)
            seg["style"] = style_from_prosody(p)
            seg.update({k: v for k, v in emotion_from_prosody(p).items()
                        if k in ("emotion", "emotion_conf")})
        segs.append(seg)

    transcript = "".join(parts).strip()
    clip = analyze_waveform(y, sr)
    return {
        "transcript": transcript,
        "emotion": clip["emotion"],
        "emotion_conf": clip["emotion_conf"],
        "style": clip["style"],
        "language": "ta",
        "duration_s": round(len(y) / sr, 2),
        "infer_s": round(time.time() - t0, 2),
        "segments": segs,
        "prosody": clip["prosody"],
        "_note": clip["_note"],
    }


# static UI (must be mounted last so /api/* wins)
app.mount("/", StaticFiles(directory=os.path.join(HERE, "static"), html=True), name="ui")
