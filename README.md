# Kural — multi-stage Tamil speech pipeline

Fine-tune **Whisper** on **Common Voice Tamil** for ASR, then tag each clip with
**emotion + speaking style** from prosody — producing **dubbing-ready annotations**
for expressive TTS, mirroring stages of a production dubbing pipeline.

```
 Tamil audio ──▶ Whisper ASR ──▶ transcript (+ timestamps)
             └─▶ prosody (pitch / energy / pace) ──▶ emotion + style tags ──▶ JSON
```

Example output:

```json
{
  "transcript": "நான் மிகவும் கோபமாக இருக்கிறேன்",
  "emotion": "angry", "emotion_conf": 0.78,
  "style": {"pace": "fast", "energy": "high", "pitch_variation": "dynamic", "overall": "energetic"},
  "segments": [{"id": 0, "start": 0.0, "end": 3.2, "speaker": "SPEAKER_1", "text": "...", "emotion": "angry"}]
}
```

## Models

### ASR — fine-tuned `openai/whisper-small` on Common Voice Tamil

([`abar-uwc/tamil-common-voice_v21`](https://huggingface.co/datasets/abar-uwc/tamil-common-voice_v21),
a Parquet mirror — the Mozilla HF datasets were retired in Oct 2025):

| Model | WER ↓ | CER ↓ |
|-------|------|------|
| [`Venky0411/whisper-small-ta-saaras-v2`](https://huggingface.co/Venky0411/whisper-small-ta-saaras-v2) **(current)** | 35.5 | 5.9 |
| [`Venky0411/whisper-small-ta-saaras`](https://huggingface.co/Venky0411/whisper-small-ta-saaras) (initial run) | 58.1 | 19.5 |

### SER — emotion (frozen Whisper encoder + attention pooling + classifier)

Speaker-independent test, same architecture both rows — only the language differs:

| Data | Acc | Macro-F1 |
|------|-----|----------|
| **English** (CREMA-D, 91 actors) | **0.872** | **0.872** |
| **Tamil** (EmoTa, 22 speakers) | 0.380 | 0.302 |

**Key finding:** the design is sound (0.87 on English); the Tamil gap is the **language** —
Whisper's *frozen* features barely model Tamil. Fix = swap the frozen backbone to a
Tamil-aware SSL model (XLS-R / IndicWav2Vec). Full write-up in [`docs/EXPERIMENTS.md`](docs/EXPERIMENTS.md).

## Layout

| Path | What |
|------|------|
| `serve/` | **Primary web app** — FastAPI + faster-whisper (fast CPU ASR) + simple record/upload UI |
| `notebooks/train_whisper_tamil_colab.ipynb` | Train **ASR** on Colab GPU + `push_to_hub` |
| `notebooks/train_emotion_whisper_colab.ipynb` | Train **Tamil emotion** (frozen Whisper + attention pooling, EmoTa) |
| `notebooks/train_emotion_english_whisper.ipynb` | Train **English emotion** (same design, CREMA-D, speaker-independent) |
| `docs/EXPERIMENTS.md` | **Experiments, results & learnings** (what worked, what didn't, why) |
| `scripts/train_whisper_tamil.py` | Same training loop as a CLI script |
| `src/prosody.py` | Prosody → emotion/style (no training needed) |
| `src/pipeline.py` | `transformers` ASR + emotion/style → JSON |
| `app/` | Alt deploy: Gradio app for a Hugging Face Space |
| `web/` | Alt: standalone static frontend that calls a Space |

## Quickstart

### Run the web app (recommended)

Fast local inference via faster-whisper (CTranslate2 int8) — ~5 s/clip on CPU,
no GPU needed. See [`serve/README.md`](serve/README.md) for detail.

```bash
pip install -r serve/requirements.txt

# one-time: convert the HF model to CTranslate2 int8
ct2-transformers-converter --model Venky0411/whisper-small-ta-saaras-v2 \
  --output_dir ct2-model --quantization int8

# run (serves UI + /api/analyze on :7860)
CT2_MODEL=ct2-model uvicorn server:app --app-dir serve --host 0.0.0.0 --port 7860
```

Open <http://localhost:7860> → record or upload Tamil audio → transcript + emotion/style + JSON.

### Train (Google Colab, free T4)

Open `notebooks/train_whisper_tamil_colab.ipynb` in Colab → **Runtime → GPU**, **Run all**.
It loads Common Voice Tamil, fine-tunes Whisper, reports **WER/CER** on a
**speaker-independent** split, and `push_to_hub`. Defaults use the Parquet dataset above
(set `USE_FLEURS = True` for a smaller/faster run). The HF token is read from a Colab
Secret / env var / prompt — never hard-coded.

## Notes & honesty

- **ASR** is genuinely fine-tuned (full fine-tune). **Emotion (SER)** is a real trained
  classifier — frozen encoder + learnable **attention pooling** + linear head, trained
  only on the head — evaluated **speaker-independent** (held-out speakers/actors). All
  numbers above are honest held-out scores, not random splits.
- The live `serve/` app still uses a transparent **prosody heuristic** for emotion (a weak
  label); wiring in the trained SER model is the next integration step.
- Whisper is the right backbone for **ASR**, but the wrong *frozen* feature extractor for
  **Tamil** emotion (it barely models Tamil) — see the SER finding above.
- Common Voice is **read speech**; EmoTa is **acted** → note generalization to spontaneous/dubbing.
- Datasets: Common Voice is CC0 (cite Mozilla); **EmoTa is gated** (approval) and is *not*
  committed to this repo.

```
@misc{commonvoice, title={Common Voice}, author={Mozilla Foundation}, howpublished={https://commonvoice.mozilla.org}}
```
