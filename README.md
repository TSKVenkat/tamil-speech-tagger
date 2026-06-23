# Tamil Speech Tagger — Saaras-inspired multi-stage pipeline

Fine-tune **Whisper** on **Common Voice Tamil** for ASR, then tag each clip with
**emotion + speaking style** from prosody — producing **dubbing-ready annotations**
for expressive TTS, mirroring stages of Sarvam's Saaras pipeline.

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

## Layout

| Path | What |
|------|------|
| `notebooks/train_whisper_tamil_colab.ipynb` | **Train on Colab GPU + push to HF Hub** |
| `src/prosody.py` | Prosody → emotion/style (no training needed) |
| `src/pipeline.py` | ASR + emotion/style → JSON |
| `app/` | Gradio app → deploy as a **Hugging Face Space** (the website) |
| `web/` | Standalone HTML/JS frontend that calls the Space |
| `scripts/setup_env.sh` | Local env that avoids the full `/home` partition |

## Quickstart

### 1. Train (Google Colab, free T4)
Open `notebooks/train_whisper_tamil_colab.ipynb` in Colab → **Runtime → GPU**.
Set `HF_USERNAME`, run all. It loads Common Voice Tamil, fine-tunes Whisper,
reports **WER/CER** on a **speaker-independent** test split, and `push_to_hub`.

> Common Voice 17 is gated — open its
> [HF dataset page](https://huggingface.co/datasets/mozilla-foundation/common_voice_17_0)
> and click **Agree** once while logged in.

### 2. Deploy the website (Hugging Face Space)
Create a **Gradio** Space, then copy `app/app.py`, `app/requirements.txt`,
`app/README.md`, **and** `src/pipeline.py` + `src/prosody.py` into the Space repo
root. Set Space variable `ASR_MODEL=your-username/whisper-small-ta-saaras`.

### 3. (Optional) Standalone frontend
Set `SPACE_ID` in `web/app.js`, then serve: `python3 -m http.server -d web 8000`.

### 4. (Optional) Run locally
No GPU + full `/home` here, so use a writable big partition:
```bash
sudo mkdir -p /opt/$USER && sudo chown $USER /opt/$USER   # one-time, needs admin
BASE=/opt/$USER ./scripts/setup_env.sh
```

## Notes & honesty
- **ASR** is genuinely fine-tuned; **emotion** labels are a transparent prosody
  heuristic (arousal/valence proxies), not a trained classifier — a weak label to
  bootstrap expressive-TTS annotation. Upgrade path: train an emotion head on the
  frozen Whisper encoder using pseudo-labels (emotion2vec) or the EmoTa dataset.
- Common Voice is **read speech** → note generalization to spontaneous/dubbing.
- Speaker-independent eval (`client_id` grouping) avoids speaker leakage.
- License: Common Voice is CC0; please cite Mozilla.
```
@misc{commonvoice, title={Common Voice}, author={Mozilla Foundation}, howpublished={https://commonvoice.mozilla.org}}
```
