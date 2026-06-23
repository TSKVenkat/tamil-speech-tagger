---
title: Tamil Speech Tagger
emoji: 🎙️
colorFrom: indigo
colorTo: pink
sdk: gradio
sdk_version: 4.44.0
app_file: app.py
pinned: false
license: cc0-1.0
---

# Tamil Speech → Transcript + Emotion/Style

Saaras-inspired pipeline for Tamil:

- **ASR** — fine-tuned Whisper (Common Voice Tamil) → Tamil transcript + timestamps
- **Emotion & style tagging** — prosody-derived (pace / energy / pitch) → dubbing-ready JSON

## Configure the model

Set a Space **Variable** `ASR_MODEL` to your fine-tuned checkpoint, e.g.
`your-username/whisper-small-ta-saaras`. Defaults to `openai/whisper-small`.

> Emotion labels are heuristic estimates from prosody, not a trained classifier.

The `app.py`, `pipeline.py`, and `prosody.py` files must sit together in the
Space repo root (copy `src/pipeline.py` and `src/prosody.py` next to `app.py`).
