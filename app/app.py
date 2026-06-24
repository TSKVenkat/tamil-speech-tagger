"""
Gradio web app: Tamil speech -> transcript + emotion/style tags.
Deploy as a Hugging Face Space (SDK: gradio). This IS the website.

Set the Space variable ASR_MODEL to your fine-tuned checkpoint, e.g.
    ASR_MODEL = your-username/whisper-small-ta-saaras
It falls back to openai/whisper-small until your model is trained & pushed.
"""
import json
import os
import sys

import gradio as gr

# allow importing the shared pipeline from ../src whether run locally or on a Space
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.dirname(__file__))

from pipeline import TamilSpeechPipeline, DEFAULT_MODEL  # noqa: E402

MODEL_ID = os.environ.get("ASR_MODEL", DEFAULT_MODEL)
_pipe = None


def get_pipe():
    global _pipe
    if _pipe is None:
        _pipe = TamilSpeechPipeline(MODEL_ID)
    return _pipe


def _emotion_badge(emotion: str, conf: float) -> str:
    colors = {
        "angry": "#e74c3c", "happy": "#f39c12", "sad": "#3498db",
        "calm": "#1abc9c", "neutral": "#95a5a6",
    }
    c = colors.get(emotion, "#95a5a6")
    return (
        f"<div style='display:inline-block;padding:6px 14px;border-radius:16px;"
        f"background:{c};color:white;font-weight:600'>"
        f"{emotion} · {conf:.0%}</div>"
    )


def analyze(audio):
    if audio is None:
        return "Please record or upload Tamil audio.", "", {}, ""
    out = get_pipe()(audio)
    transcript = out["transcript"] or "(no speech detected)"
    s = out["style"]
    style_md = (
        f"**Emotion:** {_emotion_badge(out['emotion'], out['emotion_conf'])}  \n"
        f"**Style:** pace `{s['pace']}` · energy `{s['energy']}` · "
        f"pitch `{s['pitch_variation']}` · overall **{s['overall']}**"
    )
    return transcript, style_md, out, json.dumps(out, ensure_ascii=False, indent=2)


with gr.Blocks(title="Kural") as demo:
    gr.Markdown(
        "# 🎙️ Kural — Tamil Speech → Transcript + Emotion/Style\n"
        "**ASR** (fine-tuned Whisper) + prosody-based **emotion & speaking-style** "
        "tagging → dubbing-ready annotations.\n\n"
        f"_Model: `{MODEL_ID}` · emotion tags are heuristic (prosody-derived)._"
    )
    with gr.Row():
        with gr.Column():
            audio_in = gr.Audio(sources=["microphone", "upload"], type="filepath",
                                format="wav", label="Tamil audio (record or upload)")
            btn = gr.Button("Analyze", variant="primary")
        with gr.Column():
            txt = gr.Textbox(label="Transcript (தமிழ்)", lines=3)
            style_out = gr.Markdown()
    with gr.Accordion("Full annotation (JSON)", open=False):
        json_view = gr.JSON()
        raw = gr.Code(language="json", label="Raw JSON")
    btn.click(analyze, inputs=audio_in, outputs=[txt, style_out, json_view, raw])
    audio_in.stop_recording(analyze, inputs=audio_in,
                            outputs=[txt, style_out, json_view, raw])

if __name__ == "__main__":
    demo.launch(theme=gr.themes.Soft())
