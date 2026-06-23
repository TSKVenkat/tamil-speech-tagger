// Standalone frontend that calls the deployed Gradio Space.
// 1) Deploy app/ as a HF Space.  2) Put its id here ("username/space-name").
import { Client } from "https://cdn.jsdelivr.net/npm/@gradio/client/dist/index.min.js";

const SPACE_ID = "your-username/tamil-speech-tagger"; // <-- set this

const recBtn = document.getElementById("recBtn");
const fileIn = document.getElementById("fileIn");
const player = document.getElementById("player");
const statusEl = document.getElementById("status");
const results = document.getElementById("results");
const transcriptEl = document.getElementById("transcript");
const tagsEl = document.getElementById("tags");
const jsonEl = document.getElementById("json");

const EMOTION_COLORS = {
  angry: "#e74c3c", happy: "#f39c12", sad: "#3498db",
  calm: "#1abc9c", neutral: "#95a5a6",
};

let mediaRecorder, chunks = [], recording = false;

recBtn.addEventListener("click", async () => {
  if (recording) { mediaRecorder.stop(); return; }
  try {
    const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
    mediaRecorder = new MediaRecorder(stream);
    chunks = [];
    mediaRecorder.ondataavailable = (e) => chunks.push(e.data);
    mediaRecorder.onstop = () => {
      stream.getTracks().forEach((t) => t.stop());
      const blob = new Blob(chunks, { type: "audio/webm" });
      handleAudio(blob);
    };
    mediaRecorder.start();
    recording = true;
    recBtn.textContent = "■ Stop";
    recBtn.classList.add("recording");
    setStatus("Recording…");
  } catch (e) {
    setStatus("Mic access denied: " + e.message);
  }
});

fileIn.addEventListener("change", () => {
  if (fileIn.files[0]) handleAudio(fileIn.files[0]);
});

async function handleAudio(blob) {
  recording = false;
  recBtn.textContent = "● Record";
  recBtn.classList.remove("recording");
  player.src = URL.createObjectURL(blob);
  player.hidden = false;
  setStatus("Analyzing… (first call wakes the Space, may take ~30s)");
  try {
    const app = await Client.connect(SPACE_ID);
    const res = await app.predict("/analyze", [blob]);
    render(res.data);
    setStatus("");
  } catch (e) {
    setStatus("Error: " + (e.message || e));
  }
}

function render(data) {
  // app.analyze returns [transcript, style_md, json_obj, raw_json_string]
  const [transcript, , jsonObj] = data;
  transcriptEl.textContent = transcript || "(no speech detected)";
  tagsEl.innerHTML = "";
  if (jsonObj && typeof jsonObj === "object") {
    const emo = jsonObj.emotion, conf = jsonObj.emotion_conf, s = jsonObj.style || {};
    addTag(`${emo} · ${Math.round((conf || 0) * 100)}%`, "emotion",
           EMOTION_COLORS[emo] || "#95a5a6");
    if (s.pace) addTag(`pace: ${s.pace}`);
    if (s.energy) addTag(`energy: ${s.energy}`);
    if (s.pitch_variation) addTag(`pitch: ${s.pitch_variation}`);
    if (s.overall) addTag(`overall: ${s.overall}`);
    jsonEl.textContent = JSON.stringify(jsonObj, null, 2);
  }
  results.hidden = false;
}

function addTag(text, cls = "", bg = null) {
  const span = document.createElement("span");
  span.className = "tag " + cls;
  span.textContent = text;
  if (bg) span.style.background = bg;
  tagsEl.appendChild(span);
}

function setStatus(msg) { statusEl.textContent = msg; }
