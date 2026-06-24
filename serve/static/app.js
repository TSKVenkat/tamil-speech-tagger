const recBtn = document.getElementById("rec");
const fileIn = document.getElementById("file");
const player = document.getElementById("player");
const statusEl = document.getElementById("status");
const result = document.getElementById("result");
const transcriptEl = document.getElementById("transcript");
const tagsEl = document.getElementById("tags");
const metaEl = document.getElementById("meta");
const jsonEl = document.getElementById("json");

const EMO = { angry: "#e74c3c", happy: "#f39c12", sad: "#3498db", calm: "#1abc9c", neutral: "#95a5a6" };

let mr, chunks = [], recording = false, timer = null;

recBtn.onclick = async () => {
  if (recording) { mr.stop(); return; }
  try {
    const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
    mr = new MediaRecorder(stream);
    chunks = [];
    mr.ondataavailable = (e) => e.data.size && chunks.push(e.data);
    mr.onstop = () => {
      stream.getTracks().forEach((t) => t.stop());
      const blob = new Blob(chunks, { type: mr.mimeType || "audio/webm" });
      play(blob);
      send(blob, "recording.webm");
    };
    mr.start();
    recording = true;
    recBtn.textContent = "■ Stop";
    recBtn.classList.add("rec");
    setStatus("Recording… click Stop when done.");
  } catch (e) {
    setStatus("Microphone blocked: " + e.message, true);
  }
};

fileIn.onchange = () => {
  const f = fileIn.files[0];
  if (f) { play(f); send(f, f.name); }
};

function play(blob) {
  player.src = URL.createObjectURL(blob);
  player.hidden = false;
}

async function send(blob, name) {
  recording = false;
  recBtn.textContent = "● Record";
  recBtn.classList.remove("rec");
  recBtn.disabled = true;
  result.hidden = true;

  const t0 = Date.now();
  clearInterval(timer);
  timer = setInterval(() => {
    statusEl.innerHTML = `<span class="spinner"></span>Transcribing… ${((Date.now() - t0) / 1000).toFixed(1)}s`;
  }, 100);

  try {
    const fd = new FormData();
    fd.append("audio", blob, name);
    const res = await fetch("/api/analyze", { method: "POST", body: fd });
    const data = await res.json();
    clearInterval(timer);
    if (!res.ok || data.error) { setStatus("Error: " + (data.error || res.status), true); return; }
    render(data);
    setStatus("");
  } catch (e) {
    clearInterval(timer);
    setStatus("Error: " + e.message, true);
  } finally {
    recBtn.disabled = false;
  }
}

function render(d) {
  transcriptEl.textContent = d.transcript || "(no speech detected)";
  tagsEl.innerHTML = "";
  const s = d.style || {};
  addTag(`${d.emotion} · ${Math.round((d.emotion_conf || 0) * 100)}%`, EMO[d.emotion]);
  ["pace", "energy", "pitch_variation", "overall"].forEach((k) => s[k] && addTag(`${k.replace("_", " ")}: ${s[k]}`));
  metaEl.textContent = `${d.duration_s}s audio · transcribed in ${d.infer_s}s · ${(d.segments || []).length} segment(s)`;
  jsonEl.textContent = JSON.stringify(d, null, 2);
  result.hidden = false;
}

function addTag(text, bg) {
  const el = document.createElement("span");
  el.className = "tag" + (bg ? " emo" : "");
  el.textContent = text;
  if (bg) el.style.background = bg;
  tagsEl.appendChild(el);
}

function setStatus(msg, err = false) {
  statusEl.innerHTML = msg;
  statusEl.classList.toggle("err", err);
}
