const SMOOTH_FRAMES = 8;
const MIN_AGREEMENT = 0.6;
const HOLD_MS = 1000;
const RECORD_MS = 2500;
const COUNTDOWN_FROM = 3;
const RING_LENGTH = 2 * Math.PI * 54;
const CONNECTIONS = [
  [0, 1], [1, 2], [2, 3], [3, 4], [0, 5], [5, 6], [6, 7], [7, 8],
  [5, 9], [9, 10], [10, 11], [11, 12], [9, 13], [13, 14], [14, 15], [15, 16],
  [13, 17], [17, 18], [18, 19], [19, 20], [0, 17],
];
const NOTES = {
  letters: "J and Z: draw them in the air.",
  numbers: "",
};

const $ = (id) => document.getElementById(id);
const video = $("cam");
const overlay = $("overlay");
const ctx = overlay.getContext("2d");
const grabber = document.createElement("canvas");

let running = false;
let loopId = 0;
let mode = "letters";
let recording = false;
let history = [];
let holding = null;
let holdStart = 0;
let lastAdded = null;
let text = "";

$("ringFg").style.strokeDasharray = RING_LENGTH;
setRing(0, false);

const sleep = (ms) => new Promise((resolve) => setTimeout(resolve, ms));

function setStatus(kind, message) {
  $("status").className = "status " + kind;
  $("statusText").textContent = message;
}

function setRing(progress, done) {
  const ring = $("ringFg");
  ring.style.strokeDashoffset = RING_LENGTH * (1 - Math.min(progress, 1));
  ring.classList.toggle("done", done);
}

function renderText() {
  $("text").textContent = text;
  $("placeholder").style.display = text ? "none" : "inline";
}

function addText(value) {
  text += value;
  renderText();
}

function addWord(word) {
  if (text && !text.endsWith(" ")) text += " ";
  addText(word.toUpperCase() + " ");
}

function renderGuesses(target, ranked) {
  $(target).innerHTML = ranked.map(([lab, share]) => `
    <div class="guess">
      <span class="lab">${lab}</span>
      <div class="bar"><div class="fill" style="width:${Math.round(share * 100)}%"></div></div>
      <span class="pct">${Math.round(share * 100)}%</span>
    </div>`).join("");
}

function highlightChip(sign) {
  document.querySelectorAll(".chip").forEach((chip) => {
    chip.classList.toggle("active", chip.dataset.sign === sign);
  });
}

function drawHands(hands) {
  ctx.clearRect(0, 0, overlay.width, overlay.height);
  const w = overlay.width;
  const h = overlay.height;
  for (const points of hands) {
    ctx.strokeStyle = "rgba(255,255,255,0.85)";
    ctx.lineWidth = 3;
    for (const [a, b] of CONNECTIONS) {
      ctx.beginPath();
      ctx.moveTo(points[a][0] * w, points[a][1] * h);
      ctx.lineTo(points[b][0] * w, points[b][1] * h);
      ctx.stroke();
    }
    ctx.fillStyle = "#8fd3b6";
    for (const [x, y] of points) {
      ctx.beginPath();
      ctx.arc(x * w, y * h, 5, 0, 2 * Math.PI);
      ctx.fill();
    }
  }
}

function resetSignState() {
  history = [];
  holding = null;
  lastAdded = null;
  $("bigLetter").textContent = "–";
  renderGuesses("guesses", []);
  highlightChip(null);
  setRing(0, false);
}

function handleResult(result) {
  if (!result.hand) {
    resetSignState();
    drawHands([]);
    $("hint").textContent = "Hold a sign still for 1 second to type it";
    setStatus("nohand", "No hand detected – show your hand");
    return;
  }

  drawHands([result.points]);

  if (result.motion) {
    addText(result.motion);
    lastAdded = result.motion;
    holding = result.motion;
    history = [];
    $("bigLetter").textContent = result.motion;
    highlightChip(result.motion);
    setRing(1, true);
    setStatus("added", `Added ${result.motion} (moving sign)`);
    return;
  }

  history.push(result.neighbors);
  if (history.length > SMOOTH_FRAMES) history.shift();

  const votes = {};
  let total = 0;
  for (const labels of history) {
    for (const lab of labels) {
      votes[lab] = (votes[lab] || 0) + 1;
      total += 1;
    }
  }
  const top3 = Object.entries(votes)
    .sort((a, b) => b[1] - a[1])
    .slice(0, 3)
    .map(([lab, count]) => [lab, count / total]);
  renderGuesses("guesses", top3);

  const [top, share] = top3[0];
  if (share < MIN_AGREEMENT) {
    holding = null;
    $("bigLetter").textContent = "–";
    highlightChip(null);
    setRing(0, false);
    $("hint").textContent = "Not sure yet – hold your sign still";
    setStatus("reading", "Reading…");
    return;
  }

  $("bigLetter").textContent = top;
  highlightChip(top);
  if (top !== holding || result.moving) {
    holding = top;
    holdStart = performance.now();
  }

  if (top === lastAdded) {
    setRing(1, true);
    $("hint").textContent = "Change sign, or lower your hand to repeat";
    setStatus("added", `Added ${top}`);
    return;
  }

  const progress = (performance.now() - holdStart) / HOLD_MS;
  setRing(progress, false);
  $("hint").textContent = result.moving ? "Hand moving – hold still to type" : `Keep holding to add ${top}`;
  setStatus("reading", `Recognised ${top}`);
  if (progress >= 1) {
    addText(top);
    lastAdded = top;
    setRing(1, true);
    setStatus("added", `Added ${top}`);
  }
}

async function postFrame(url, extra = {}) {
  grabber.width = video.videoWidth;
  grabber.height = video.videoHeight;
  grabber.getContext("2d").drawImage(video, 0, 0, grabber.width, grabber.height);
  const blob = await new Promise((resolve) => grabber.toBlob(resolve, "image/jpeg", 0.75));
  const form = new FormData();
  form.append("frame", blob, "frame.jpg");
  for (const [key, value] of Object.entries(extra)) form.append(key, value);
  const response = await fetch(url, { method: "POST", body: form });
  return response.json();
}

async function loop(id) {
  while (running && id === loopId) {
    if (mode === "words") {
      await sleep(100);
      continue;
    }
    try {
      const requested = mode;
      const result = await postFrame("/predict", { mode: requested });
      if (requested === mode && running && id === loopId) {
        handleResult(result);
      }
    } catch (err) {
      setStatus("error", "Cannot reach the server – is app.py running?");
      await sleep(1000);
    }
    await new Promise((resolve) => requestAnimationFrame(resolve));
  }
}

async function recordWord() {
  if (!running || recording) return;
  recording = true;
  $("recordBtn").disabled = true;
  $("stopBtn").disabled = true;
  $("repliesCard").classList.add("hidden");
  $("wordResult").textContent = "–";
  $("wordResult").classList.remove("unsure");
  renderGuesses("wordGuesses", []);

  for (let n = COUNTDOWN_FROM; n > 0; n--) {
    $("countdown").textContent = n;
    $("countdown").classList.remove("hidden");
    setStatus("reading", `Get ready… ${n}`);
    await sleep(700);
  }
  $("countdown").classList.add("hidden");

  const frames = [];
  const start = performance.now();
  setStatus("nohand", "Recording – sign your word now");
  while (performance.now() - start < RECORD_MS) {
    $("recordProgress").style.width = `${((performance.now() - start) / RECORD_MS) * 100}%`;
    try {
      const f = await postFrame("/word_frame");
      if (f.body) {
        frames.push({ shoulders: f.shoulders, left: f.left, right: f.right });
        drawHands(f.draw);
      } else {
        drawHands([]);
      }
    } catch (err) {
      setStatus("error", "Cannot reach the server – is app.py running?");
    }
  }
  $("recordProgress").style.width = "0%";
  drawHands([]);
  setStatus("reading", "Recognising…");

  try {
    const response = await fetch("/classify_word", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ frames }),
    });
    showWordResult(await response.json());
  } catch (err) {
    setStatus("error", "Cannot reach the server – is app.py running?");
  }
  recording = false;
  $("recordBtn").disabled = false;
  $("stopBtn").disabled = false;
}

function showWordResult(result) {
  if (!result.ok) {
    $("wordResult").textContent = "–";
    setStatus("nohand", result.error);
    return;
  }
  const [top] = result.ranked[0];
  const sure = result.ranked.length < 2 || result.ranked[1][1] < 0.85;
  $("wordResult").textContent = sure ? top : `${top}?`;
  $("wordResult").classList.toggle("unsure", !sure);
  renderGuesses("wordGuesses", result.ranked);
  highlightChip(top);
  addWord(top);
  setStatus("added", sure ? `Recognised “${top}”` : `Not sure – maybe “${top}”`);

  $("replies").innerHTML = result.replies.map((r) => `
    <div class="reply" title="Click to add this reply to the text" data-word="${r.word}">
      <video src="${r.clip}" autoplay loop muted playsinline></video>
      <div><strong>${r.text}</strong><span>Sign: ${r.word}</span></div>
    </div>`).join("");
  document.querySelectorAll(".reply").forEach((card) => {
    card.addEventListener("click", () => addWord(card.dataset.word));
  });
  $("repliesCard").classList.toggle("hidden", result.replies.length === 0);
}

function setMode(newMode) {
  if (recording || newMode === mode) return;
  mode = newMode;
  document.querySelectorAll(".tab").forEach((tab) => tab.classList.toggle("active", tab.dataset.mode === mode));
  $("modeNote").textContent = NOTES[mode] || "";
  $("modeNote").classList.toggle("hidden", !NOTES[mode]);
  $("signPanel").classList.toggle("hidden", mode === "words");
  $("wordPanel").classList.toggle("hidden", mode !== "words");
  $("repliesCard").classList.add("hidden");
  resetSignState();
  drawHands([]);
  if (running) {
    setStatus("idle", mode === "words" ? "Press Record, then sign one word" : "Show your hand");
  }
}

async function startCamera() {
  try {
    const stream = await navigator.mediaDevices.getUserMedia({
      video: { width: 640, height: 480 },
      audio: false,
    });
    video.srcObject = stream;
    await video.play();
    overlay.width = video.videoWidth;
    overlay.height = video.videoHeight;
    $("startScreen").classList.add("hidden");
    $("stopBtn").classList.remove("hidden");
    $("camError").textContent = "";
    running = true;
    loopId += 1;
    setStatus("idle", mode === "words" ? "Press Record, then sign one word" : "Show your hand");
    loop(loopId);
  } catch (err) {
    $("camError").textContent = "Could not open the camera: " + err.message;
  }
}

function stopCamera() {
  if (recording) return;
  running = false;
  if (video.srcObject) video.srcObject.getTracks().forEach((track) => track.stop());
  video.srcObject = null;
  resetSignState();
  drawHands([]);
  $("stopBtn").classList.add("hidden");
  $("startScreen").classList.remove("hidden");
  setStatus("idle", "Camera off");
}

$("startBtn").addEventListener("click", startCamera);
$("stopBtn").addEventListener("click", stopCamera);
$("recordBtn").addEventListener("click", recordWord);
$("spaceBtn").addEventListener("click", () => addText(" "));
$("deleteBtn").addEventListener("click", () => { text = text.slice(0, -1); renderText(); });
$("clearBtn").addEventListener("click", () => { text = ""; renderText(); });
$("copyBtn").addEventListener("click", async () => {
  await navigator.clipboard.writeText(text);
  $("copyBtn").textContent = "Copied";
  setTimeout(() => { $("copyBtn").textContent = "Copy"; }, 1200);
});
document.querySelectorAll(".tab").forEach((tab) => {
  tab.addEventListener("click", () => setMode(tab.dataset.mode));
});
document.querySelectorAll("button").forEach((btn) => {
  btn.addEventListener("click", () => btn.blur());
});

document.addEventListener("keydown", (event) => {
  if (event.code === "Space") {
    event.preventDefault();
    addText(" ");
  } else if (event.code === "Backspace") {
    event.preventDefault();
    text = text.slice(0, -1);
    renderText();
  } else if (event.code === "Enter" && mode === "words") {
    event.preventDefault();
    recordWord();
  }
});

renderText();
