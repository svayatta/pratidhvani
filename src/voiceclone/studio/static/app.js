"use strict";
const $ = (id) => document.getElementById(id);
const BUSY = new Set(["uploaded", "queued", "cleaning", "transcribing"]);
const TRAIN_BUSY = new Set(["queued", "preparing", "training"]);
const STATUS_TEXT = { uploaded: "Uploading", queued: "Waiting", cleaning: "Cleaning", transcribing: "Writing words",
  ready: "Ready", failed: "Failed" };
let config = { languages: {} };
let current = null;      // id of the voice on screen
let chosen = null;       // File or Blob for the new voice
let pollTimer = null;
let panelsFor = null;    // which voice (and trained state) the option panels were drawn for
let shownFor = null;     // id of the voice that the voice screen has content for
let layoutFor = null;    // which voice (and layout) the cards were put in order for

async function api(path, opts = {}) {
  const r = await fetch(path, opts);
  if (!r.ok) {
    let msg = r.statusText;
    try { const j = await r.json(); msg = typeof j.detail === "string" ? j.detail : (j.detail?.[0]?.msg || msg); } catch {}
    throw new Error(msg);
  }
  return r.status === 204 ? null : r.json();
}
const el = (tag, props = {}, ...kids) => { const e = Object.assign(document.createElement(tag), props); e.append(...kids); return e; };
const fmtSec = (s) => s == null ? "" : `${Math.floor(s / 60)}:${String(Math.round(s % 60)).padStart(2, "0")}`;

// ---------- options (schema from /api/config, see studio/options.py) ----------
const defaultsOf = (group) => Object.fromEntries(config.options[group].fields.map((f) => [f.key, f.default]));

function renderOptions(id, group, values, unavailable = {}) {
  const g = config.options[group];
  const v = { ...defaultsOf(group), ...(values || {}) };
  // A field that this server or this voice cannot use does not show. The server keeps its default for it.
  const fields = g.fields.filter((f) => f.available && !unavailable[f.key]);
  const field = (f) => {
    let input;
    if (f.type === "bool") input = el("input", { type: "checkbox", checked: !!v[f.key] });
    else if (f.type === "choice") input = el("select", {}, ...f.choices.map(([k, l]) => el("option", { value: k, textContent: l })));
    else input = el("input", { type: "number", min: f.min, max: f.max, step: f.step ?? 1, value: v[f.key] });
    if (f.type === "choice") input.value = v[f.key];
    Object.assign(input.dataset, { key: f.key, type: f.type });
    const hint = f.help ? el("span", { className: "hint", textContent: f.help }) : "";
    return f.type === "bool"
      ? el("label", { className: "opt check" }, input, el("span", {}, f.label, el("br"), hint))
      : el("label", { className: "opt" }, el("span", { textContent: f.label }), input, hint);
  };
  // A setting that depends on a switch sits beside that switch, and shows only when the switch is on.
  const grouped = fields.some((f) => f.needs);
  const grid = el("div", { className: `opt-grid ${grouped ? "rows" : ""}` });
  for (const f of fields.filter((f) => !f.needs)) {
    if (!grouped) { grid.append(field(f)); continue; }
    grid.append(el("div", { className: "opt-row" }, field(f), el("div", { className: "opt-kids" }, ...fields.filter((k) => k.needs === f.key).map(field))));
  }
  const sync = () => grid.querySelectorAll(".opt-row").forEach((r) => {
    const main = r.querySelector(":scope > .opt [data-key]");
    r.querySelector(".opt-kids").hidden = main.type === "checkbox" && !main.checked;
  });
  const head = el("div", { className: "opt-head" }, el("h4", { textContent: g.label }),
    el("button", { type: "button", className: "btn btn-quiet small", textContent: "Reset", onclick: () => renderOptions(id, group, null, unavailable) }));
  grid.addEventListener("change", sync);
  $(id).replaceChildren(head, grid);
  sync();
}

function readOptions(id) {
  const o = {};
  $(id).querySelectorAll("[data-key]").forEach((i) => {
    o[i.dataset.key] = i.dataset.type === "bool" ? i.checked : i.dataset.type === "choice" ? i.value : Number(i.value);
  });
  return o;
}

function describeClean(c) {
  if (!c) return "";
  const on = [c.declip && "clipped peaks repaired", c.highpass && `rumble cut below ${c.highpass_hz} Hz`,
    c.denoise && `hiss reduced by ${c.denoise_db} dB`, c.enhance && `AI speech enhancement${c.super_res ? " with high frequencies restored" : ""}`,
    c.normalize && "level evened out"].filter(Boolean);
  return `Cleaning done: ${on.length ? on.join(", ") : "none"}.`;
}

// ---------- list ----------
async function refreshList() {
  const voices = await api("/api/voices");
  const ul = $("voice-list");
  ul.replaceChildren(...voices.map((v) => {
    const t = v.train.status === "trained" ? "trained" : v.status;
    const cls = t === "ready" || t === "trained" ? t : t === "failed" ? "failed" : "busy";
    const label = t === "trained" ? "Trained" : STATUS_TEXT[t] || t;
    const b = el("button", { type: "button", onclick: () => go(`#/voice/${v.id}`) },
      el("span", { className: "vn", textContent: v.name }), el("span", { className: `pill ${cls}`, textContent: label }));
    if (v.id === current) b.setAttribute("aria-current", "true");
    return el("li", {}, b);
  }));
  $("empty-list").hidden = voices.length > 0;
}

// ---------- routing ----------
function go(hash) { if (location.hash === hash) route(); else location.hash = hash; }
function route() {
  clearTimeout(pollTimer);
  activeRec?.cancel();   // a recording does not continue on a screen that the user left
  const m = location.hash.match(/^#\/voice\/([0-9a-f]{12})$/);
  current = m ? m[1] : null;
  // Each screen shows only when it has its content: a reload does not show one screen and then the other.
  $("view-create").hidden = !!m;
  $("view-voice").hidden = !m || shownFor !== current;
  if (m) loadVoice(); else $("f-name").focus();
  refreshList().catch(() => {});
}

// ---------- create ----------
function setChosen(fileOrBlob, name, recorded = false) {
  chosen = fileOrBlob;
  $("chosen-again").hidden = !recorded;
  $("drop-idle").hidden = !!fileOrBlob;
  $("drop-chosen").hidden = !fileOrBlob;
  $("drop-recording").hidden = true;
  if (fileOrBlob) {
    $("chosen-name").textContent = name;
    $("chosen-audio").src = URL.createObjectURL(fileOrBlob);
  } else {
    $("chosen-audio").removeAttribute("src");
    $("f-file").value = "";
  }
}

function setupDrop() {
  const drop = $("drop");
  ["dragenter", "dragover"].forEach((t) => drop.addEventListener(t, (e) => { e.preventDefault(); drop.classList.add("over"); }));
  ["dragleave", "drop"].forEach((t) => drop.addEventListener(t, (e) => { e.preventDefault(); drop.classList.remove("over"); }));
  drop.addEventListener("drop", (e) => { const f = e.dataTransfer.files[0]; if (f) setChosen(f, f.name); });
  $("f-file").addEventListener("change", (e) => { const f = e.target.files[0]; if (f) setChosen(f, f.name); });
  $("chosen-clear").addEventListener("click", () => setChosen(null));
}

// ---------- recording ----------
// One recording panel: the time, a live level, a bar for the length, a hint, Stop and Cancel.
// The "New voice" screen and step 4 each make one, with their own limits.
//   maxSec: the recording stops here.   zone: [from, to] seconds marked as the best length, or null.
//   scale: the three labels below the bar.   hint(seconds): the text for this moment.
//   onStart(), onDone(file, seconds), onCancel(), onDenied(): what the screen does.
let activeRec = null;   // the panel that records now; route() cancels it when the user leaves the screen
function recordPanel({ maxSec, zone, scale, hint, stopLabel, onStart, onDone, onCancel, onDenied }) {
  const time = el("p", { className: "rec-time", textContent: "0:00" });
  const cv = el("canvas", { className: "rec-wave", ariaHidden: "true" });
  const fill = el("span", { className: "rec-fill" });
  const track = el("div", { className: "rec-track", ariaHidden: "true" }, fill);
  if (zone) track.prepend(el("span", { className: "rec-zone", style: `left:${zone[0] / maxSec * 100}%;width:${(zone[1] - zone[0]) / maxSec * 100}%` }));
  const note = el("p", { className: "small muted rec-hint", role: "status" });
  let rec = null, cancelled = false;
  const stop = (cancel) => { if (rec && rec.state !== "inactive") { cancelled = cancel; rec.stop(); } };
  const panel = el("div", { className: "rec" },
    el("p", { className: "drop-title" }, el("span", { className: "dot live" }), "Recording"), time, cv, track,
    el("div", { className: "rec-scale muted small", ariaHidden: "true" }, ...scale.map((t) => el("span", { textContent: t }))), note,
    el("div", { className: "drop-actions" },
      el("button", { type: "button", className: "btn btn-ink", textContent: stopLabel, onclick: () => stop(false) }),
      el("button", { type: "button", className: "btn btn-quiet", textContent: "Cancel", onclick: () => stop(true) })));

  async function start() {
    if (activeRec) return;
    // The audio context starts here, in the click, before the wait for the microphone. A context that starts
    // later can stay suspended, and then the level reads as silence.
    const ctx = new AudioContext();
    ctx.resume().catch(() => {});
    let stream;
    try { stream = await navigator.mediaDevices.getUserMedia({ audio: { echoCancellation: false, noiseSuppression: false, autoGainControl: false } }); }
    catch { ctx.close(); return onDenied("The browser did not allow the microphone. Allow it in the address bar, then press Record again."); }
    const parts = [];
    cancelled = false;
    rec = new MediaRecorder(stream);
    const me = rec, self = { cancel: () => stop(true) };
    activeRec = self;
    onStart();

    // live level: one bar for each 1/15 s, newest on the right. It shows that the microphone hears the speaker.
    // Some browsers do not run nodes that do not reach the output, so the level goes to the output at zero volume.
    const an = ctx.createAnalyser(), src = ctx.createMediaStreamSource(stream), mute = ctx.createGain();
    mute.gain.value = 0;
    src.connect(an); an.connect(mute); mute.connect(ctx.destination);
    const buf = new Float32Array(an.fftSize), levels = [];
    const g = cv.getContext("2d"), dpr = window.devicePixelRatio || 1;
    cv.width = cv.clientWidth * dpr; cv.height = cv.clientHeight * dpr;
    const accent = getComputedStyle(document.documentElement).getPropertyValue("--accent");
    let peak = 0, frame = 0;
    const draw = () => {
      if (me.state === "inactive") return;
      requestAnimationFrame(draw);
      if (ctx.state === "suspended") ctx.resume().catch(() => {});
      an.getFloatTimeDomainData(buf);
      let m = 0, sum = 0;
      for (const x of buf) { m = Math.max(m, Math.abs(x)); sum += x * x; }
      peak = Math.max(peak, m);
      if (frame++ % 4) return;
      // The bar height follows the level in decibels: -60 dB is the floor, -15 dB is the full height.
      // The recording has no automatic gain, so a raw level is small and a linear bar stays flat.
      const db = 20 * Math.log10(Math.sqrt(sum / buf.length) + 1e-9);
      levels.push(Math.max(0, Math.min(1, (db + 60) / 45)));
      const step = 6 * dpr, n = Math.floor(cv.width / step);
      if (levels.length > n) levels.shift();
      g.clearRect(0, 0, cv.width, cv.height);
      g.fillStyle = accent;
      levels.forEach((l, i) => {
        const hgt = Math.max(2 * dpr, l * cv.height);
        g.fillRect(i * step, (cv.height - hgt) / 2, 4 * dpr, hgt);
      });
    };
    // The recorder starts before the first draw: draw() stops when the recorder is inactive.
    me.start(1000);   // a part each second: a long recording does not stay in one large buffer
    draw();

    const t0 = Date.now();
    const secs = () => (Date.now() - t0) / 1000;
    const tick = () => {
      const s = secs(), silent = s > 3 && peak < 0.001;   // 0.001 is -60 dB: a live microphone is above this in a quiet room
      time.textContent = fmtSec(Math.floor(s));
      fill.style.width = `${Math.min(100, s / maxSec * 100)}%`;
      note.textContent = silent ? "The microphone hears no sound. Check that it is on and not muted." : hint(s);
      note.className = `small rec-hint ${silent ? "error" : "muted"}`;
      if (s >= maxSec) me.stop();
    };
    tick();
    const timer = setInterval(tick, 250);
    me.ondataavailable = (e) => parts.push(e.data);
    me.onstop = () => {
      const s = secs();
      stream.getTracks().forEach((t) => t.stop());
      clearInterval(timer); ctx.close();
      if (activeRec === self) activeRec = null;
      if (cancelled) return onCancel();
      const type = me.mimeType || "audio/webm";
      const ext = type.includes("mp4") ? "mp4" : type.includes("ogg") ? "ogg" : "webm";
      onDone(new File(parts, `recording.${ext}`, { type }), s);
    };
  }
  return { el: panel, start };
}

const REC_MIN_SEC = 5;
const sampleRec = recordPanel({
  maxSec: 45, zone: [15, 30], scale: ["0:00", "15 to 30 seconds is best", "0:45"], stopLabel: "Stop and use",
  hint: (s) => s < REC_MIN_SEC ? "Speak naturally. The minimum is 5 seconds."
    : s < 15 ? "Good. Continue for a few more sentences."
    : s < 30 ? "This is a good length. Stop when you finish the sentence."
    : "The recording stops at 45 seconds.",
  onStart: () => { showError("create-error", ""); $("drop-idle").hidden = true; $("drop-chosen").hidden = true; $("drop-recording").hidden = false; },
  onDone: (file, s) => {
    setChosen(file, `Recording, ${fmtSec(s)}`, true);
    if (s < REC_MIN_SEC) showError("create-error", `The recording is ${Math.round(s)} seconds long. The minimum is ${REC_MIN_SEC} seconds. Record again.`);
  },
  onCancel: () => setChosen(null),
  onDenied: (msg) => showError("create-error", msg),
});

// Step 4: longer recordings for training. The user listens to each recording before it goes to the server.
let trainFile = null;
const TRAIN_REC_MAX_SEC = 600;
const trainRec = recordPanel({
  maxSec: TRAIN_REC_MAX_SEC, zone: null, scale: ["0:00", "one recording, up to 10 minutes", "10:00"], stopLabel: "Stop",
  hint: (s) => s < 30 ? "Read a book or a newspaper aloud, in the usual way of the speaker."
    : s < TRAIN_REC_MAX_SEC - 30 ? "Continue. Short pauses are fine. You can make more than one recording."
    : "The recording stops at 10 minutes.",
  onStart: () => { showError("v-train-error", ""); $("v-rec-review").hidden = true; $("v-rec-panel").hidden = false; },
  onDone: (file, s) => {
    trainFile = file;
    $("v-rec-panel").hidden = true; $("v-rec-review").hidden = false;
    $("v-rec-name").textContent = `Recording, ${fmtSec(s)}`;
    $("v-rec-audio").src = URL.createObjectURL(file);
  },
  onCancel: () => { $("v-rec-panel").hidden = true; },
  onDenied: (msg) => showError("v-train-error", msg),
});

async function addRecordings(files) {
  const fd = new FormData();
  for (const f of files) {
    if (f.size > config.recording_max_mb * 1048576) return showError("v-train-error", `${f.name} is over ${config.recording_max_mb} MB.`);
    fd.append("files", f, f.name);
  }
  $("v-train-status").textContent = "Uploading…";
  showError("v-train-error", "");
  let ok = true;
  try { await api(`/api/voices/${current}/recordings`, { method: "POST", body: fd }); } catch (err) { showError("v-train-error", err.message); ok = false; }
  loadVoice();
  return ok;
}

async function submitCreate(e) {
  e.preventDefault();
  const name = $("f-name").value.trim();
  if (!name) return showError("create-error", "Give the voice a name.");
  if (!chosen) return showError("create-error", "Add a voice sample first.");
  if (chosen.size > config.sample_max_mb * 1048576) return showError("create-error", `The sample is over ${config.sample_max_mb} MB.`);
  const fd = new FormData();
  fd.append("name", name);
  fd.append("lang", $("f-lang").value);
  fd.append("options", JSON.stringify({ clean: readOptions("o-create-clean"), reference: readOptions("o-create-reference") }));
  fd.append("file", chosen, chosen.name || "sample.webm");
  const btn = $("create-submit");
  btn.disabled = true; btn.textContent = "Uploading…";
  try {
    const v = await api("/api/voices", { method: "POST", body: fd });
    $("create-form").reset(); setChosen(null); showError("create-error", "");
    renderCreateOptions();
    go(`#/voice/${v.id}`);
  } catch (err) { showError("create-error", err.message); }
  finally { btn.disabled = false; btn.textContent = "Create voice"; }
}

// A yes or no question in a dialog. Cancel has the focus first, and Escape is the same as Cancel.
function ask(title, text, yes) {
  const d = $("ask");
  $("ask-title").textContent = title; $("ask-text").textContent = text; $("ask-yes").textContent = yes;
  d.returnValue = "no";
  d.showModal();
  return new Promise((done) => d.addEventListener("close", () => done(d.returnValue === "yes"), { once: true }));
}

function showError(id, msg) { $(id).textContent = msg; $(id).hidden = !msg; }

// ---------- voice ----------
async function loadVoice() {
  const id = current;
  let v;
  try { v = await api(`/api/voices/${id}`); }
  catch { location.hash = ""; return; }
  if (id !== current) return;
  renderVoice(v);
  const busy = BUSY.has(v.status) || TRAIN_BUSY.has(v.train.status) || v.previews.some((p) => p.status === "queued" || p.status === "running");
  if (busy) pollTimer = setTimeout(() => { loadVoice(); refreshList().catch(() => {}); }, 1500);
}

function setAudio(id, url) { const a = $(id); if (a.dataset.src !== url) { a.dataset.src = url; a.src = url; } }

// The summary line of one card. `num`: "done", a step number, or null for no number. `now`: the step in work.
function setFold(id, { num = null, title, why = "", pill = "", pillCls = "", locked = false, fixed = false, now = false }) {
  const d = $(id), [n, b, w, p] = d.querySelector("summary").children;
  n.hidden = num == null;
  n.className = `num ${num === "done" ? "done" : now ? "now" : ""}`;
  n.textContent = num === "done" ? "✓" : num ?? "";
  b.textContent = title; w.textContent = why;
  p.hidden = !pill; p.textContent = pill; p.className = `pill ${pillCls}`;
  d.toggleAttribute("data-locked", locked);
  d.toggleAttribute("data-fixed", fixed);
  d.querySelector("summary").setAttribute("aria-disabled", String(locked));
}

function renderVoice(v) {
  const base = `/api/voices/${v.id}/audio`;
  shownFor = v.id; $("view-voice").hidden = false;
  $("v-name").textContent = v.name;
  $("v-meta").textContent = config.languages[v.lang] || v.lang;

  // The cards are the four steps of a voice, always in the order that the user does them:
  // sample, words, hear the voice, improve. A voice goes through three layouts:
  //   work:  the sample is not ready (or it failed). The progress line and the sample show.
  //   guide: the sample is ready, the user did not confirm the words. Step 2 is open, step 3 is locked.
  //   use:   the words are confirmed. Steps 1 and 2 are closed lines, step 3 is open.
  const ready = v.status === "ready";
  const mode = !ready ? "work" : v.ref_text_by === "user" ? "use" : "guide";
  const guide = mode === "guide";

  const at = { uploaded: 1, queued: 1, cleaning: 1, transcribing: 2, ready: 4, failed: -1 }[v.status];
  const labels = ["Uploaded", "Cleaning noise", "Writing down the words", "Ready"];
  $("v-steps").hidden = ready;
  $("v-steps").replaceChildren(...labels.map((l, i) => {
    const cls = i === 0 || i < at || at === 4 ? "done" : i === at ? "now" : "";
    return el("li", { className: cls, textContent: l });
  }));
  if (v.status === "failed") $("v-steps").lastChild.className = "fail";
  showError("v-error", v.status === "failed" ? `The sample failed: ${v.error}. Clean it again with other options, or delete this voice and add a new sample.` : "");

  const trainedNow = v.train.status === "trained";
  if (panelsFor !== `${v.id}:${trainedNow}`) {   // draw once per voice, so polling never wipes edits
    panelsFor = `${v.id}:${trainedNow}`;
    renderOptions("o-voice-clean", "clean", v.options?.clean);
    renderOptions("o-voice-reference", "reference", v.options?.reference);
    renderOptions("o-generate", "generate", null, trainedNow ? {} : { trained: "Train the voice first" });
    renderOptions("o-train", "train", v.train.options);
    ["v-ref-error", "v-try-error", "v-train-error"].forEach((i) => showError(i, ""));
  }
  if (layoutFor !== `${v.id}:${mode}`) {   // the open state changes with the layout only, not on each poll
    layoutFor = `${v.id}:${mode}`;
    const [audio, ref, say, train] = ["v-audio-card", "v-ref-card", "v-try-card", "v-train-card"].map($);
    audio.open = mode === "work"; ref.open = guide; say.open = mode === "use"; train.open = false;
  }

  const cleaned = v.sample_sec != null;
  $("v-done").classList.toggle("two", mode === "use");   // steps 1 and 2 are done: two short rows side by side
  $("v-audio-card").hidden = !v.original;
  setFold("v-audio-card", {
    num: ready ? "done" : null,
    title: ready ? "Sample cleaned" : "Your sample",
    why: !cleaned ? "" : mode === "use" ? `${v.sample_sec} seconds` : `${v.sample_sec} seconds` });
  $("v-clean-used").textContent = describeClean(v.options?.clean);
  $("v-reclean").hidden = BUSY.has(v.status);
  if (v.original) setAudio("a-original", `${base}/original`);
  if (cleaned) setAudio("a-clean", `${base}/clean`);
  $("a-clean").parentElement.hidden = !cleaned;

  $("v-ref-card").hidden = !ready;
  $("v-try-card").hidden = !ready;
  $("v-train-card").hidden = !ready;
  if (!ready) return;

  setFold("v-ref-card", guide
    ? { num: "2", now: true, title: "Check the words", why: "The studio wrote down what it heard" }
    : { num: "done", title: "Words checked", why: "Open to correct them" });
  $("v-ref-save").textContent = guide ? "The words are correct" : "Save words";
  $("v-ref-save").className = `btn ${guide ? "btn-accent" : "btn-line"}`;
  setAudio("a-reference", `${base}/reference`);
  const ta = $("v-ref-text");
  if (document.activeElement !== ta && ta.dataset.v !== v.ref_text) { ta.value = v.ref_text || ""; ta.dataset.v = v.ref_text; }
  ["v-ref-text", "v-say"].forEach((i) => $(i).lang = v.lang);

  setFold("v-try-card", guide
    ? { num: "3", title: "Hear the voice", why: "Check the words first", locked: true }
    : { num: "3", now: true, title: "Hear the voice", why: "Type a text and the voice says it", fixed: true });
  $("v-say").placeholder = `Type a sentence in ${config.languages[v.lang] || "the language of the voice"}`;
  const trained = v.train.status === "trained";
  // One group for each text, the newest text first. In a group, the takes are in the order they were made,
  // so "Make another version" adds a row at the end of its group and no group moves.
  const byText = new Map();
  [...v.previews].reverse().forEach((p) => byText.set(p.text, [...(byText.get(p.text) || []), p]));
  const groups = [...byText.values()].reverse();
  const ul = $("v-previews");
  syncRows(ul, groups, (g) => g[0].id, (g) => groupRow(v, g[0].text));
  groups.forEach((g, n) => {
    const li = ul.children[n];
    // The page shows the newest finished take and each take after it. The takes before it go to the history.
    const cut = Math.max(0, g.findLastIndex((p) => p.status === "done"));
    const takes = g.map((p, t) => ({ p, t }));
    const rows = (sel, items) => syncRows(li.querySelector(sel), items, ({ p }) => `${p.id}:${p.status}`, ({ p, t }) => takeRow(v, p, t + 1));
    rows(".takes.cur", takes.slice(cut));
    rows(".takes.old", takes.slice(0, cut));
    li.querySelector(".hist").hidden = !cut;
    li.querySelector(".hist span").textContent = `${cut} earlier version${cut > 1 ? "s" : ""}`;
    if (!cut) li.querySelector(".takes.old").hidden = true;
    li.querySelector(".again").onclick = () =>
      generate(v.id, g[0].text, { ...readOptions("o-generate"), seed: Math.max(...g.map((p) => p.seed)) + 1, takes: 1 });
  });
  $("v-no-takes").hidden = v.previews.length > 0;
  renderNow(v);

  const t = v.train;
  const tBusy = TRAIN_BUSY.has(t.status);
  const st = { none: v.recordings.length ? "" : "Record or add files first. Then you can start training.", queued: "Waiting for other work to finish…",
    preparing: "Preparing the recordings: cleaning, writing down the words, removing other speakers…", training: "Training… this takes several hours.",
    trained: `Trained on ${t.speech_min ?? "?"} min of speech. New results use the trained voice.`, failed: `Training failed: ${t.error}` }[t.status];
  setFold("v-train-card", {
    num: "4", title: "Improve the voice",
    why: tBusy ? st : trained ? `Trained on ${t.speech_min ?? "?"} min of speech` : t.status === "failed" ? "Open to see the error"
      : `Add ${config.min_train_minutes ?? 8} minutes or more of this person's speech, for a closer voice`,
    pill: tBusy ? "Training" : trained ? "Trained" : t.status === "failed" ? "Failed" : "Optional",
    pillCls: tBusy ? "busy" : trained ? "trained" : t.status === "failed" ? "failed" : "" });
  $("v-recordings").replaceChildren(...v.recordings.map((r) =>
    el("li", {}, el("span", { textContent: r.file.replace(/^\d{3}_/, "") }), el("span", { className: "muted", textContent: fmtSec(r.sec) }))));
  $("v-train").disabled = tBusy || !v.recordings.length;
  $("v-train").textContent = trained ? "Train again" : "Start training";
  $("v-rec-files").disabled = tBusy; $("v-rec-start").disabled = tBusy; $("v-rec-add").disabled = tBusy;
  $("v-train-status").textContent = st || "";
  $("v-train-status").className = `small ${t.status === "failed" ? "error" : ""}`;
  $("v-train-log").hidden = !t.log.length;
  $("v-train-log").textContent = t.log.join("\n");
  $("v-train-log").scrollTop = 1e9;
}

const runSince = {};   // take id -> when this page first saw it generate

// The progress panel below the Generate button: the take in work, its step, and the time.
// `stage`, `sentence` and `sentences` come from jobs.py.
function renderNow(v) {
  const waiting = v.previews.filter((p) => p.status === "queued");
  const p = v.previews.find((x) => x.status === "running") || waiting.at(-1);   // the oldest in the queue is next
  const was = $("v-now").hidden;
  $("v-now").hidden = !p;
  if (!p) return;
  if (was) $("v-now").scrollIntoView({ block: "nearest", behavior: "smooth" });
  const run = p.status === "running", n = p.sentences || 1, done = p.sentence || 0;
  const stage = !run ? "wait" : p.stage || "speaking";
  $("now-title").textContent = run ? "Generating" : "Waiting for other work to finish";
  $("now-time").textContent = run ? fmtSec((Date.now() - (runSince[p.id] ??= Date.now())) / 1000) : "";
  $("now-text").textContent = p.text; $("now-text").lang = v.lang;
  const bar = $("now-bar");
  if (run && n > 1) { bar.max = n; bar.value = done; } else bar.removeAttribute("value");   // no value: the bar moves without a measure
  bar.setAttribute("aria-label", n > 1 ? `${done} of ${n} sentences done` : "Generating");
  const steps = [["wait", "Wait for a turn"], ["loading", "Load the voice model"],
    ["speaking", n > 1 ? `Speak sentence ${Math.min(done + 1, n)} of ${n}` : "Speak the sentence"], ["joining", "Join the sentences"]]
    .filter(([k]) => k !== "joining" || n > 1);
  const at = steps.findIndex(([k]) => k === stage);
  $("now-steps").replaceChildren(...steps.map(([, label], i) => el("li", { className: i < at ? "done" : i === at ? "now" : "", textContent: label })));
  const more = waiting.length - (run ? 0 : 1);
  $("now-note").textContent = [
    stage === "loading" ? "The first time after the studio starts, the model must load. This takes some minutes. After that, each text is much faster." : "",
    more > 0 ? `${more} more wait${more > 1 ? "" : "s"} for a turn.` : "",
  ].filter(Boolean).join(" ");
  $("now-note").hidden = !$("now-note").textContent;
}

// Make `parent` hold one child for each item, in order. A child whose key did not change stays in the page:
// a node that is put in again stops its audio.
function syncRows(parent, items, key, make) {
  const have = new Map([...parent.children].map((c) => [c.dataset.key, c]));
  items.forEach((it, n) => {
    const k = String(key(it)), node = have.get(k) || make(it);
    node.dataset.key = k;
    if (parent.children[n] !== node) parent.insertBefore(node, parent.children[n] || null);
  });
  while (parent.children.length > items.length) parent.lastChild.remove();
}

const DOWNLOAD_ICON = '<svg viewBox="0 0 24 24" width="18" height="18" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="M12 4v11"/><path d="m7 11 5 5 5-5"/><path d="M5 20h14"/></svg>';
const HISTORY_ICON = '<svg viewBox="0 0 24 24" width="15" height="15" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="M3 12a9 9 0 1 0 3-6.7L3 8"/><path d="M3 3v5h5"/><path d="M12 7v5l4 2"/></svg>';

function groupRow(v, text) {
  const btn = (cls, label, onclick) => el("button", { className: `btn btn-quiet small ${cls}`, textContent: label, onclick });
  // The history button is in the row of actions, at the right. It shows and hides the earlier versions below.
  const old = el("ol", { className: "takes old", hidden: true });
  const hist = el("button", { className: "btn btn-quiet small hist", innerHTML: `${HISTORY_ICON}<span></span>`, ariaExpanded: "false",
    onclick: () => { old.hidden = !old.hidden; hist.ariaExpanded = String(!old.hidden); } });
  return el("li", {}, el("p", { className: "pt", textContent: text, lang: v.lang }), el("ol", { className: "takes cur" }),
    el("div", { className: "take-acts" }, btn("again", "Make another version"),
      btn("", "Edit the text", () => { $("v-say").value = text; $("v-say").dispatchEvent(new Event("input")); $("v-say").focus(); }), hist),
    old);
}

// The settings that change the sound of a version most, as [label, value]. The other settings show only when
// they are not the default.
function usedSettings(p) {
  const d = defaultsOf("generate"), g = p.params || {};
  return [p.trained && ["Voice", "trained"], ["Variation", p.seed],
    g.temperature != null && ["Temperature", g.temperature],
    g.top_p != null && g.top_p !== d.top_p && ["Top-p", g.top_p],
    g.repetition_penalty != null && g.repetition_penalty !== d.repetition_penalty && ["Repetition penalty", g.repetition_penalty],
  ].filter(Boolean);
}

const PLAY_ICON = '<svg viewBox="0 0 24 24" width="16" height="16" fill="currentColor" aria-hidden="true"><path d="M8 5.5v13a1 1 0 0 0 1.5.86l11-6.5a1 1 0 0 0 0-1.72l-11-6.5A1 1 0 0 0 8 5.5z"/></svg>';
const PAUSE_ICON = '<svg viewBox="0 0 24 24" width="16" height="16" fill="currentColor" aria-hidden="true"><rect x="6" y="5" width="4.5" height="14" rx="1"/><rect x="13.5" y="5" width="4.5" height="14" rx="1"/></svg>';

// A small player: a play button, a position slider and the time. The browser's own player is wide and its
// menu repeats the download button. Only one player plays at a time.
function player(url, name) {
  const audio = el("audio", { preload: "metadata", src: url });
  const btn = el("button", { className: "play", type: "button", innerHTML: PLAY_ICON, ariaLabel: `Play ${name}` });
  const seek = el("input", { type: "range", min: 0, max: 1000, value: 0, ariaLabel: `Position in ${name}` });
  const time = el("span", { className: "ptime", textContent: "0:00" });
  const show = () => {
    const d = Number.isFinite(audio.duration) ? audio.duration : 0;
    seek.value = d ? audio.currentTime / d * 1000 : 0;
    seek.style.setProperty("--p", `${seek.value / 10}%`);   // the played part of the track, for the style
    time.textContent = `${fmtSec(Math.floor(audio.currentTime))} / ${fmtSec(d)}`;
  };
  const follow = () => { show(); if (!audio.paused) requestAnimationFrame(follow); };
  const state = (playing) => { btn.innerHTML = playing ? PAUSE_ICON : PLAY_ICON; btn.ariaLabel = `${playing ? "Pause" : "Play"} ${name}`; };
  btn.onclick = () => (audio.paused ? audio.play().catch(() => {}) : audio.pause());
  audio.onplay = () => {
    document.querySelectorAll("audio").forEach((a) => a !== audio && a.pause());
    state(true); follow();
  };
  audio.onpause = () => state(false);
  audio.onended = () => { audio.currentTime = 0; show(); };
  audio.onloadedmetadata = show;
  audio.onerror = () => { time.textContent = "Cannot play"; btn.disabled = true; };
  seek.oninput = () => { if (Number.isFinite(audio.duration)) { audio.currentTime = seek.value / 1000 * audio.duration; show(); } };
  return el("div", { className: "player" }, audio, btn, seek, time);
}

function takeRow(v, p, n) {
  const url = `/api/voices/${v.id}/audio/preview?id=${p.id}`;
  const li = el("li", {}, el("span", { className: "tn", textContent: `Version ${n}` }));
  if (p.status === "done") {
    li.append(
      el("span", { className: "sets" }, ...usedSettings(p).map(([k, val]) => el("span", { className: "set" }, `${k} `, el("b", { textContent: val })))),
      player(url, `version ${n}`),
      el("span", { className: "pm made", textContent: p.gen_time != null ? `Made in ${Math.round(p.gen_time)} s` : "" }),
      el("a", { className: "icon-btn", innerHTML: DOWNLOAD_ICON, href: url, download: `${v.name} version ${n}.wav`,
        ariaLabel: `Download version ${n}`, title: "Download" }));
  } else if (p.status === "failed") {
    li.append(el("span", { className: "error small wide-cell", textContent: `Failed: ${p.error || "unknown error"}` }),
      el("button", { className: "btn btn-quiet small", textContent: "Try again",
        onclick: () => generate(v.id, p.text, { ...readOptions("o-generate"), seed: p.seed, takes: 1 }) }));
  } else {
    li.append(el("span", { className: "pm wide-cell", textContent: p.status === "running" ? "Generating now. The progress is above." : "Waiting for its turn." }));
  }
  return li;
}

async function generate(id, text, options) {
  showError("v-try-error", "");
  try { await api(`/api/voices/${id}/previews`, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ text, options }) }); }
  catch (err) { showError("v-try-error", err.message); return false; }
  clearTimeout(pollTimer); loadVoice();
  return true;
}

function setupVoice() {
  $("v-say").addEventListener("input", () => {
    const n = $("v-say").value.trim().split(/(?<=[.?!।॥])(?:\s+|(?=[^\s\p{Nd}.?!।॥"'”’)\]]))/u).filter(Boolean).length;
    $("v-say-count").textContent = n > 1 ? `${n} sentences. The voice says them in order, with a short pause between them.` : "";
    $("v-say-len").textContent = `${$("v-say").value.length} / ${$("v-say").maxLength}`;
  });
  $("v-say").addEventListener("keydown", (e) => { if (e.key === "Enter" && (e.metaKey || e.ctrlKey)) $("v-generate").click(); });
  // a locked step does not open
  document.querySelectorAll(".fold > summary").forEach((sm) => sm.addEventListener("click", (e) => {
    if (sm.parentElement.matches("[data-locked], [data-fixed]")) e.preventDefault();
  }));
  $("v-generate").addEventListener("click", () => {
    const text = $("v-say").value.trim();
    if (!text) { showError("v-try-error", "Type what the voice should say."); return $("v-say").focus(); }
    generate(current, text, readOptions("o-generate")).then((ok) => {
      if (ok) { $("v-say").value = ""; $("v-say").dispatchEvent(new Event("input")); }   // the text is now in the take below
    });
  });
  // Saving the words also confirms them: the server records that the user set them.
  $("v-ref-save").addEventListener("click", async () => {
    const ref_text = $("v-ref-text").value.trim();
    if (!ref_text) { showError("v-ref-error", "Type the words you hear in the clip."); return $("v-ref-text").focus(); }
    const first = $("v-ref-save").classList.contains("btn-accent");
    try { await api(`/api/voices/${current}`, { method: "PATCH", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ ref_text }) }); }
    catch (err) { return showError("v-ref-error", err.message); }
    showError("v-ref-error", "");
    clearTimeout(pollTimer); await loadVoice();
    if (first) $("v-say").focus();
    else { $("ref-saved").hidden = false; setTimeout(() => ($("ref-saved").hidden = true), 1500); }
  });
  $("v-delete").addEventListener("click", async () => {
    const name = $("v-name").textContent;
    if (!await ask(`Delete "${name}"?`, "This removes the voice, its sample, its recordings and all its results. You cannot get them back.", "Delete voice")) return;
    await api(`/api/voices/${current}`, { method: "DELETE" });
    go("#/new");
  });
  $("v-rec-files").addEventListener("change", async (e) => {
    const files = [...e.target.files];
    e.target.value = "";
    addRecordings(files);
  });
  $("v-rec-panel").append(trainRec.el);
  $("v-rec-start").addEventListener("click", trainRec.start);
  const dropTrainFile = () => { trainFile = null; $("v-rec-review").hidden = true; $("v-rec-audio").removeAttribute("src"); };
  $("v-rec-discard").addEventListener("click", dropTrainFile);
  $("v-rec-add").addEventListener("click", async () => {
    if (trainFile && await addRecordings([trainFile])) dropTrainFile();
  });
  $("v-reclean-go").addEventListener("click", async () => {
    const body = { clean: readOptions("o-voice-clean"), reference: readOptions("o-voice-reference") };
    try { await api(`/api/voices/${current}/clean`, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) }); }
    catch (err) { return showError("v-error", err.message); }
    $("v-reclean").open = false;   // the panels keep what was just chosen
    clearTimeout(pollTimer); loadVoice();
  });
  $("v-train").addEventListener("click", async () => {
    const body = { options: readOptions("o-train") };
    try { await api(`/api/voices/${current}/train`, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) }); }
    catch (err) { return showError("v-train-error", err.message); }
    showError("v-train-error", "");
    clearTimeout(pollTimer); loadVoice();
  });
}

function renderCreateOptions() {
  renderOptions("o-create-clean", "clean");
  renderOptions("o-create-reference", "reference");
}

// ---------- start ----------
(async function init() {
  // The sidebar remembers its state on this browser. This runs before the first request, so the page does not jump.
  const setSide = (on) => {
    document.querySelector(".shell").classList.toggle("side-off", !on);
    const t = $("side-toggle"), label = on ? "Hide the list of voices" : "Show the list of voices";
    t.setAttribute("aria-expanded", String(on)); t.setAttribute("aria-label", label); t.title = label;
    $("side-open").disabled = on;   // the logo opens the sidebar only when the sidebar is narrow
    try { localStorage.setItem("side", on ? "on" : "off"); } catch {}
  };
  $("side-toggle").addEventListener("click", () => { setSide(false); $("side-open").focus(); });
  $("side-open").addEventListener("click", () => { setSide(true); $("side-toggle").focus(); });
  try { if (localStorage.getItem("side") === "off") setSide(false); } catch {}
  refreshList().catch(() => {});
  config = await api("/api/config");
  $("f-lang").replaceChildren(...Object.entries(config.languages).map(([k, n]) => el("option", { value: k, textContent: n })));
  $("max-mb").textContent = config.sample_max_mb;
  renderCreateOptions();
  if (config.min_train_minutes) $("min-min").textContent = config.min_train_minutes;
  setupDrop(); setupVoice();
  $("drop-recording").append(sampleRec.el);
  $("rec-start").addEventListener("click", sampleRec.start);
  $("chosen-again").addEventListener("click", sampleRec.start);
  $("v-keys").textContent = `or press ${/Mac|iPhone|iPad/.test(navigator.platform) ? "⌘" : "Ctrl"} Enter`;
  $("create-form").addEventListener("submit", submitCreate);
  $("new-voice").addEventListener("click", () => go("#/new"));
  window.addEventListener("hashchange", route);
  route();
  setInterval(() => refreshList().catch(() => {}), 5000);
})();
