"use strict";

const $ = (s) => document.querySelector(s);
const TITLES = { auto: "Auto", vector: "Vector (passages)", graph: "Graph (relationships)", hybrid: "Hybrid" };
const NOTES = { cache: "Answered instantly from an earlier identical question", rule: "Counted or listed directly from the knowledge graph" };
const DOC_ICON = '<svg viewBox="0 0 24 24" width="16" height="16" aria-hidden="true"><path d="M7 3h7l4 4v14H7zM14 3v4h4" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linejoin="round"/></svg>';
const reduceMotion = matchMedia("(prefers-reduced-motion: reduce)").matches;

function el(tag, attrs = {}, ...kids) {
  const n = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs)) {
    if (k === "class") n.className = v;
    else if (k === "text") n.textContent = v;
    else if (k.startsWith("on")) n.addEventListener(k.slice(2), v);
    else if (v !== false && v != null) n.setAttribute(k, v);
  }
  for (const kid of kids.flat()) if (kid != null) n.append(kid.nodeType ? kid : document.createTextNode(String(kid)));
  return n;
}
async function api(path, opts) {
  const r = await fetch(path, opts);
  let d = null;
  try { d = await r.json(); } catch { /* not JSON */ }
  if (!r.ok) {
    const detail = d && d.detail;
    throw new Error(typeof detail === "string" ? detail : detail ? JSON.stringify(detail) : `HTTP ${r.status}`);
  }
  return d;
}
const store = {  // localStorage may be blocked: never let it break the page
  get(k) { try { return localStorage.getItem(k); } catch { return null; } },
  set(k, v) { try { localStorage.setItem(k, v); } catch { /* ignore */ } },
};
const postJson = (body) => ({ method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) });

// ---------- toasts ----------
function toast(text, bad = false) {
  const t = el("div", { class: "toast" + (bad ? " bad" : ""), text });
  $("#toasts").append(t);
  setTimeout(() => t.remove(), 3700);
}

// ---------- sidebar ----------
const shell = () => $("#shell");
const isMobile = () => matchMedia("(max-width: 800px)").matches;
function setSide(open, remember = true) {
  shell().classList.toggle("collapsed", !open);
  $("#side-open").setAttribute("aria-expanded", String(open));
  $("#scrim").hidden = !(open && isMobile());
  if (remember && !isMobile()) store.set("side", open ? "open" : "closed");
}
const sideOpen = () => !shell().classList.contains("collapsed");

let knownDocs = new Set(), firstLoad = true;
async function loadDocs() {
  let docs = [];
  try { docs = (await api("/kb/docs")).documents; } catch { return 0; }
  const list = $("#doc-list");
  const fresh = [];
  list.replaceChildren(...docs.map((d) => {
    const isNew = !firstLoad && !knownDocs.has(d.source);
    if (isNew) fresh.push(d);
    const li = el("li", { class: "doc" + (isNew ? " new" : ""), "data-source": d.source });
    li.insertAdjacentHTML("afterbegin", DOC_ICON);  // constant markup, no user text
    li.append(el("div", { class: "nm" }, d.name, el("small", { text: `${d.chunks} chunks` })),
      el("span", { class: "tag" + (d.graph ? "" : " dim"), title: d.graph ? "Has a knowledge graph" : "Text search only", text: d.graph ? "graph" : "text" }));
    return li;
  }));
  knownDocs = new Set(docs.map((d) => d.source));
  firstLoad = false;
  $("#doc-empty").hidden = docs.length > 0;
  const c = $("#doc-count");
  if (c.textContent !== String(docs.length)) { c.textContent = docs.length; c.classList.add("bump"); setTimeout(() => c.classList.remove("bump"), 300); }
  for (const d of fresh) toast(`${d.name} is ready`);
  return docs.length;
}
function highlightDoc(source) {
  const docs = [...document.querySelectorAll(".doc")];
  const base = (x) => x.split("/").pop();
  // exact match first; then same file name (older ingests stored the path without "uploads/")
  const li = docs.find((n) => n.dataset.source === source) || docs.find((n) => base(n.dataset.source) === base(source));
  if (!li) { toast("That document is no longer in the list", true); return; }
  if (!sideOpen()) setSide(true, false);
  li.scrollIntoView({ block: "nearest", behavior: reduceMotion ? "auto" : "smooth" });
  li.classList.remove("pulse"); void li.offsetWidth; li.classList.add("pulse");
}

// ---------- suggestions ----------
async function loadChips() {
  const box = $("#chips");
  try {
    const qs = (await api("/kb/suggest")).questions;
    box.replaceChildren(...qs.map((q, i) => el("button", {
      type: "button", class: "chip", style: `animation-delay:${i * 60}ms`,
      onclick: () => { $("#question").value = q; $("#ask-form").requestSubmit(); },
    }, q)));
    $("#empty-text").textContent = qs.length ? "Try one of these, or ask your own question below." : "Add documents on the left, then ask a question below.";
  } catch { /* chips are optional */ }
}

// ---------- "starting up" notice: models load in the background after the server starts ----------
async function watchWarmup() {
  const note = $("#warm");
  if (!note) return;
  for (let i = 0; i < 90; i++) {  // at most ~3 minutes
    let st = null;
    try { st = await api("/kb/status"); } catch { /* server still starting */ }
    if (st && st.ready !== false) { note.hidden = true; return; }
    note.hidden = false;
    note.textContent = "Starting up" + (st && st.step ? ": " + st.step : "") + "… your first answer may take a little longer.";
    await new Promise((r) => setTimeout(r, 2000));
  }
  note.hidden = true;
}

// ---------- answers ----------
function reveal(node, text) {  // short typing effect, capped at ~0.7 s so it never slows reading
  if (reduceMotion || text.length < 40) { node.textContent = text; return; }
  const steps = Math.min(40, Math.ceil(text.length / 6)), per = Math.ceil(text.length / steps);
  let i = 0;
  node.classList.add("cursor");
  const tick = () => {
    i = Math.min(text.length, i + per);
    node.textContent = text.slice(0, i);
    if (i < text.length) setTimeout(tick, 18); else node.classList.remove("cursor");
    scrollDown();
  };
  tick();
}
const scrollDown = () => { const c = $("#chat"); c.scrollTop = c.scrollHeight; };

function reply(title, r, animate = true) {
  const ans = el("div", { class: "answer" });
  const copy = el("button", { type: "button", class: "copy", "aria-label": "Copy answer", onclick: async () => {
    try { await navigator.clipboard.writeText(r.answer); copy.textContent = "Copied"; } catch { copy.textContent = "Press Ctrl+C"; }
    setTimeout(() => (copy.textContent = "Copy"), 1500);
  } }, "Copy");
  const kids = [el("div", { class: "reply-head" }, el("h3", { text: title }), copy), ans];
  if (animate) reveal(ans, r.answer); else ans.textContent = r.answer;
  const badges = el("div", { class: "badges" });
  if (r.router && r.router.route) badges.append(el("span", { class: "badge", text: `Auto chose ${TITLES[r.router.route] || r.router.route}` }));
  else if (r.route) badges.append(el("span", { class: "badge", text: `Method: ${TITLES[r.route] || r.route}` }));
  if (r.escalated) badges.append(el("span", { class: "badge", text: "Widened the search to the documents" }));
  const note = r.cached ? NOTES.cache : r.guard === "rule" ? NOTES.rule : "";
  if (note) badges.append(el("span", { class: "badge", text: note }));
  kids.push(badges);
  if (r.grounded === false) kids.push(el("div", { class: "warn", text: "Some details may not be fully supported by your documents. Please check the sources." }));
  const seen = new Set(), chips = [];
  for (const s of r.sources || []) {
    if (!s.source || seen.has(s.source)) continue;
    seen.add(s.source);
    chips.push(el("button", { type: "button", title: "Show this document in the list", onclick: () => highlightDoc(s.source) },
      s.source.replace(/^uploads\//, "")));
  }
  if (chips.length) kids.push(el("div", { class: "src", "aria-label": "Sources" }, chips));
  if (r.latency_ms != null) kids.push(el("div", { class: "meta", text: `${(r.latency_ms / 1000).toFixed(1)} s` }));
  return el("article", { class: "reply" }, kids);
}

function addTurn(question) {
  $("#empty")?.remove();
  const q = el("div", { class: "msg-user", text: question });
  const box = el("div", { class: "replies" },
    el("div", { class: "reply", "aria-label": "Thinking" }, el("span", { class: "dots" }, el("i"), el("i"), el("i"))));
  $("#chat").append(q, box);
  scrollDown();
  return box;
}

// ---------- chat history (kept in THIS browser only, survives a reload) ----------
const HIST_KEY = "chat.v1", HIST_TURNS = 40, HIST_CHARS = 300000;
const slim = (r) => ({  // only what the page draws; no context, no prompts
  answer: String(r.answer || ""), route: r.route, router: r.router && r.router.route ? { route: r.router.route } : undefined,
  escalated: !!r.escalated, cached: !!r.cached, guard: r.guard, grounded: r.grounded,
  sources: (r.sources || []).map((x) => ({ source: x.source })), latency_ms: r.latency_ms,
});
function loadHistory() {
  try { const v = JSON.parse(store.get(HIST_KEY) || "[]"); return Array.isArray(v) ? v : []; } catch { return []; }
}
function saveTurn(turn) {
  const h = loadHistory(); h.push(turn);
  while (h.length > HIST_TURNS) h.shift();
  let text = JSON.stringify(h);
  while (text.length > HIST_CHARS && h.length > 1) { h.shift(); text = JSON.stringify(h); }
  store.set(HIST_KEY, text);
}
function restoreHistory() {
  const h = loadHistory();
  if (!h.length) return;
  $("#empty")?.remove();
  for (const t of h) {
    if (!t || !t.q || !Array.isArray(t.replies)) continue;
    $("#chat").append(el("div", { class: "msg-user", text: String(t.q) }),
      el("div", { class: "replies" }, t.replies.map((x) => reply(String(x.title || "Answer"), x.r || {}, false))));
  }
  scrollDown();
}

async function ask(question) {
  const route = $("#route").value, compare = $("#compare").checked;
  const btn = $("#ask-btn");
  btn.disabled = true;
  const box = addTurn(question);
  try {
    if (compare) {
      const d = await api("/compare", postJson({ question }));
      box.replaceChildren(reply("Normal RAG (vector search)", d.vector), reply("GraphRAG (knowledge graph)", d.graph));
      saveTurn({ q: question, replies: [{ title: "Normal RAG (vector search)", r: slim(d.vector) }, { title: "GraphRAG (knowledge graph)", r: slim(d.graph) }] });
    } else {
      const d = await api("/ask", postJson({ question, route }));
      const title = route === "auto" ? "Answer" : `Answer · ${TITLES[route]}`;
      box.replaceChildren(reply(title, d));
      saveTurn({ q: question, replies: [{ title, r: slim(d) }] });
    }
  } catch (e) {
    box.replaceChildren(el("div", { class: "reply" }, el("span", { class: "err", text: e.message })));
  } finally {
    btn.disabled = false;
    scrollDown();
    $("#question").focus();
  }
}

// ---------- upload ----------
let uploading = false;
async function startUpload(files) {
  if (uploading || !files || !files.length) return;
  uploading = true;
  $("#drop").classList.add("busy");
  if (!sideOpen()) setSide(true, false);
  const seenReady = new Set();
  try {
    await Uploader.start(files, $("#do-graph").checked, $("#upload-out"), {
      onUpdate: (job) => {  // refresh the list as soon as a file becomes searchable
        const newly = job.files.some((f) => f.searchable && !seenReady.has(f.source) && seenReady.add(f.source));
        if (newly) loadDocs().then(loadChips);
      },
      onDone: async (job) => {
        await loadDocs(); loadChips();
        if (job.files.some((f) => f.stage === "skipped")) toast("Some files were already uploaded", false);
        if (job.files.some((f) => f.stage === "error")) toast("Some files failed. See the panel for details.", true);
      },
    });
  } finally {
    uploading = false;
    $("#drop").classList.remove("busy");
    $("#files").value = "";
  }
}

document.addEventListener("DOMContentLoaded", async () => {
  // sidebar state: remembered on desktop, closed on phones unless there is nothing uploaded yet
  setSide(!isMobile() && store.get("side") !== "closed", false);
  $("#side-open").addEventListener("click", () => setSide(!sideOpen()));
  $("#side-close").addEventListener("click", () => setSide(false));
  $("#scrim").addEventListener("click", () => setSide(false));
  document.addEventListener("keydown", (ev) => { if (ev.key === "Escape" && isMobile() && sideOpen()) setSide(false); });

  // drag and drop (the zone, and anywhere on the page)
  const drop = $("#drop"), veil = $("#veil");
  let depth = 0;
  const hasFiles = (ev) => ev.dataTransfer && [...ev.dataTransfer.types].includes("Files");
  window.addEventListener("dragenter", (ev) => { if (hasFiles(ev)) { depth++; veil.hidden = false; } });
  window.addEventListener("dragleave", (ev) => { if (hasFiles(ev) && --depth <= 0) { depth = 0; veil.hidden = true; drop.classList.remove("over"); } });
  window.addEventListener("dragover", (ev) => { if (hasFiles(ev)) { ev.preventDefault(); drop.classList.toggle("over", ev.target instanceof Node && drop.contains(ev.target)); } });
  window.addEventListener("drop", (ev) => {
    if (!hasFiles(ev)) return;
    ev.preventDefault(); depth = 0; veil.hidden = true; drop.classList.remove("over");
    startUpload(ev.dataTransfer.files);
  });
  $("#files").addEventListener("change", (ev) => startUpload(ev.target.files));

  $("#compare").addEventListener("change", (ev) => { $("#route").disabled = ev.target.checked; });
  $("#clear").addEventListener("click", () => {
    store.set(HIST_KEY, "[]");
    $("#chat").replaceChildren(el("div", { id: "empty", class: "empty" }, el("p", { id: "empty-text", text: "Chat cleared. Ask a new question below." }), el("div", { id: "chips", class: "chips" })));
    loadChips();
  });
  $("#ask-form").addEventListener("submit", (ev) => {
    ev.preventDefault();
    const q = $("#question").value.trim();
    if (!q) return;
    $("#question").value = "";
    ask(q);
  });
  $("#question").addEventListener("keydown", (ev) => {
    if (ev.key === "Enter" && !ev.shiftKey) { ev.preventDefault(); $("#ask-form").requestSubmit(); }
  });

  restoreHistory();
  watchWarmup();
  const n = await loadDocs();
  if (n === 0) setSide(true, false); else loadChips();
});
