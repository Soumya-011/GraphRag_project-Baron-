"use strict";

const $ = (sel) => document.querySelector(sel);
const SVG_NS = "http://www.w3.org/2000/svg";
const EXAMPLES = [
  "Which companies are linked to Priya Nair?",
  "How is Priya Nair connected to SunArc Power?",
  "Who is the CEO of SunArc Power?",
  "Summarize NovaTech Labs",
  "How many companies are there?",
  "What is the capital of France?",
  "hello",
];
const GUARDS = {
  cache: "Served from cache",
  rule: "Answered by a Cypher rule (no LLM)",
  gate: "Not found: LLM skipped",
  blocked: "Blocked by input guard",
  smalltalk: "Small-talk reply",
  off_topic: "Off-topic filter",
};
const TITLES = { auto: "Auto", vector: "Normal RAG (vector)", graph: "GraphRAG", hybrid: "Hybrid" };

let lastGraphEdges = [];   // edges the GraphRAG answer used (highlighted in the graph)
let graphData = [];        // edges loaded from /graph/summary

// ---------- helpers (no innerHTML anywhere: answers are untrusted text) ----------
function el(tag, attrs = {}, ...kids) {
  const n = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs)) {
    if (k === "class") n.className = v;
    else if (k === "text") n.textContent = v;
    else if (k.startsWith("on")) n.addEventListener(k.slice(2), v);
    else if (v !== false && v != null) n.setAttribute(k, v);
  }
  for (const kid of kids.flat()) {
    if (kid != null) n.append(kid.nodeType ? kid : document.createTextNode(String(kid)));
  }
  return n;
}
function svg(tag, attrs = {}) {
  const n = document.createElementNS(SVG_NS, tag);
  for (const [k, v] of Object.entries(attrs)) n.setAttribute(k, v);
  return n;
}
async function api(path, opts) {
  const r = await fetch(path, opts);
  let data = null;
  try { data = await r.json(); } catch { /* not JSON */ }
  if (!r.ok) {
    const d = data && data.detail;
    throw new Error(typeof d === "string" ? d : d ? JSON.stringify(d) : `HTTP ${r.status}`);
  }
  return data;
}
const postJson = (body) => ({
  method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body),
});
const edgeKey = (e) => `${e.source}|${e.rel}|${e.target}`;

// ---------- health ----------
async function loadHealth() {
  const box = $("#health");
  box.replaceChildren(el("span", { class: "pill", text: "checking…" }));
  try {
    const h = await api("/dev/health");
    const pills = [];
    for (const [name, c] of Object.entries(h.checks)) {
      if (name === "llm" && c.roles) {
        for (const [role, r] of Object.entries(c.roles)) {
          pills.push(el("span", {
            class: "pill " + (r.ok ? "ok" : "bad"),
            title: r.error || r.model,
          }, `${role}: ${r.model}`));
        }
        const f = c.fallback;
        if (f) {
          pills.push(el("span", {
            class: "pill" + (f.enabled ? " ok" : ""),
            title: f.enabled ? `${f.model} is used when the primary model fails` : "Fallback off",
          }, `fallback: ${f.enabled ? f.model : "off"}`));
        }
      } else {
        pills.push(el("span", { class: "pill " + (c.ok ? "ok" : "bad"), title: c.error || "" }, name));
      }
    }
    box.replaceChildren(...pills);
  } catch (e) {
    box.replaceChildren(el("span", { class: "pill bad", text: "API unreachable" }));
  }
}

// ---------- answer cards ----------
function inspectInfo(r) {
  const g = r.graph || {};
  return {
    route: r.route, router: r.router || null, guard: r.guard || null, cached: r.cached || false,
    escalated: !!r.escalated, llm_calls: r.llm_calls, latency_ms: r.latency_ms, rate_limiter_wait_ms: r.wait_ms ?? 0,
    net_latency_ms: r.latency_ms != null ? r.latency_ms - (r.wait_ms || 0) : null,
    best_score: r.best_score ?? null, grounding: r.grounding || null, fallback_events: r.fallback || null,
    graph_facts: (g.edges || []).length, graph_paths: (g.paths || []).length,
    context_chars: (r.context || "").length,
    sources: (r.sources || []).map((s) => (s.chunk != null ? `${s.source}#${s.chunk}` : s.source) + (s.score != null ? ` (${s.score})` : "")),
  };
}

function card(title, r) {
  const badges = el("div", { class: "badges" }, el("span", { class: "badge", text: r.route }));
  if (r.router) {
    badges.append(el("span", { class: "badge cache", title: r.router.reason,
      text: `auto → ${r.router.route} (${r.router.via})` }));
  }
  if (r.guard) {
    badges.append(el("span", { class: "badge guard", text: GUARDS[r.guard] || r.guard }));
  }
  if (r.escalated) {
    badges.append(el("span", { class: "badge guard", title: "The graph had no answer, so documents were added",
      text: "escalated: graph → hybrid" }));
  }
  if (r.fallback) {
    badges.append(el("span", { class: "badge guard", title: r.fallback.join("\n"), text: "Fallback model used" }));
  }
  if (r.cached) badges.append(el("span", { class: "badge cache", text: `cache: ${r.cached}` }));

  const meta = el("div", { class: "meta" },
    el("span", { text: `${r.latency_ms} ms` }),
    el("span", { text: `${r.llm_calls} LLM call${r.llm_calls === 1 ? "" : "s"}` }));
  if (r.best_score != null) meta.append(el("span", { text: `best score ${r.best_score}` }));
  if (r.graph && r.graph.edges) meta.append(el("span", { text: `${r.graph.edges.length} graph facts` }));

  const kids = [el("h3", { text: title }), badges, el("div", { class: "answer", text: r.answer }), meta];

  if (r.router) {
    kids.push(el("div", { class: "muted small", text: `Router: ${r.router.reason}` }));
  }
  if (r.grounding && !r.grounding.ok) {
    kids.push(el("div", { class: "warn" },
      "Not found in the retrieved context (possible hallucination): ",
      r.grounding.unsupported.join(", ")));
  }
  if (r.sources && r.sources.length) {
    const seen = new Set();
    const chips = [];
    for (const s of r.sources) {
      const label = s.chunk != null ? `${s.source} #${s.chunk}` : s.source;
      if (!seen.has(label)) { seen.add(label); chips.push(el("span", { text: label })); }
    }
    kids.push(el("div", { class: "src" }, chips));
  }
  if (r.context) {
    kids.push(el("details", {}, el("summary", { text: "Retrieved context" }), el("pre", { text: r.context })));
  }
  kids.push(el("details", {}, el("summary", { text: "Inspect (timings, router, sources)" }),
    el("pre", { text: JSON.stringify(inspectInfo(r), null, 2) })));
  return el("article", { class: "card" }, kids);
}

function renderResults(nodes) {
  $("#results").replaceChildren(...nodes);
}

async function ask(question) {
  const compare = $("#compare").checked;
  const route = $("#route").value;
  const btn = $("#ask-btn");
  btn.disabled = true;
  renderResults([el("p", { class: "muted empty", text: "Thinking…" })]);
  try {
    if (compare) {
      const d = await api("/dev/compare", postJson({ question }));
      lastGraphEdges = (d.graph.graph && d.graph.graph.edges) || [];
      renderResults([card(TITLES.vector, d.vector), card(TITLES.graph, d.graph)]);
    } else {
      const d = await api("/dev/ask", postJson({ question, route }));
      lastGraphEdges = (d.graph && d.graph.edges) || [];
      renderResults([card(route === "auto" ? `Auto → ${TITLES[d.route] || d.route}` : TITLES[route], d)]);
    }
  } catch (e) {
    lastGraphEdges = [];
    renderResults([el("p", { class: "err", text: e.message })]);
  } finally {
    btn.disabled = false;
    drawGraph();
    loadMetrics();
  }
}

// ---------- graph (static force layout, plain SVG) ----------
function layout(nodes, links, w, h) {
  nodes.forEach((n, i) => {
    const a = (2 * Math.PI * i) / nodes.length;
    n.x = w / 2 + Math.cos(a) * w / 3; n.y = h / 2 + Math.sin(a) * h / 3; n.vx = 0; n.vy = 0;
  });
  const idx = new Map(nodes.map((n, i) => [n.id, i]));
  for (let t = 0; t < 300; t++) {
    const cool = 1 - t / 300;
    for (let i = 0; i < nodes.length; i++) {
      for (let j = i + 1; j < nodes.length; j++) {
        let dx = nodes[i].x - nodes[j].x, dy = nodes[i].y - nodes[j].y;
        const d2 = dx * dx + dy * dy || 0.01, d = Math.sqrt(d2), f = 2600 / d2;
        dx /= d; dy /= d;
        nodes[i].vx += dx * f; nodes[i].vy += dy * f; nodes[j].vx -= dx * f; nodes[j].vy -= dy * f;
      }
    }
    for (const l of links) {
      const a = nodes[idx.get(l.s)], b = nodes[idx.get(l.t)];
      let dx = b.x - a.x, dy = b.y - a.y;
      const d = Math.sqrt(dx * dx + dy * dy) || 0.01, f = (d - 95) * 0.03;
      dx /= d; dy /= d;
      a.vx += dx * f; a.vy += dy * f; b.vx -= dx * f; b.vy -= dy * f;
    }
    for (const n of nodes) {
      n.vx += (w / 2 - n.x) * 0.006; n.vy += (h / 2 - n.y) * 0.006;
      n.x = Math.min(w - 30, Math.max(30, n.x + Math.max(-20, Math.min(20, n.vx * cool))));
      n.y = Math.min(h - 25, Math.max(25, n.y + Math.max(-20, Math.min(20, n.vy * cool))));
      n.vx *= 0.6; n.vy *= 0.6;
    }
  }
  // stretch the result to fill the canvas
  const xs = nodes.map((n) => n.x), ys = nodes.map((n) => n.y);
  const [x0, x1, y0, y1] = [Math.min(...xs), Math.max(...xs), Math.min(...ys), Math.max(...ys)];
  for (const n of nodes) {
    n.x = 50 + ((n.x - x0) / (x1 - x0 || 1)) * (w - 100);
    n.y = 30 + ((n.y - y0) / (y1 - y0 || 1)) * (h - 70);
  }
}

function drawGraph() {
  const root = $("#graph");
  root.replaceChildren();
  root.classList.toggle("dim", lastGraphEdges.length > 0);
  if (!graphData.length) {
    const t = svg("text", { x: 320, y: 210, class: "node-label" });
    t.textContent = "Graph is empty. Run python -m app.seed or ingest documents.";
    root.append(t);
    return;
  }
  const used = new Set(lastGraphEdges.map(edgeKey));
  const byKey = new Map();
  for (const e of graphData) {
    const k = edgeKey(e);
    if (!byKey.has(k)) byKey.set(k, { ...e, docs: [] });
    if (e.doc) byKey.get(k).docs.push(e.doc);
  }
  const edges = [...byKey.values()];
  const nodeMap = new Map();
  for (const e of edges) {
    nodeMap.set(e.source, { id: e.source, label: e.source_label || "Company" });
    nodeMap.set(e.target, { id: e.target, label: e.target_label || "Company" });
  }
  const nodes = [...nodeMap.values()];
  const links = edges.map((e) => ({ s: e.source, t: e.target }));
  layout(nodes, links, 640, 420);
  const pos = new Map(nodes.map((n) => [n.id, n]));

  const defs = svg("defs");
  const marker = svg("marker", { id: "arrow", viewBox: "0 0 10 10", refX: 17, refY: 5,
    markerWidth: 9, markerHeight: 9, markerUnits: "userSpaceOnUse", orient: "auto-start-reverse" });
  marker.append(svg("path", { d: "M0,0 L10,5 L0,10 z", fill: "currentColor" }));
  defs.append(marker);
  root.append(defs);

  const usedNodes = new Set();
  const labels = [];
  for (const e of edges) {
    const a = pos.get(e.source), b = pos.get(e.target);
    const isUsed = used.has(edgeKey(e));
    if (isUsed) { usedNodes.add(e.source); usedNodes.add(e.target); }
    const line = svg("line", { x1: a.x, y1: a.y, x2: b.x, y2: b.y, "marker-end": "url(#arrow)",
      class: "edge" + (isUsed ? " used" : "") + (e.status === "past" ? " past" : "") });
    line.style.color = isUsed ? "var(--accent)" : "var(--muted)";
    const tip = svg("title");
    tip.textContent = `${e.source} ${e.rel} ${e.target}` +
      (e.status === "past" ? " (past)" : "") + (e.docs.length ? ` [${e.docs.join(", ")}]` : "");
    line.append(tip);
    root.append(line);
    if (isUsed) {
      const t = svg("text", { x: (a.x + b.x) / 2, y: (a.y + b.y) / 2 - 3, class: "edge-label" });
      t.textContent = e.rel;
      labels.push(t);
    }
  }
  for (const t of labels) root.append(t);
  for (const n of nodes) {
    const g = svg("g", { class: "node" + (usedNodes.has(n.id) ? " used" : "") });
    g.append(svg("circle", { cx: n.x, cy: n.y, r: 9, class: "n-" + n.label.toLowerCase() }));
    const t = svg("text", { x: n.x, y: n.y + 21, class: "node-label" });
    t.textContent = n.id.length > 20 ? n.id.slice(0, 19) + "…" : n.id;
    const tip = svg("title");
    tip.textContent = `${n.id} (${n.label})`;
    g.append(t, tip);
    root.append(g);
  }
  $("#graph-note").textContent =
    `${nodes.length} entities, ${edges.length} relations` +
    (lastGraphEdges.length ? `; ${used.size} highlighted from the last GraphRAG answer` : "");
}

async function loadGraph() {
  try {
    const d = await api("/graph/summary?limit=500");
    graphData = d.edges || [];
  } catch (e) {
    graphData = [];
    $("#graph-note").textContent = `Graph unavailable: ${e.message}`;
  }
  drawGraph();
}

// ---------- metrics ----------
async function loadMetrics() {
  const box = $("#metrics");
  try {
    const m = await api("/metrics");
    const stat = (v, label) => el("div", { class: "stat" }, el("b", { text: String(v) }), el("span", { text: label }));
    box.replaceChildren(
      stat(m.requests, "requests"),
      stat(m.llm_calls, "LLM calls used"),
      stat(`${m.llm_calls_saved} (${m.saved_pct}%)`, "LLM calls saved"),
      el("div", { class: "breakdown" },
        Object.entries(m.saved_by).map(([k, v]) => `${k.replace("_", " ")}: ${v}`).join(" · ") +
        (m.grounding_flags ? ` · grounding flags: ${m.grounding_flags}` : "") +
        (m.fallback_used ? ` · fallback model used: ${m.fallback_used}` : "")));
  } catch (e) {
    box.replaceChildren(el("p", { class: "muted small", text: "Metrics unavailable." }));
  }
}

// ---------- upload (shared widget, jobs + progress) ----------
async function upload(ev) {
  ev.preventDefault();
  const btn = $("#upload-btn");
  btn.disabled = true;
  try {
    await Uploader.start($("#files").files, $("#do-graph").checked, $("#upload-out"), {
      maxMB: 50, onDone: () => { if ($("#graph-fold").open) loadGraph(); loadHealth(); if ($("#docs-fold").open) loadDocs(); },
    });
  } finally { btn.disabled = false; }
}

// ---------- documents manager ----------
async function loadDocs() {
  const box = $("#docs");
  try {
    const d = await api("/dev/docs");
    if (!d.documents.length) { box.replaceChildren(el("p", { class: "muted small", text: "No documents registered yet. Run python -m app.ingest or upload files." })); return; }
    const head = el("tr", {}, ["Source", "Chunks", "Entities", "Relations", "Graph", "Ingested", ""].map((h) => el("th", { text: h })));
    const rows = d.documents.map((x) => el("tr", {},
      el("td", { class: "src-cell", text: x.source }),
      el("td", { text: x.chunks }), el("td", { text: x.entities }), el("td", { text: x.relations }),
      el("td", { text: x.graph ? "yes" : "no", class: x.graph ? "" : "err" }),
      el("td", { class: "small", text: new Date(x.ingested_at * 1000).toLocaleString() }),
      el("td", { class: "actions" },
        el("button", { type: "button", class: "ghost", title: "Embed and extract again", onclick: () => reingest(x.source) }, "Re-ingest"),
        el("button", { type: "button", class: "ghost danger", onclick: () => removeDoc(x) }, "Delete"))));
    box.replaceChildren(el("table", { class: "docs" }, el("thead", {}, head), el("tbody", {}, rows)),
      el("p", { class: "muted small", text: `${d.documents.length} documents, ${d.documents.reduce((a, x) => a + x.chunks, 0)} chunks.` }));
  } catch (e) {
    box.replaceChildren(el("p", { class: "err small", text: e.message }));
  }
}
async function reingest(source) {
  const out = $("#upload-out");
  try {
    const d = await api(`/dev/docs/reingest?source=${encodeURIComponent(source)}`, { method: "POST" });
    await Uploader.watch(d.job_id, out, { onDone: () => { if ($("#graph-fold").open) loadGraph(); loadDocs(); } });
  } catch (e) { out.replaceChildren(el("span", { class: "err", text: e.message })); }
}
async function removeDoc(x) {
  const extra = x.upload ? "The uploaded file is removed too." : "The file stays in data/raw and returns on the next `python -m app.ingest`.";
  if (!confirm(`Delete ${x.source} from the vector store and the graph?\n${extra}`)) return;
  try {
    await api(`/dev/docs?source=${encodeURIComponent(x.source)}`, { method: "DELETE" });
  } catch (e) { alert(e.message); }
  loadDocs(); if ($("#graph-fold").open) loadGraph(); loadMetrics();
}

// ---------- settings ----------
async function loadSettings() {
  const box = $("#settings");
  try {
    const d = await api("/dev/settings");
    const rows = Object.entries(d).sort(([a], [b]) => a.localeCompare(b)).map(([k, v]) =>
      el("tr", {}, el("td", { text: k }), el("td", { text: v === "" ? "(default)" : String(v) })));
    box.replaceChildren(el("table", { class: "docs" }, el("tbody", {}, rows)));
  } catch (e) { box.replaceChildren(el("p", { class: "err small", text: e.message })); }
}

async function logout() {
  await fetch("/auth/logout", { method: "POST" });
  location.href = "/login";
}

// ---------- init ----------
// ---------- collapsible panels (remembered, content loaded on first open) ----------
const store = {
  get(k) { try { return localStorage.getItem(k); } catch { return null; } },
  set(k, v) { try { localStorage.setItem(k, v); } catch { /* ignore */ } },
};
function fold(id, onFirstOpen) {
  const d = $(id);
  let loaded = false;
  const open = () => { if (!loaded) { loaded = true; onFirstOpen && onFirstOpen(); } };
  if (store.get("dev-fold" + id) === "open") { d.open = true; open(); }
  d.addEventListener("toggle", () => { store.set("dev-fold" + id, d.open ? "open" : "closed"); if (d.open) open(); });
}

// ---------- tiny markdown renderer (headings, tables, bullets, bold, code). No innerHTML ----------
function inline(text) {
  const out = [];
  for (const part of text.split(/(\*\*[^*]+\*\*|`[^`]+`)/)) {
    if (part.startsWith("**")) out.push(el("b", { text: part.slice(2, -2) }));
    else if (part.startsWith("`")) out.push(el("code", { text: part.slice(1, -1) }));
    else if (part) out.push(part);
  }
  return out;
}
function renderMarkdown(text) {
  const nodes = [], lines = text.split("\n");
  for (let i = 0; i < lines.length; i++) {
    const l = lines[i];
    if (/^#{1,6} /.test(l)) nodes.push(el(l.startsWith("## ") ? "h3" : l.startsWith("### ") ? "h4" : "h3", {}, inline(l.replace(/^#+ /, ""))));
    else if (l.startsWith("|")) {
      const rows = [];
      while (i < lines.length && lines[i].startsWith("|")) { if (!/^\|[-| ]+\|$/.test(lines[i])) rows.push(lines[i]); i++; }
      i--;
      const cells = (r) => r.split("|").slice(1, -1).map((c) => c.trim());
      nodes.push(el("table", {}, rows.map((r, k) => el("tr", {}, cells(r).map((c) => el(k === 0 ? "th" : "td", {}, inline(c)))))));
    } else if (l.startsWith("- ")) {
      const items = [];
      while (i < lines.length && lines[i].startsWith("- ")) { items.push(el("li", {}, inline(lines[i].slice(2)))); i++; }
      i--;
      nodes.push(el("ul", {}, items));
    } else if (l.trim()) nodes.push(el("p", {}, inline(l)));
  }
  return nodes;
}

// ---------- evaluation reports ----------
async function loadEvalList() {
  const a = $("#eval-a"), b = $("#eval-b");
  try {
    const d = await api("/dev/eval/reports");
    const label = (r) => `${r.name} (${new Date(r.modified * 1000).toLocaleString()})`;
    a.replaceChildren(...d.reports.map((r) => el("option", { value: r.name, text: label(r) })));
    b.replaceChildren(el("option", { value: "", text: "(none)" }), ...d.reports.map((r) => el("option", { value: r.name, text: label(r) })));
    $("#eval-hint").textContent = d.reports.length ? `${d.reports.length} runs in eval/results` : "No reports yet. Run python -m eval.run_eval";
    if (d.reports.length) showEval();
  } catch (e) { $("#eval-hint").textContent = e.message; }
}
const fmt = (v, suf = "") => (v == null ? "n/a" : `${v}${suf}`);
function delta(a, b, goodWhenHigher = true) {
  if (a == null || b == null || a === b) return el("span", { class: "muted", text: " " });
  const up = b > a;
  return el("span", { class: up === goodWhenHigher ? "delta-up" : "delta-down", text: ` ${up ? "+" : ""}${b - a}` });
}
async function showEval() {
  const out = $("#eval-out"), na = $("#eval-a").value, nb = $("#eval-b").value;
  if (!na) return;
  try {
    const A = (await api(`/dev/eval/report?name=${encodeURIComponent(na)}&summary=true`)).summary;
    const B = nb ? (await api(`/dev/eval/report?name=${encodeURIComponent(nb)}&summary=true`)).summary : null;
    const metrics = [["Correct", "correct_pct", "%", true], ["Recall", "recall_pct", "%", true], ["Precision", "precision_pct", "%", true],
      ["LLM calls", "llm_calls", "", false], ["Avg ms", "avg_ms", "", false], ["p95 ms", "p95_ms", "", false]];
    const modes = [...new Set([...A.modes, ...(B ? B.modes : [])].map((m) => m.mode))];
    const rows = modes.map((mode) => {
      const x = A.modes.find((m) => m.mode === mode), y = B && B.modes.find((m) => m.mode === mode);
      return el("tr", {}, el("td", { text: mode }), metrics.map(([, k, suf, hi]) =>
        el("td", {}, x ? fmt(x[k], suf) : "n/a", B ? [" → ", y ? fmt(y[k], suf) : "n/a", x && y ? delta(x[k], y[k], hi) : null] : null)));
    });
    out.replaceChildren(
      el("p", { class: "muted small", text: B ? `${na}  →  ${nb} (second run shown after the arrow)` : A.settings || A.title }),
      el("table", { class: "docs" }, el("thead", {}, el("tr", {}, el("th", { text: "Mode" }), metrics.map(([n]) => el("th", { text: n })))), el("tbody", {}, rows)));
    $("#eval-md").replaceChildren();
  } catch (e) { out.replaceChildren(el("p", { class: "err small", text: e.message })); }
}
async function showEvalFull() {
  const name = $("#eval-a").value;
  if (!name) return;
  try {
    const d = await api(`/dev/eval/report?name=${encodeURIComponent(name)}`);
    $("#eval-md").replaceChildren(...renderMarkdown(d.text));
  } catch (e) { $("#eval-md").replaceChildren(el("p", { class: "err small", text: e.message })); }
}

// ---------- extraction quality ----------
async function runExtraction() {
  const out = $("#extract-out"), btn = $("#extract-run");
  btn.disabled = true;
  out.replaceChildren(el("p", { class: "muted small", text: "Checking…" }));
  try {
    const r = await api(`/dev/extraction?corpus=${$("#extract-corpus").checked}`);
    if (r.empty) { out.replaceChildren(el("p", { class: "warn", text: `The graph has no gold-labelled documents (${r.excluded} uploaded edges ignored). Run python -m app.ingest.` })); return; }
    const stat = (v, l) => el("div", { class: "stat" }, el("b", { text: v }), el("span", { text: l }));
    const list = (title, rows) => rows.length ? el("details", {}, el("summary", { text: `${title} (${rows.length})` }),
      el("div", { class: "triples" }, rows.slice(0, 80).map(([s, rel, t, st]) => el("div", { text: `${s} -${rel}-> ${t} (${st})` })))) : null;
    out.replaceChildren(
      el("div", { class: "kv" }, stat(`${Math.round(r.precision * 100)}%`, `precision (${r.good}/${r.extracted})`),
        stat(`${Math.round(r.recall * 100)}%`, `recall (${r.hit}/${r.gold})`), stat(String(r.excluded), "upload edges ignored"),
        r.dated ? stat(`${r.dated.pct == null ? "n/a" : Math.round(r.dated.pct * 100) + "%"}`, `year in quote (${r.dated.year_in_quote}/${r.dated.count} dated facts, not scored)`) : null),
      list("Extracted but not in gold", r.wrong), list("Missing from the graph", r.missing));
  } catch (e) { out.replaceChildren(el("p", { class: "err small", text: e.message })); }
  finally { btn.disabled = false; }
}

// ---------- router playground ----------
async function checkRoute(ev) {
  ev.preventDefault();
  const q = $("#router-q").value.trim(), out = $("#router-out"), btn = $("#router-go");
  if (!q) return;
  btn.disabled = true;
  try {
    const d = await api(`/route?q=${encodeURIComponent(q)}&llm=${$("#router-llm").checked}`);
    const row = (k, v) => el("tr", {}, el("td", { text: k }), el("td", { text: v }));
    out.replaceChildren(el("table", { class: "docs" }, el("tbody", {},
      row("Route", d.route), row("Decided by", d.via || "n/a"), row("Reason", d.reason || "n/a"),
      row("Linked entities", d.entities.length ? d.entities.join(", ") : "none"), row("Router LLM calls", String(d.llm_calls ?? 0)))));
  } catch (e) { out.replaceChildren(el("p", { class: "err small", text: e.message })); }
  finally { btn.disabled = false; }
}

document.addEventListener("DOMContentLoaded", () => {
  $("#examples").replaceChildren(...EXAMPLES.map((q) =>
    el("button", { type: "button", class: "chip", onclick: () => { $("#question").value = q; $("#question").focus(); } }, q)));
  $("#ask-form").addEventListener("submit", (ev) => {
    ev.preventDefault();
    const q = $("#question").value.trim();
    if (q) ask(q);
  });
  $("#question").addEventListener("keydown", (ev) => {
    if (ev.key === "Enter" && !ev.shiftKey) { ev.preventDefault(); $("#ask-form").requestSubmit(); }
  });
  $("#upload-form").addEventListener("submit", upload);
  $("#refresh-health").addEventListener("click", loadHealth);
  $("#reload-graph").addEventListener("click", loadGraph);
  $("#logout").addEventListener("click", logout);
  $("#reload-docs").addEventListener("click", loadDocs);
  $("#eval-a").addEventListener("change", showEval);
  $("#eval-b").addEventListener("change", showEval);
  $("#eval-full").addEventListener("click", showEvalFull);
  $("#extract-run").addEventListener("click", runExtraction);
  $("#router-form").addEventListener("submit", checkRoute);
  api("/auth/me").then((m) => { if (!m.protected) $("#logout").hidden = true; }).catch(() => {});

  fold("#graph-fold", loadGraph);       // the graph query and drawing wait until the panel is opened
  fold("#docs-fold", loadDocs);
  fold("#eval-fold", loadEvalList);
  fold("#extract-fold");
  fold("#router-fold");
  fold("#settings-wrap", loadSettings);
  loadHealth(); loadMetrics();
});
