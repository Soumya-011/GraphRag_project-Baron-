"use strict";
/* Shared upload widget: POST /ingest -> job id -> poll /ingest/jobs/<id> and show per-file progress.
   Used by the user page and the developer page. No innerHTML: file names are untrusted text. */
(function () {
  const STAGES = {
    queued: "Waiting in the queue", checking: "Checking", reading: "Reading the file",
    embedding: "Indexing the text", extracting: "Building the knowledge graph (about a minute per file)",
    done: "Done", skipped: "Skipped", error: "Failed",
  };
  const secs = (ms) => (ms < 10000 ? (ms / 1000).toFixed(1) : Math.round(ms / 1000)) + " s";
  const EXT = [".txt", ".md", ".pdf"];

  function h(tag, cls, text) {
    const n = document.createElement(tag);
    if (cls) n.className = cls;
    if (text != null) n.textContent = text;
    return n;
  }
  async function readError(r) {
    let d = null;
    try { d = await r.json(); } catch { /* not JSON */ }
    const detail = d && d.detail;
    return typeof detail === "string" ? detail : detail ? JSON.stringify(detail) : `HTTP ${r.status}`;
  }
  const sleep = (ms) => new Promise((res) => setTimeout(res, ms));

  function render(mount, job) {
    const list = h("ul", "up-list");
    for (const f of job.files) {
      const finished = ["done", "skipped", "error"].includes(f.stage);
      const li = h("li", `up-row ${f.stage}` + (finished ? "" : " busy") + (f.searchable && !finished ? " ready" : ""));
      const head = h("div", "up-head");
      const label = f.stage === "extracting" && f.searchable ? "Ready to ask · building the graph" : (STAGES[f.stage] || f.stage);
      head.append(h("span", "up-name", f.name), h("span", "up-stage", label));
      const bar = h("div", "up-bar"), fill = h("i");
      fill.style.width = `${f.progress || 0}%`;
      bar.append(fill);
      li.append(head, bar);
      const took = f.elapsed_ms != null ? secs(f.elapsed_ms)
        : f.started ? secs(Math.max(0, (job.server_time - f.started) * 1000)) : "";
      let note = "";
      if (f.stage === "done") {
        note = `${f.chunks} text chunks` + (f.entities != null ? ` · ${f.entities} entities · ${f.relations} relations` : "");
        if (f.llm_calls != null) note += ` · ${f.llm_calls} LLM calls`;
        if (took) note += ` · ${took}`;
        if (f.graph_error) note += ` · graph step failed: ${f.graph_error}`;
      } else if (f.stage === "skipped") note = f.detail || "Already ingested";
      else if (f.stage === "error") note = f.error || "Something went wrong";
      else if (took) note = f.stage === "queued" ? "" : `${took} so far`;
      if (note) li.append(h("div", "up-note", note));
      list.append(li);
    }
    mount.replaceChildren(list);
  }

  async function start(files, graph, mount, opts = {}) {
    const maxMB = opts.maxMB || 5;
    const bad = [...files].filter((f) => !EXT.some((e) => f.name.toLowerCase().endsWith(e)));
    if (!files.length) { mount.replaceChildren(h("p", "up-error", "Choose at least one .txt, .md or .pdf file.")); return null; }
    if (bad.length) { mount.replaceChildren(h("p", "up-error", `Not supported: ${bad.map((f) => f.name).join(", ")}. Use .txt, .md or .pdf.`)); return null; }
    const big = [...files].filter((f) => f.size > maxMB * 1024 * 1024);
    if (big.length) { mount.replaceChildren(h("p", "up-error", `Too large (max ${maxMB} MB): ${big.map((f) => f.name).join(", ")}`)); return null; }

    mount.replaceChildren(h("p", "up-note", "Uploading…"));
    const fd = new FormData();
    for (const f of files) fd.append("files", f);
    let id;
    try {
      const r = await fetch(`/ingest?graph=${graph}`, { method: "POST", body: fd });
      if (!r.ok) throw new Error(await readError(r));
      id = (await r.json()).job_id;
    } catch (e) {
      mount.replaceChildren(h("p", "up-error", e.message || "Upload failed"));
      return null;
    }

    return watch(id, mount, opts);
  }

  async function watch(id, mount, opts = {}) {
    let failures = 0, job = null;
    for (let i = 0; i < 1200; i++) {  // about 30 minutes at most
      try {
        const r = await fetch(`/ingest/jobs/${id}`);
        if (!r.ok) throw new Error(await readError(r));
        job = await r.json();
        failures = 0;
        render(mount, job);
        if (opts.onUpdate) opts.onUpdate(job);
        if (job.status === "done" || job.status === "error") break;
      } catch (e) {
        if (++failures >= 5) {
          mount.append(h("p", "up-error", `Lost contact with the server: ${e.message}`));
          return null;
        }
      }
      await sleep(1500);
    }
    if (job) {
      const done = job.files.filter((f) => f.stage === "done").length;
      const skipped = job.files.filter((f) => f.stage === "skipped").length;
      const failed = job.files.filter((f) => f.stage === "error").length;
      const parts = [];
      if (done) parts.push(`${done} added`);
      if (skipped) parts.push(`${skipped} already there`);
      if (failed) parts.push(`${failed} failed`);
      mount.append(h("p", "up-summary" + (failed ? " up-error" : ""), parts.join(" · ") || "Finished"));
      if (opts.onDone) opts.onDone(job);
    }
    return job;
  }
  window.Uploader = { start, watch };
})();
