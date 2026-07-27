"use strict";

const startPanel = document.getElementById("start-panel");
const wizard = document.getElementById("wizard");
const sessionListEl = document.getElementById("session-list");

let state = null; // last-known server state for the active session

async function api(path, opts) {
  const res = await fetch(path, opts);
  let body;
  try {
    body = await res.json();
  } catch {
    body = null;
  }
  if (!res.ok) {
    const err = new Error((body && body.error) || `request failed (${res.status})`);
    err.log = body && body.log;
    throw err;
  }
  return body;
}

function errorHtml(err) {
  const log = err.log
    ? `<details class="log" open><summary>full log</summary><pre>${escapeHtml(err.log)}</pre></details>`
    : "";
  return `<p class="error">${escapeHtml(err.message)}</p>${log}`;
}

function fmtPct(x) {
  return (x * 100).toFixed(1) + "%";
}

function spinnerHtml(text) {
  return `<p class="loading"><span class="spinner"></span>${text}</p>`;
}

function setBusy(form, busy) {
  form.querySelectorAll("button").forEach((b) => (b.disabled = busy));
}

// Only one step can run at a time: while any request is in flight, every
// step's controls are disabled and the session list is locked, so it's
// never possible to fire a second request that races the first.
const newSessionBtn = document.getElementById("new-session-btn");
const STEP_EL = {
  gen: document.getElementById("step-gen"),
  levels: document.getElementById("step-levels"),
  build: document.getElementById("step-build"),
};
let uiLocked = false;

function setSidebarBusy(busy) {
  newSessionBtn.disabled = busy;
  sessionListEl.classList.toggle("disabled", busy);
}

function lockUI(activeKey) {
  uiLocked = true;
  setSidebarBusy(true);
  for (const [key, el] of Object.entries(STEP_EL)) {
    el.querySelectorAll("input, select, textarea, button").forEach((i) => (i.disabled = true));
    el.classList.toggle("working", key === activeKey);
    el.classList.toggle("locked", key !== activeKey);
  }
}

function unlockUI() {
  uiLocked = false;
  setSidebarBusy(false);
  for (const el of Object.values(STEP_EL)) {
    el.querySelectorAll("input, select, textarea, button").forEach((i) => (i.disabled = false));
    el.classList.remove("working", "locked");
  }
}

// ---------- sidebar ----------

async function loadSessionList() {
  const sessions = await api("/api/sessions");
  sessionListEl.innerHTML = "";
  for (const s of sessions) {
    const el = document.createElement("div");
    el.className = "session-item" + (state && state.id === s.id ? " active" : "");
    el.innerHTML = `
      ${s.thumb ? `<img src="${s.thumb}">` : `<div class="stl-icon" style="width:36px;height:36px;border-radius:6px;background:var(--border)"></div>`}
      <div class="meta">
        <div class="prompt">${escapeHtml(s.prompt || "(no prompt)")}</div>
        <div class="created">${s.created || ""}</div>
      </div>`;
    el.addEventListener("click", () => selectSession(s.id));
    sessionListEl.appendChild(el);
  }
}

function escapeHtml(s) {
  const d = document.createElement("div");
  d.textContent = s;
  return d.innerHTML;
}

async function selectSession(id) {
  if (uiLocked) return;
  state = await api(`/api/sessions/${id}`);
  history.replaceState(null, "", `?session=${id}`);
  render();
  loadSessionList();
}

// ---------- start panel ----------

document.getElementById("new-session-btn").addEventListener("click", () => {
  if (uiLocked) return;
  state = null;
  history.replaceState(null, "", location.pathname);
  startPanel.hidden = false;
  wizard.hidden = true;
  document.getElementById("start-form").reset();
  loadSessionList();
});

document.getElementById("start-form").addEventListener("submit", async (e) => {
  e.preventDefault();
  const form = e.target;
  const errEl = document.getElementById("start-error");
  errEl.hidden = true;
  setBusy(form, true);
  setSidebarBusy(true);
  try {
    const fd = new FormData(form);
    const res = await fetch("/api/sessions", { method: "POST", body: fd });
    const body = await res.json();
    if (!res.ok) {
      const err = new Error(body.error || "failed to create session");
      err.log = body.log;
      throw err;
    }
    state = body;
    history.replaceState(null, "", `?session=${state.id}`);
    render();
    loadSessionList();
  } catch (err) {
    errEl.innerHTML = errorHtml(err);
    errEl.hidden = false;
  } finally {
    setBusy(form, false);
    setSidebarBusy(false);
  }
});

// ---------- gen step ----------

const genForm = document.getElementById("gen-form");
genForm.addEventListener("submit", async (e) => {
  e.preventDefault();
  const fd = new FormData(genForm);
  const body = {
    prompt: fd.get("prompt"),
    n_colors: Number(fd.get("n_colors")),
    aspect: fd.get("aspect") || null,
    ref_source: fd.get("ref_source") || null,
  };
  lockUI("gen");
  document.getElementById("gen-result").innerHTML = spinnerHtml("Asking Gemini&hellip; this can take up to a minute.");
  try {
    state = await api(`/api/sessions/${state.id}/gen`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
    render();
  } catch (err) {
    document.getElementById("gen-result").innerHTML = errorHtml(err);
  } finally {
    unlockUI();
  }
});

// ---------- levels step ----------

const levelsForm = document.getElementById("levels-form");
levelsForm.addEventListener("submit", async (e) => {
  e.preventDefault();
  const fd = new FormData(levelsForm);
  const body = {
    levels: Number(fd.get("levels")),
    order: fd.get("order"),
    depth_order: fd.get("depth_order") === "on",
    width_mm: Number(fd.get("width_mm")),
    min_feature: Number(fd.get("min_feature")),
  };
  lockUI("levels");
  document.getElementById("levels-result").innerHTML = spinnerHtml("Quantizing&hellip;");
  try {
    state = await api(`/api/sessions/${state.id}/levels`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
    render();
  } catch (err) {
    document.getElementById("levels-result").innerHTML = errorHtml(err);
  } finally {
    unlockUI();
  }
});

// ---------- build step ----------

const buildForm = document.getElementById("build-form");
buildForm.addEventListener("submit", async (e) => {
  e.preventDefault();
  const fd = new FormData(buildForm);
  const levelsRun = activeRun("levels_runs", state.active_levels);
  const body = Object.assign({}, levelsRun ? levelsRun.params : {}, {
    base_mm: Number(fd.get("base_mm")),
    step_mm: Number(fd.get("step_mm")),
    relief_mm: Number(fd.get("relief_mm")),
    layer_mm: Number(fd.get("layer_mm")),
  });
  lockUI("build");
  document.getElementById("build-result").innerHTML = spinnerHtml("Meshing&hellip;");
  try {
    state = await api(`/api/sessions/${state.id}/build`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
    render();
  } catch (err) {
    document.getElementById("build-result").innerHTML = errorHtml(err);
  } finally {
    unlockUI();
  }
});

// ---------- selecting / unwinding to a past run ----------

async function selectRun(step, runId) {
  if (uiLocked) return;
  lockUI(step);
  try {
    state = await api(`/api/sessions/${state.id}/select`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ step, run_id: runId }),
    });
    render();
  } finally {
    unlockUI();
  }
}

function activeRun(key, id) {
  return (state[key] || []).find((r) => r.id === id) || null;
}

// ---------- deleting a generated image ----------

async function deleteGenRun(runId) {
  if (uiLocked) return;
  const usedBy = state.levels_runs.some((r) => r.gen_id === runId) || state.build_runs.some((r) => r.gen_id === runId);
  const warning = usedBy ? " This also deletes the levels/build results made from it." : "";
  if (!window.confirm(`Delete generated image #${runId}?${warning}`)) return;
  lockUI("gen");
  try {
    state = await api(`/api/sessions/${state.id}/gen/${runId}`, { method: "DELETE" });
    render();
  } catch (err) {
    window.alert(err.message);
  } finally {
    unlockUI();
  }
}

// ---------- rendering ----------

function render() {
  startPanel.hidden = true;
  wizard.hidden = false;

  fillGenForm();
  renderGenResult();
  renderHistory("gen-history", state.gen_runs, state.active_gen, "gen", (r) => r.url, deleteGenRun);

  const genActive = !!state.active_gen;
  document.getElementById("step-levels").classList.toggle("stale", !genActive);
  fillLevelsForm();
  renderLevelsResult();
  renderHistory("levels-history", state.levels_runs, state.active_levels, "levels", (r) => r.url);

  const levelsActive = !!state.active_levels;
  document.getElementById("step-build").classList.toggle("stale", !levelsActive);
  fillBuildForm();
  renderBuildResult();
  renderHistory("build-history", state.build_runs, state.active_build, "build", (r) => r.relief_preview_url);
}

function fillGenForm() {
  const f = genForm;
  f.prompt.value = state.prompt || "";
  f.n_colors.value = state.n_colors || 6;
  f.aspect.value = state.aspect || "";

  const options = [`<option value="">No reference image</option>`];
  if (state.has_ref) options.push(`<option value="input">Original uploaded photo</option>`);
  if (state.active_gen) options.push(`<option value="active">Currently selected image (#${state.active_gen})</option>`);
  f.ref_source.innerHTML = options.join("");
  // Default to refining from whatever's currently selected, since that's
  // the common case; falls back to the original photo on the very first run.
  f.ref_source.value = state.active_gen ? "active" : (state.has_ref ? "input" : "");
}

function refSourceLabel(refSource) {
  if (refSource == null) return null;
  if (refSource === "input") return "the original uploaded photo";
  return `generated image #${refSource}`;
}

function renderGenResult() {
  const el = document.getElementById("gen-result");
  const run = activeRun("gen_runs", state.active_gen);
  if (!run) {
    el.innerHTML = `<p class="placeholder">No image generated yet.</p>`;
    return;
  }
  const refLabel = refSourceLabel(run.ref_source);
  el.innerHTML = `
    <img src="${run.url}" alt="generated source image">
    ${refLabel ? `<p class="report-line">refined from ${refLabel}</p>` : ""}
    <details class="log"><summary>full log</summary><pre>${escapeHtml(run.log || "")}</pre></details>
  `;
}

const LEVEL_DEFAULTS = { levels: 6, order: "dark-low", depth_order: true, width_mm: 100, min_feature: 0.5 };

function fillLevelsForm() {
  const f = levelsForm;
  const last = state.levels_runs.length ? state.levels_runs[state.levels_runs.length - 1].params : null;
  const v = Object.assign({}, LEVEL_DEFAULTS, { levels: state.n_colors }, last || {});
  f.levels.value = v.levels;
  f.order.value = v.order;
  f.depth_order.checked = !!v.depth_order;
  f.width_mm.value = v.width_mm;
  f.min_feature.value = v.min_feature;
}

function renderLevelsResult() {
  const el = document.getElementById("levels-result");
  const run = activeRun("levels_runs", state.active_levels);
  if (!run) {
    el.innerHTML = `<p class="placeholder">Not run yet.</p>`;
    return;
  }
  const rep = run.report;
  const rows = rep.levels
    .slice()
    .reverse()
    .map(
      (l) => `<tr>
        <td>L${l.level}</td>
        <td><span class="swatch" style="background:${l.hex}"></span> ${l.hex}</td>
        <td>${fmtPct(l.coverage)}</td>
        <td>${l.height_mm.toFixed(2)} mm</td>
      </tr>`
    )
    .join("");
  const warnings = rep.warnings.map((w) => `<p class="report-line warn">${escapeHtml(w)}</p>`).join("");
  el.innerHTML = `
    <img src="${run.url}" alt="levels preview">
    <table class="palette-table">
      <tr><th>Level</th><th>Color</th><th>Coverage</th><th>Height</th></tr>
      ${rows}
    </table>
    <p class="report-line">mean residual &Delta;E ${rep.mean_residual.toFixed(1)}, ${fmtPct(rep.bad_fraction)} unexplained</p>
    ${warnings}
    <details class="log"><summary>full log</summary><pre>${escapeHtml(run.log)}</pre></details>
  `;
}

const BUILD_DEFAULTS = { base_mm: 2.0, step_mm: 0.4, relief_mm: 1.6, layer_mm: 0.05 };

function fillBuildForm() {
  const f = buildForm;
  const levelsRun = activeRun("levels_runs", state.active_levels);
  const last = state.build_runs.length ? state.build_runs[state.build_runs.length - 1].params : null;
  const v = Object.assign({}, BUILD_DEFAULTS, last || {}, levelsRun ? {} : {});
  f.base_mm.value = v.base_mm;
  f.step_mm.value = v.step_mm;
  f.relief_mm.value = v.relief_mm;
  f.layer_mm.value = v.layer_mm;
}

function renderBuildResult() {
  const el = document.getElementById("build-result");
  const run = activeRun("build_runs", state.active_build);
  if (!run) {
    el.innerHTML = `<p class="placeholder">Not built yet.</p>`;
    return;
  }
  const status = run.watertight && run.volume_ok
    ? `<p class="report-line ok">watertight, volume matches (${run.volume_mesh.toFixed(1)} mm&sup3;)</p>`
    : `<p class="report-line warn">check failed: open edges ${run.open_edges}, non-manifold ${run.nonmanifold_edges}</p>`;
  const plateWarning = run.plate_warning
    ? `<p class="report-line warn">exceeds the configured print plate size</p>`
    : "";
  el.innerHTML = `
    <img src="${run.relief_preview_url}" alt="relief preview">
    <img src="${run.levels_preview_url}" alt="levels preview" style="margin-top:0.5rem">
    <p class="report-line">${run.triangles.toLocaleString()} triangles, ${run.size_mb.toFixed(1)} MB &mdash;
      ${run.size_mm[0].toFixed(1)} &times; ${run.size_mm[1].toFixed(1)} &times; ${run.size_mm[2].toFixed(2)} mm</p>
    ${status}
    ${plateWarning}
    <a class="download" href="${run.stl_url}" download>Download ${run.id}.stl</a>
    <details class="log"><summary>full log</summary><pre>${escapeHtml(run.log)}</pre></details>
  `;
}

function renderHistory(elId, runs, activeId, step, thumbUrl, onDelete) {
  const el = document.getElementById(elId);
  el.innerHTML = "";
  for (const r of runs) {
    const t = document.createElement("div");
    t.className = "thumb" + (r.id === activeId ? " active" : "");
    const url = thumbUrl(r);
    t.innerHTML = url
      ? `<img src="${url}">`
      : `<div class="stl-icon">stl</div>`;
    t.innerHTML += `<div class="label">#${r.id}</div>`;
    t.addEventListener("click", () => selectRun(step, r.id));
    if (onDelete) {
      const del = document.createElement("button");
      del.type = "button";
      del.className = "thumb-delete";
      del.title = "Delete this image";
      del.textContent = "×";
      del.addEventListener("click", (e) => {
        e.stopPropagation();
        onDelete(r.id);
      });
      t.appendChild(del);
    }
    el.appendChild(t);
  }
}

// ---------- boot ----------

(async function boot() {
  await loadSessionList();
  const params = new URLSearchParams(location.search);
  const sid = params.get("session");
  if (sid) {
    try {
      await selectSession(sid);
      return;
    } catch {
      history.replaceState(null, "", location.pathname);
    }
  }
  startPanel.hidden = false;
  wizard.hidden = true;
})();
