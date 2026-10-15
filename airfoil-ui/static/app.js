(() => {
"use strict";

const $ = (s, r = document) => r.querySelector(s);
const NS = "http://www.w3.org/2000/svg";
const INK = "#16302a", GRAPHITE = "#56635d", PEN = "#c01f4c";

const S = {
  status: null,
  name: null,
  foil: null,          // { name, coords, geometry }
  alpha: 4,
  re: 1e6,
  cons: { t_min: null, t_max: null, cl_min: null },
  pred: null,          // prediction for baseline at the current flow
  job: null,           // running job id
  result: null,        // finished optimizer result
  resultName: null,
  list: [], active: -1,
  predictTimer: null, predictSeq: 0, setClDefault: false,
  firstDraw: true,
};

// ---------- helpers ----------
async function api(url, opts) {
  let res;
  try {
    res = await fetch(url, opts);
  } catch (e) {
    throw new Error("Can't reach the app server. Check that app.py is still running.");
  }
  let data = null;
  try { data = await res.json(); } catch (e) { /* not json */ }
  if (!res.ok) throw new Error((data && data.error) || `The server returned an error (${res.status}).`);
  return data;
}

function svgEl(tag, attrs = {}, parent) {
  const e = document.createElementNS(NS, tag);
  for (const k in attrs) e.setAttribute(k, attrs[k]);
  if (parent) parent.appendChild(e);
  return e;
}
function text(parent, x, y, str, cls, attrs = {}) {
  const t = svgEl("text", { x, y, class: cls || "", ...attrs }, parent);
  t.textContent = str;
  return t;
}
const f = (n, d = 3) => (n == null || !isFinite(n) ? "n/a" : Number(n).toFixed(d));
const pct = (n, d = 1) => `${(n * 100).toFixed(d)}%`;
const signed = (n, d = 1, unit = "%") => (n >= 0 ? "+" : "-") + Math.abs(n).toFixed(d) + unit;
const displayName = (n) => (n || "").toUpperCase().replace(/^NACA/, "NACA ");
const fmtRe = (v) => Math.round(v).toLocaleString("en-US");
const parseNum = (s) => {
  const t = String(s).trim().replace(/,/g, "");
  if (t === "" || !/^[-+]?(\d+\.?\d*|\.\d+)(e[-+]?\d+)?$/i.test(t)) return NaN;
  return Number(t);
};
const LOG_MIN = Math.log10(5e4), LOG_MAX = Math.log10(1e7);
const reToSlider = (re) => Math.round(((Math.log10(re) - LOG_MIN) / (LOG_MAX - LOG_MIN)) * 1000);
const sliderToRe = (v) => {
  const raw = Math.pow(10, LOG_MIN + (v / 1000) * (LOG_MAX - LOG_MIN));
  const mag = Math.pow(10, Math.floor(Math.log10(raw)) - 1);
  return Math.round(raw / mag) * mag;
};

// ---------- title block ----------
function renderTitleBlock() {
  const st = S.status;
  if (!st) return;
  const src = st.sources;
  const set = (id, label, isDemo) => {
    const el = $(id);
    el.textContent = label;
    el.classList.toggle("demo-src", !!isDemo);
  };
  const lab = st.labels || {};
  set("#tb-primary", src.primary === "project" ? (lab.primary || "Your project code") : "Demo stand-in", src.primary !== "project");
  set("#tb-check", src.check === "project" ? (lab.check || "Your project code") : src.check === "demo" ? "Demo stand-in" : "Not connected", src.check !== "project");
  set("#tb-opt", src.optimizer === "project" ? (lab.optimizer || "Your project code") : "Built-in search", src.optimizer !== "project");
  const n = st.library_count;
  set("#tb-lib", n > 0 ? `${n.toLocaleString("en-US")} UIUC files + 7 NACA` : "7 built-in NACA sections", n === 0);

  const note = $("#demo-note");
  const parts = [];
  if (st.hooks_error) parts.push(`project_hooks.py could not be loaded (${st.hooks_error}). Fix that file and restart app.py.`);
  else if (src.primary !== "project") parts.push("These numbers come from a demo stand-in, not your project code. Connect it in project_hooks.py.");
  note.textContent = parts.join(" ");
  note.hidden = parts.length === 0;

  const fn = $("#foil-note");
  fn.textContent = n > 0
    ? "Showing the closest matches. Type to narrow the list."
    : "Only the 7 built-in NACA sections are loaded. To search your UIUC files, copy your airfoils folder (the .dat files) into the airfoil-ui folder, then restart app.py.";
}

// ---------- airfoil list ----------
let listSeq = 0;
async function loadList(q) {
  const seq = ++listSeq;
  try {
    const d = await api(`/api/airfoils?q=${encodeURIComponent(q)}`);
    if (seq !== listSeq) return;
    S.list = d.names;
    S.active = -1;
    renderList();
  } catch (e) {
    if (seq !== listSeq) return;
    S.list = [];
    renderList(e.message);
  }
}
function renderList(errMsg) {
  const ul = $("#foil-list");
  ul.textContent = "";
  if (errMsg || S.list.length === 0) {
    const li = document.createElement("li");
    li.className = "empty-row";
    li.textContent = errMsg || `No airfoil matches "${$("#foil-q").value.trim()}". Try fewer characters.`;
    ul.appendChild(li);
    return;
  }
  S.list.forEach((n, i) => {
    const li = document.createElement("li");
    li.id = `foil-opt-${i}`;
    li.setAttribute("role", "option");
    li.setAttribute("aria-selected", n === S.name ? "true" : "false");
    if (i === S.active) li.classList.add("active");
    li.textContent = displayName(n);
    li.addEventListener("click", () => selectAirfoil(n));
    ul.appendChild(li);
  });
  const q = $("#foil-q");
  if (S.active >= 0) q.setAttribute("aria-activedescendant", `foil-opt-${S.active}`);
  else q.removeAttribute("aria-activedescendant");
}

let qTimer = null;
$("#foil-q").addEventListener("input", (e) => {
  clearTimeout(qTimer);
  qTimer = setTimeout(() => loadList(e.target.value), 140);
});
$("#foil-q").addEventListener("keydown", (e) => {
  if (e.key === "ArrowDown" || e.key === "ArrowUp") {
    if (!S.list.length) return;
    e.preventDefault();
    S.active = (S.active + (e.key === "ArrowDown" ? 1 : -1) + S.list.length) % S.list.length;
    renderList();
    const el = $(`#foil-opt-${S.active}`);
    if (el) el.scrollIntoView({ block: "nearest" });
  } else if (e.key === "Enter") {
    e.preventDefault();
    const pick = S.active >= 0 ? S.list[S.active] : S.list[0];
    if (pick) selectAirfoil(pick);
  }
});

// ---------- selecting an airfoil ----------
async function selectAirfoil(name) {
  try {
    const d = await api(`/api/airfoil/${encodeURIComponent(name)}`);
    if (S.job) stopPolling();
    S.name = d.name;
    S.foil = d;
    S.result = null; S.resultName = null; S.pred = null;
    const t = d.geometry.max_thickness;
    S.cons.t_min = Math.max(0.02, Math.round((t - 0.03) * 200) / 200);
    S.cons.t_max = Math.min(0.4, Math.round((t + 0.03) * 200) / 200);
    S.setClDefault = true;
    syncConstraintInputs();
    $("#run-state").textContent = "";
    $("#run").disabled = false;
    $("#run").textContent = "Run optimizer";
    renderList();
    renderHead();
    renderAll();
    schedulePredict(0);
  } catch (e) {
    $("#tbl-msg").textContent = e.message;
  }
}

function renderHead() {
  const g = S.foil.geometry;
  $("#foil-name").textContent = displayName(S.foil.name);
  $("#foil-desc").textContent =
    `Thickest at ${pct(g.max_thickness)} of chord, ${pct(g.max_thickness_x, 0)} back from the leading edge. ` +
    (Math.abs(g.max_camber) < 0.0005
      ? "Symmetric, with no camber."
      : `Camber peaks at ${pct(Math.abs(g.max_camber))} of chord, ${pct(g.max_camber_x, 0)} back.`);
  document.title = `${displayName(S.foil.name)} - Airfoil optimizer`;
}

// ---------- inputs ----------
function setErr(id, msg) {
  const inp = $(`#${id}-n`);
  $(`#${id}-err`).textContent = msg || "";
  inp.setAttribute("aria-invalid", msg ? "true" : "false");
}

function syncConstraintInputs() {
  $("#tmin-n").value = (S.cons.t_min * 100).toFixed(1);
  $("#tmax-n").value = (S.cons.t_max * 100).toFixed(1);
  $("#clmin-n").value = S.cons.cl_min == null ? "" : S.cons.cl_min.toFixed(2);
  ["tmin", "tmax", "clmin"].forEach((id) => setErr(id, ""));
}

function onAlpha(raw, fromSlider) {
  const v = parseNum(raw);
  if (isNaN(v) || v < -10 || v > 20) { setErr("alpha", "Angle of attack must be a number from -10 to 20 degrees."); return; }
  setErr("alpha", "");
  S.alpha = v;
  if (!fromSlider) $("#alpha-r").value = v;
  else $("#alpha-n").value = String(v);
  schedulePredict(180);
}
$("#alpha-n").addEventListener("input", (e) => onAlpha(e.target.value, false));
$("#alpha-r").addEventListener("input", (e) => onAlpha(e.target.value, true));

function onRe(raw, fromSlider) {
  const v = fromSlider ? sliderToRe(Number(raw)) : parseNum(raw);
  if (isNaN(v) || v < 5e4 || v > 1e7) { setErr("re", "Reynolds number must be between 50,000 and 10,000,000. You can type 1e6 for one million."); return; }
  setErr("re", "");
  S.re = v;
  if (fromSlider) $("#re-n").value = fmtRe(v);
  else $("#re-r").value = reToSlider(v);
  schedulePredict(180);
}
$("#re-n").addEventListener("input", (e) => onRe(e.target.value, false));
$("#re-n").addEventListener("blur", () => { if (!$("#re-err").textContent) $("#re-n").value = fmtRe(S.re); });
$("#re-r").addEventListener("input", (e) => onRe(e.target.value, true));

function onCons(id, key, scale, lo, hi, label, unit) {
  $(`#${id}-n`).addEventListener("input", (e) => {
    const v = parseNum(e.target.value);
    if (isNaN(v) || v < lo || v > hi) { setErr(id, `${label} must be a number from ${lo} to ${hi}${unit}.`); return; }
    setErr(id, "");
    S.cons[key] = v / scale;
  });
}
onCons("tmin", "t_min", 100, 2, 40, "Thinnest allowed", " % of chord");
onCons("tmax", "t_max", 100, 2, 40, "Thickest allowed", " % of chord");
onCons("clmin", "cl_min", 1, -2, 3, "Lowest lift", "");

// ---------- prediction ----------
function schedulePredict(delay) {
  clearTimeout(S.predictTimer);
  S.predictTimer = setTimeout(doPredict, delay);
}
async function doPredict() {
  if (!S.foil) return;
  const seq = ++S.predictSeq;
  try {
    const d = await api("/api/predict", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ name: S.name, alpha: S.alpha, re: S.re }),
    });
    if (seq !== S.predictSeq) return;
    S.pred = d;
    if (S.setClDefault) {
      S.setClDefault = false;
      S.cons.cl_min = Math.floor((d.cl - 0.03) / 0.05) * 0.05;
      $("#clmin-n").value = S.cons.cl_min.toFixed(2);
    }
    $("#tbl-msg").textContent = "";
    renderTable();
    renderDrawing();
  } catch (e) {
    if (seq !== S.predictSeq) return;
    S.pred = null;
    $("#tbl-msg").textContent = e.message;
    renderTable();
  }
}

// ---------- running the optimizer ----------
let pollTimer = null;
function stopPolling() { clearTimeout(pollTimer); pollTimer = null; S.job = null; }

function validateAll() {
  const bad = ["alpha", "re", "tmin", "tmax", "clmin"].find((id) => $(`#${id}-err`).textContent);
  if (bad) { $(`#${bad}-n`).focus(); return false; }
  if (S.cons.cl_min == null || isNaN(S.cons.cl_min)) { setErr("clmin", "Enter the lowest lift (CL) the optimized shape may have."); $("#clmin-n").focus(); return false; }
  if (S.cons.t_min > S.cons.t_max) { setErr("tmin", "Thinnest allowed must not be larger than thickest allowed."); $("#tmin-n").focus(); return false; }
  return true;
}

$("#run").addEventListener("click", async () => {
  if (!S.foil || S.job || !validateAll()) return;
  const btn = $("#run");
  btn.disabled = true; btn.textContent = "Optimizing";
  $("#run-state").innerHTML = "";
  renderRunState({ generation: 0, total: 0, history: [] });
  try {
    const d = await api("/api/optimize", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ name: S.name, alpha: S.alpha, re: S.re, constraints: { ...S.cons } }),
    });
    S.job = d.id;
    poll(d.id, S.name);
  } catch (e) {
    runFailed(e.message);
  }
});

function runFailed(msg) {
  stopPolling();
  const btn = $("#run");
  btn.disabled = false; btn.textContent = S.result ? "Run again" : "Run optimizer";
  const box = $("#run-state");
  box.textContent = "";
  const p = document.createElement("p");
  p.className = "fail";
  p.textContent = msg;
  box.appendChild(p);
}

async function poll(id, name) {
  try {
    const j = await api(`/api/jobs/${id}`);
    if (S.job !== id || S.name !== name) return;
    if (j.status === "running") {
      renderRunState(j);
      pollTimer = setTimeout(() => poll(id, name), 350);
    } else if (j.status === "done") {
      stopPolling();
      S.result = j.result; S.resultName = name;
      S.lastJob = id;
      const btn = $("#run");
      btn.disabled = false; btn.textContent = "Run again";
      $("#run-state").textContent = "";
      renderAll(true);
    } else {
      runFailed(j.error || "The optimizer stopped without a result.");
    }
  } catch (e) {
    if (S.job === id) runFailed(e.message);
  }
}

function renderRunState(j) {
  const box = $("#run-state");
  box.textContent = "";
  const label = document.createElement("div");
  const best = j.history && j.history.length ? j.history[j.history.length - 1] : null;
  label.textContent = j.total
    ? `Generation ${j.generation} of ${j.total}` + (best != null ? `, best L/D ${best.toFixed(1)}` : "")
    : "Starting";
  box.appendChild(label);
  const bar = document.createElement("div"); bar.className = "bar";
  const i = document.createElement("i"); i.style.width = `${Math.round((j.progress || 0) * 100)}%`;
  bar.appendChild(i); box.appendChild(bar);
  const hist = (j.history || []).filter((v) => v != null);
  if (hist.length > 1) {
    const sv = svgEl("svg", { class: "spark", viewBox: "0 0 260 34", preserveAspectRatio: "none", role: "img", "aria-label": "Best L/D found so far, by generation" });
    const lo = Math.min(...hist), hi = Math.max(...hist), span = hi - lo || 1;
    const pts = hist.map((v, k) => `${(k / Math.max(j.total - 1, 1)) * 258 + 1},${31 - ((v - lo) / span) * 28}`);
    svgEl("polyline", { points: pts.join(" "), fill: "none", stroke: PEN, "stroke-width": 1.6, "vector-effect": "non-scaling-stroke" }, sv);
    box.appendChild(sv);
  }
}

// ---------- results table ----------
function row(label, b, o, change, cls, sub) {
  const tr = document.createElement("tr");
  if (sub) tr.className = "sub";
  const cells = [[label, ""], [b, "muted"], [o, o === "-" ? "none" : "opt"], [change, change === "-" ? "none" : ""]];
  cells.forEach(([v, c], k) => {
    const td = document.createElement("td");
    td.textContent = v;
    if (k === 0) td.scope = "row";
    if (k > 0 && k < 3) td.className = k === 1 ? "" : (c || "");
    if (k === 3 && c) td.className = c;
    tr.appendChild(td);
  });
  return tr;
}
const rel = (a, b) => (a && b != null && Math.abs(a) > 1e-9 ? ((b - a) / Math.abs(a)) * 100 : null);

function renderTable() {
  const body = $("#tbl-body");
  body.textContent = "";
  const res = S.result && S.resultName === S.name ? S.result : null;
  const base = res ? res.baseline.at_design : S.pred;
  const opt = res ? res.optimized.at_design : null;
  if (!base) {
    const tr = document.createElement("tr");
    const td = document.createElement("td");
    td.colSpan = 4; td.className = "none"; td.textContent = "Waiting for a prediction.";
    tr.appendChild(td); body.appendChild(tr);
    $("#verdict").textContent = "";
    return;
  }
  const chg = (a, b, goodDown) => {
    const r = rel(a, b);
    return r == null ? "-" : signed(r, 1);
  };
  body.appendChild(row("Lift coefficient CL", f(base.cl), opt ? f(opt.cl) : "-", opt ? chg(base.cl, opt.cl) : "-"));
  body.appendChild(row("Drag coefficient CD", f(base.cd, 4), opt ? f(opt.cd, 4) : "-", opt ? chg(base.cd, opt.cd) : "-"));
  body.appendChild(row("Lift-to-drag ratio L/D", base.ld != null ? base.ld.toFixed(1) : "n/a", opt && opt.ld != null ? opt.ld.toFixed(1) : "-", opt ? chg(base.ld, opt.ld) : "-"));
  if (res && base.check && opt && opt.check) {
    body.appendChild(row(`L/D by the check model`, base.check.ld != null ? base.check.ld.toFixed(1) : "n/a", opt.check.ld != null ? opt.check.ld.toFixed(1) : "n/a", chg(base.check.ld, opt.check.ld), "", true));
  }
  const v = $("#verdict");
  v.textContent = "";
  if (res) {
    const parts = [];
    if (opt.ld != null && base.ld != null) {
      const gain = rel(base.ld, opt.ld);
      parts.push(`At ${S.alpha}° the optimized shape has an L/D of ${opt.ld.toFixed(1)} against ${base.ld.toFixed(1)} for the baseline (${signed(gain, 1)}).`);
    }
    if (opt.check && opt.ld != null && opt.check.ld != null) {
      const d = Math.abs(opt.check.ld - opt.ld) / Math.abs(opt.ld) * 100;
      const sameWay = base.check && base.check.ld != null && (opt.check.ld > base.check.ld) === (opt.ld > base.ld);
      parts.push(`The check model puts the optimized L/D within ${d.toFixed(0)}% of the main model and ${sameWay ? "agrees" : "does not agree"} that the shape improved.`);
    }
    parts.push(`Across the sweep it wins at ${res.angles_better} of ${res.angles_total} angles.`);
    v.textContent = parts.join(" ");
  }
}

// ---------- drawing ----------
function renderDrawing(animate) {
  const svg = $("#drawing");
  svg.textContent = "";
  if (!S.foil) return;
  const res = S.result && S.resultName === S.name ? S.result : null;
  const base = S.foil.coords;
  const opt = res ? res.optimized.coords : null;
  const W = 1000, X0 = 150, CH = 780, CY = 170, AIR_H = 340;
  const H = res ? 440 : AIR_H;
  svg.setAttribute("viewBox", `0 0 ${W} ${H}`);

  let maxY = 0.01;
  for (const [, y] of base) maxY = Math.max(maxY, Math.abs(y));
  if (opt) for (const [, y] of opt) maxY = Math.max(maxY, Math.abs(y));
  const sc = Math.min(CH, 135 / maxY);
  const cw = sc; // drawn chord length
  const X = (x) => X0 + x * sc, Y = (y) => CY - y * sc;

  // grid tied to chord fractions: minor every 0.05c, major every 0.1c
  const gridG = svgEl("g", {}, svg);
  for (let i = 0; i <= 20; i++) {
    const x = X0 + (i / 20) * cw;
    svgEl("line", { x1: x, y1: 24, x2: x, y2: AIR_H - 24, stroke: INK, "stroke-opacity": i % 2 ? 0.07 : 0.15, "stroke-width": 1 }, gridG);
  }
  for (let k = -3; k <= 3; k++) {
    const y = CY + (k / 20) * cw;
    if (y < 24 || y > AIR_H - 24) continue;
    svgEl("line", { x1: X0, y1: y, x2: X0 + cw, y2: y, stroke: INK, "stroke-opacity": k % 2 ? 0.07 : 0.15, "stroke-width": 1 }, gridG);
  }
  [0, 0.5, 1].forEach((c) => {
    text(svg, X(c), AIR_H - 6, c === 0 ? "0" : c === 1 ? "1.0 chord" : "0.5", "tick-text", { "text-anchor": c === 1 ? "end" : "middle" });
  });

  // chord line
  svgEl("line", { x1: X0 - 8, y1: CY, x2: X0 + cw + 8, y2: CY, stroke: INK, "stroke-width": 1, "stroke-opacity": 0.55 }, svg);

  // free stream arrows showing the angle of attack
  const a = (S.alpha * Math.PI) / 180, L = 86;
  const defs = svgEl("defs", {}, svg);
  const mk = svgEl("marker", { id: "ah", viewBox: "0 0 10 10", refX: 9, refY: 5, markerWidth: 8, markerHeight: 8, orient: "auto-start-reverse" }, defs);
  svgEl("path", { d: "M0 1 L10 5 L0 9", fill: "none", stroke: GRAPHITE, "stroke-width": 1.6 }, mk);
  [-52, 0, 52].forEach((dy) => {
    const sx = 26, sy = CY + dy - Math.sin(a) * L * 0.5;
    svgEl("line", { x1: sx, y1: sy, x2: sx + Math.cos(a) * L, y2: sy + Math.sin(a) * L, stroke: GRAPHITE, "stroke-width": 1.4, "marker-end": "url(#ah)" }, svg);
  });
  text(svg, 26, CY + 52 + 34, `Flow at ${S.alpha}° to the chord`, "tick-text");

  // shapes
  const pathD = (c) => c.map(([x, y], i) => `${i ? "L" : "M"}${X(x).toFixed(1)} ${Y(y).toFixed(1)}`).join("") + "Z";
  const anim = S.firstDraw || animate;
  svgEl("path", { d: pathD(base), pathLength: 1, fill: opt ? "none" : INK, "fill-opacity": 0.06, stroke: opt ? GRAPHITE : INK, "stroke-width": opt ? 1.6 : 2.2, "stroke-linejoin": "round", class: S.firstDraw ? "draw-in" : "" }, svg);

  // camber line
  const g = S.foil.geometry;
  const cd = g.x.map((x, i) => `${i ? "L" : "M"}${X(x).toFixed(1)} ${Y(g.camber[i]).toFixed(1)}`).join("");
  svgEl("path", { d: cd, fill: "none", stroke: opt ? GRAPHITE : INK, "stroke-width": 1, "stroke-dasharray": "5 4", "stroke-opacity": 0.7 }, svg);
  if (Math.abs(g.max_camber) > 0.0005) svgEl("circle", { cx: X(g.max_camber_x), cy: Y(g.max_camber), r: 3.2, fill: "none", stroke: opt ? GRAPHITE : INK, "stroke-width": 1.3 }, svg);

  // thickness dimension for the baseline
  dimension(svg, X(g.max_thickness_x), Y(g.yu_at_t), Y(g.yl_at_t), `t ${pct(g.max_thickness)}`, opt ? GRAPHITE : INK, "above");

  if (opt) {
    const og = res.optimized.geometry;
    svgEl("path", { d: pathD(opt), pathLength: 1, fill: PEN, "fill-opacity": 0.08, stroke: PEN, "stroke-width": 2.2, "stroke-linejoin": "round", class: animate ? "draw-in" : "" }, svg);
    dimension(svg, X(og.max_thickness_x), Y(og.yu_at_t), Y(og.yl_at_t), `t ${pct(og.max_thickness)}`, PEN, "below");

    // surface change strip
    const ile = (c) => c.reduce((m, p, i) => (p[0] < c[m][0] ? i : m), 0);
    const ib = ile(base), io = ile(opt);
    if (ib === io && base.length === opt.length) {
      const dU = [], dL = [];
      for (let i = 0; i < base.length; i++) {
        const d = opt[i][1] - base[i][1];
        (i <= ib ? dU : dL).push([base[i][0], d]);
      }
      const maxD = Math.max(0.004, ...dU.concat(dL).map((p) => Math.abs(p[1])));
      const zy = 392, k = 34 / maxD;
      svgEl("line", { x1: X0, y1: zy, x2: X0 + cw, y2: zy, stroke: INK, "stroke-opacity": 0.45, "stroke-width": 1 }, svg);
      const line = (pts, dash) => svgEl("path", { d: pts.map(([x, d], i) => `${i ? "L" : "M"}${X(x).toFixed(1)} ${(zy - d * k).toFixed(1)}`).join(""), fill: "none", stroke: PEN, "stroke-width": 1.8, "stroke-dasharray": dash || "none" }, svg);
      line(dU.slice().sort((a, b) => a[0] - b[0]));
      line(dL.slice().sort((a, b) => a[0] - b[0]), "6 4");
      text(svg, 10, zy - 16, "Surface change", "legend-text");
      text(svg, 10, zy + 2, "solid: upper side", "tick-text");
      text(svg, 10, zy + 18, "dashed: lower side", "tick-text");
      text(svg, 10, zy + 34, `max ${pct(maxD, 1)} of chord`, "tick-text");
    }

    // legend
    const lx = X0 + cw - 150;
    svgEl("line", { x1: lx, y1: 20, x2: lx + 26, y2: 20, stroke: GRAPHITE, "stroke-width": 2 }, svg);
    text(svg, lx + 34, 25, "Baseline", "legend-text");
    svgEl("line", { x1: lx, y1: 42, x2: lx + 26, y2: 42, stroke: PEN, "stroke-width": 2.6 }, svg);
    text(svg, lx + 34, 47, "Optimized", "legend-text", { fill: "#a3173f" });
  }
  S.firstDraw = false;
}

function dimension(svg, x, y1, y2, label, color, where) {
  svgEl("line", { x1: x, y1, x2: x, y2, stroke: color, "stroke-width": 1.2 }, svg);
  svgEl("line", { x1: x - 5, y1, x2: x + 5, y2: y1, stroke: color, "stroke-width": 1.2 }, svg);
  svgEl("line", { x1: x - 5, y1: y2, x2: x + 5, y2, stroke: color, "stroke-width": 1.2 }, svg);
  const ty = where === "above" ? Math.min(y1, y2) - 10 : Math.max(y1, y2) + 20;
  text(svg, x, ty, label, "dim-text", { "text-anchor": "middle", fill: color === PEN ? "#a3173f" : color });
}

// ---------- sweep plots ----------
const plots = [];
function niceTicks(lo, hi, n = 4) {
  const span = hi - lo || 1;
  const step0 = span / n, mag = Math.pow(10, Math.floor(Math.log10(step0)));
  const step = [1, 2, 2.5, 5, 10].map((m) => m * mag).find((s) => s >= step0) || mag * 10;
  const a = Math.floor(lo / step) * step, b = Math.ceil(hi / step) * step, out = [];
  for (let v = a; v <= b + step / 2; v += step) out.push(+v.toFixed(10));
  return out;
}

function makePlot(host, cfg) {
  host.textContent = "";
  const head = document.createElement("div"); head.className = "ph";
  const t = document.createElement("span"); t.className = "pt"; t.textContent = cfg.title;
  const rd = document.createElement("span"); rd.className = "pv";
  head.append(t, rd); host.appendChild(head);

  const w = 320, h = 210, m = { l: 44, r: 10, t: 10, b: 30 };
  const svg = svgEl("svg", { viewBox: `0 0 ${w} ${h}`, role: "img", "aria-label": cfg.aria }, host);
  const all = cfg.series.flatMap((s) => s.ys.filter((v) => v != null && isFinite(v)));
  let lo = Math.min(...all), hi = Math.max(...all);
  if (cfg.zero) lo = Math.min(lo, 0);
  const ticks = niceTicks(lo, hi);
  const ylo = ticks[0], yhi = ticks[ticks.length - 1];
  const xs = cfg.xs, xlo = xs[0], xhi = xs[xs.length - 1];
  const X = (v) => m.l + ((v - xlo) / (xhi - xlo)) * (w - m.l - m.r);
  const Y = (v) => h - m.b - ((v - ylo) / (yhi - ylo || 1)) * (h - m.t - m.b);

  (cfg.bands || []).forEach(([a, b]) => svgEl("rect", { x: X(a), y: m.t, width: Math.max(1, X(b) - X(a)), height: h - m.t - m.b, fill: PEN, "fill-opacity": 0.1 }, svg));
  ticks.forEach((v) => {
    svgEl("line", { x1: m.l, x2: w - m.r, y1: Y(v), y2: Y(v), stroke: INK, "stroke-opacity": v === 0 ? 0.45 : 0.12 }, svg);
    text(svg, m.l - 6, Y(v) + 4, +v.toPrecision(3), "tick-text", { "text-anchor": "end" });
  });
  for (let v = -4; v <= 16; v += 4) {
    svgEl("line", { x1: X(v), x2: X(v), y1: h - m.b, y2: h - m.b + 4, stroke: INK, "stroke-opacity": 0.5 }, svg);
    text(svg, X(v), h - m.b + 17, v, "tick-text", { "text-anchor": "middle" });
  }
  text(svg, w - m.r, h - 2, "angle of attack, deg", "tick-text", { "text-anchor": "end" });
  svgEl("line", { x1: m.l, x2: w - m.r, y1: h - m.b, y2: h - m.b, stroke: INK, "stroke-opacity": 0.6 }, svg);

  if (cfg.mark >= xlo && cfg.mark <= xhi) svgEl("line", { x1: X(cfg.mark), x2: X(cfg.mark), y1: m.t, y2: h - m.b, stroke: INK, "stroke-width": 1.2, "stroke-dasharray": "4 3", "stroke-opacity": 0.7 }, svg);

  cfg.series.forEach((s) => {
    let d = "", pen = false;
    s.ys.forEach((v, i) => {
      if (v == null || !isFinite(v)) { pen = false; return; }
      d += `${pen ? "L" : "M"}${X(xs[i]).toFixed(1)} ${Y(v).toFixed(1)}`; pen = true;
    });
    svgEl("path", { d, fill: "none", stroke: s.color, "stroke-width": s.width, "stroke-linejoin": "round", "stroke-linecap": "round" }, svg);
  });

  const cross = svgEl("line", { y1: m.t, y2: h - m.b, stroke: INK, "stroke-width": 1, visibility: "hidden" }, svg);
  const dots = cfg.series.map((s) => svgEl("circle", { r: 3.6, fill: s.color, visibility: "hidden" }, svg));
  const overlay = svgEl("rect", { x: m.l, y: m.t, width: w - m.l - m.r, height: h - m.t - m.b, fill: "transparent" }, svg);
  const p = { svg, rd, cross, dots, X, Y, cfg, w, m, overlay };
  plots.push(p);
  rd.innerHTML = "";
  return p;
}

function setHover(i) {
  plots.forEach((p) => {
    if (i == null) {
      p.cross.setAttribute("visibility", "hidden");
      p.dots.forEach((d) => d.setAttribute("visibility", "hidden"));
      p.rd.textContent = "";
      return;
    }
    const x = p.X(p.cfg.xs[i]);
    p.cross.setAttribute("x1", x); p.cross.setAttribute("x2", x); p.cross.setAttribute("visibility", "visible");
    const parts = [`${p.cfg.xs[i]}°`];
    p.cfg.series.forEach((s, k) => {
      const v = s.ys[i];
      if (v != null && isFinite(v)) {
        p.dots[k].setAttribute("cx", x); p.dots[k].setAttribute("cy", p.Y(v)); p.dots[k].setAttribute("visibility", "visible");
      } else p.dots[k].setAttribute("visibility", "hidden");
    });
    p.rd.textContent = "";
    p.cfg.series.forEach((s, k) => {
      const v = s.ys[i];
      const span = document.createElement("span");
      if (k === 1) span.className = "o";
      span.textContent = (k ? " " : `${parts[0]}  `) + (v == null ? "n/a" : v.toFixed(p.cfg.digits));
      p.rd.appendChild(span);
    });
  });
}

function renderSweep() {
  const res = S.result && S.resultName === S.name ? S.result : null;
  $("#sweep-empty").hidden = !!res;
  $("#sweep-grid").hidden = !res;
  $("#sweep-key").hidden = !res;
  plots.length = 0;
  if (!res) return;
  const sw = res.sweep, xs = sw.alpha;
  const bands = [];
  let start = null;
  for (let i = 0; i < xs.length; i++) {
    const win = sw.opt.ld[i] != null && sw.base.ld[i] != null && sw.opt.ld[i] > sw.base.ld[i];
    const half = (xs[1] - xs[0]) / 2;
    if (win && start == null) start = xs[i] - half;
    if ((!win || i === xs.length - 1) && start != null) { bands.push([Math.max(xs[0], start), Math.min(xs[xs.length - 1], win ? xs[i] : xs[i - 1] + half)]); start = null; }
  }
  const mk = (id, title, key, digits, aria, withBands, zero) =>
    makePlot($(id), {
      title, xs, digits, aria, zero, mark: S.alpha,
      series: [
        { ys: sw.base[key], color: GRAPHITE, width: 1.8 },
        { ys: sw.opt[key], color: PEN, width: 2.4 },
      ],
      bands: withBands ? bands : [],
    });
  mk("#plot-cl", "Lift coefficient CL", "cl", 3, "CL against angle of attack for the baseline and optimized shapes", false, true);
  mk("#plot-cd", "Drag coefficient CD", "cd", 4, "CD against angle of attack for the baseline and optimized shapes", false, false);
  mk("#plot-ld", "Lift-to-drag ratio L/D", "ld", 1, "L/D against angle of attack, with the angles where the optimized shape wins shaded", true, true);

  plots.forEach((p) => {
    p.overlay.addEventListener("pointermove", (e) => {
      const r = p.svg.getBoundingClientRect();
      const px = ((e.clientX - r.left) / r.width) * p.w;
      let best = 0, bd = 1e9;
      p.cfg.xs.forEach((v, i) => { const d = Math.abs(p.X(v) - px); if (d < bd) { bd = d; best = i; } });
      setHover(best);
    });
    p.overlay.addEventListener("pointerleave", () => setHover(null));
  });

  $("#sweep-key").textContent =
    `Grey is the baseline and red is the optimized shape. The dashed line marks the design angle of ${S.alpha}°. ` +
    (bands.length ? "Red shading on the L/D plot marks angles where the optimized shape is better." : "The optimized shape is not better at any swept angle.") +
    " Move over a plot to read values.";
}

// ---------- exports ----------
function renderExports() {
  const res = S.result && S.resultName === S.name ? S.result : null;
  $("#exports").hidden = !res;
  const ul = $("#ex-list");
  ul.textContent = "";
  if (!res || !S.lastJob) return;
  [["optimized", "Optimized airfoil"], ["baseline", "Baseline airfoil"]].forEach(([which, label]) => {
    const li = document.createElement("li");
    const s = document.createElement("span"); s.textContent = label;
    const links = document.createElement("span"); links.className = "links";
    [["dat", "Download .dat"], ["csv", "Download .csv"]].forEach(([fmt, t]) => {
      const a = document.createElement("a");
      a.href = `/api/jobs/${S.lastJob}/export?which=${which}&format=${fmt}`;
      a.textContent = t;
      a.setAttribute("download", "");
      links.appendChild(a);
    });
    li.append(s, links); ul.appendChild(li);
  });
}

function renderAll(animateOpt) {
  renderTable();
  renderDrawing(animateOpt);
  renderSweep();
  renderExports();
}

// ---------- start ----------
async function init() {
  $("#re-n").value = fmtRe(S.re);
  $("#re-r").value = reToSlider(S.re);
  try {
    S.status = await api("/api/status");
    renderTitleBlock();
    await loadList("");
    await selectAirfoil(S.status.default_airfoil);
  } catch (e) {
    $("#foil-name").textContent = "Can't load";
    $("#foil-desc").textContent = e.message;
  }
}
init();
})();
