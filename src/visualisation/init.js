// SAE Feature Dashboard — X-Ray Conglomerate
// Clean two-panel layout: sidebar feature list + main detail panel.

// ─── Dataset config (must match DATASET_COLORS in html_dashboard.py) ─────────

const DATASET_COLORS = {
  chexpert:   "#3B82F6",
  rsna:       "#F59E0B",
  siim_covid: "#10B981",
  tb:         "#8B5CF6",
  unknown:    "#9CA3AF",
};

const DATASET_LABELS = {
  chexpert:   "CheXpert",
  rsna:       "RSNA",
  siim_covid: "SIIM-COVID",
  tb:         "TB Shenzhen",
  unknown:    "Unknown",
};

function dsColor(name) {
  return DATASET_COLORS[name] || DATASET_COLORS.unknown;
}
function dsLabel(name) {
  return DATASET_LABELS[name] || name;
}

// ─── Utilities ────────────────────────────────────────────────────────────────

function actColor(norm) {
  // norm in [0,1] → white → #f97316 (orange)
  return d3.interpolateRgb("#ffffff", "#f97316")(norm);
}

function fmt(v, d = 3) {
  return typeof v === "number" ? v.toFixed(d) : v;
}

// ─── Sidebar feature list ─────────────────────────────────────────────────────

function buildSidebar(allKeys) {
  const list = document.getElementById("feature-list");
  const search = document.getElementById("feature-search");

  function renderList(filter) {
    list.innerHTML = "";
    const keys = filter
      ? allKeys.filter(k => k.includes(filter))
      : allKeys;

    keys.forEach(k => {
      const featureData = DATA[k];
      // Universality score: 1 - max_fraction from universality profile
      let uScore = 0;
      if (featureData && featureData.universalityProfile) {
        const bars = featureData.universalityProfile.bars || [];
        const maxFrac = bars.length ? Math.max(...bars.map(b => b.fraction)) : 1;
        uScore = 1 - maxFrac;
      }

      const item = document.createElement("div");
      item.className = "feature-item";
      item.dataset.key = k;
      item.innerHTML = `
        <span class="fid">F${k}</span>
        <div class="u-score-bar">
          <div class="u-score-fill" style="width:${(uScore * 100).toFixed(0)}%"></div>
        </div>
        <span class="u-badge">${uScore.toFixed(2)}</span>
      `;
      item.addEventListener("click", () => {
        document.querySelectorAll(".feature-item").forEach(el => el.classList.remove("active"));
        item.classList.add("active");
        renderMainPanel(k);
      });
      list.appendChild(item);
    });
  }

  search.addEventListener("input", e => renderList(e.target.value.trim()));
  renderList("");

  return renderList;
}

function setActiveItem(key) {
  document.querySelectorAll(".feature-item").forEach(el => {
    el.classList.toggle("active", el.dataset.key === String(key));
  });
}

// ─── Feature header ───────────────────────────────────────────────────────────

function renderFeatureHeader(key, featureData) {
  const el = document.getElementById("feature-header");
  if (!el) return;

  const hist = featureData.actsHistogram || {};
  const n_active = hist.title ? (hist.title.match(/n=(\d+)/) || [])[1] || "—" : "—";
  const maxAct = hist.x && hist.x.length
    ? Math.max(...featureData.actsHistogram.x).toFixed(3)
    : "—";

  const bars = (featureData.universalityProfile || {}).bars || [];
  const uScore = bars.length
    ? (1 - Math.max(...bars.map(b => b.fraction))).toFixed(3)
    : "—";

  // Collect datasets present in top patches
  const dsSet = new Set((featureData.topPatches || []).map(p => p.dataset));
  const dsPills = [...dsSet].map(ds => `
    <span class="ds-pill" style="background:${dsColor(ds)}">
      <span class="ds-dot"></span>${dsLabel(ds)}
    </span>
  `).join("");

  el.innerHTML = `
    <div class="fh-id">F${key}</div>
    <div class="fh-meta">
      <div class="fh-stats">
        <div class="fh-stat">
          <span class="label">Active images</span>
          <span class="value">${n_active}</span>
        </div>
        <div class="fh-stat">
          <span class="label">Max activation</span>
          <span class="value">${maxAct}</span>
        </div>
        <div class="fh-stat">
          <span class="label">Universality</span>
          <span class="value" style="color:var(--accent)">${uScore}</span>
        </div>
      </div>
      <div class="fh-datasets">${dsPills}</div>
    </div>
  `;
}

// ─── Activation histogram ─────────────────────────────────────────────────────

function renderHistogram(featureData) {
  const histData = featureData.actsHistogram;
  const container = document.getElementById("hist-container");
  if (!container) return;

  if (!histData || !histData.x || histData.x.length === 0) {
    container.innerHTML = `<div class="dead-feature-msg">Dead feature — never activates</div>`;
    return;
  }

  const maxX = Math.max(...histData.x);
  const traces = [{
    x: histData.x,
    y: histData.y,
    type: "bar",
    marker: {
      color: histData.x.map(v => actColor(v / maxX)),
      line: { width: 0 },
    },
    hovertemplate: "act=%{x:.3f}<br>count=%{y}<extra></extra>",
  }];

  const layout = {
    paper_bgcolor: "rgba(0,0,0,0)",
    plot_bgcolor:  "rgba(0,0,0,0)",
    margin: { l: 44, r: 12, t: 8, b: 32 },
    xaxis: {
      title: { text: "activation", font: { size: 10 } },
      gridcolor: "#e9ecef",
      tickfont: { size: 10, family: "JetBrains Mono, monospace" },
      range: [0, maxX * 1.08],
    },
    yaxis: {
      title: { text: "images", font: { size: 10 } },
      gridcolor: "#e9ecef",
      tickfont: { size: 10 },
    },
    bargap: 0.04,
    showlegend: false,
    font: { family: "Inter, system-ui, sans-serif", size: 11 },
  };

  Plotly.newPlot(container, traces, layout, {
    responsive: true,
    displayModeBar: false,
  });
}

// ─── Universality profile ─────────────────────────────────────────────────────

function renderUniversalityProfile(featureData) {
  const profData = featureData.universalityProfile;
  const el = document.getElementById("u-profile-container");
  if (!el || !profData) return;

  const bars = profData.bars || [];
  if (bars.length === 0) {
    el.innerHTML = `<div class="empty-state">No universality data</div>`;
    return;
  }

  // Stacked bar
  const stackedHtml = bars.map(b => `
    <div class="u-stacked-seg"
         style="width:${(b.fraction * 100).toFixed(1)}%;background:${dsColor(b.dataset)}"
         title="${dsLabel(b.dataset)}: ${(b.fraction * 100).toFixed(1)}%">
    </div>
  `).join("");

  // Legend
  const legendHtml = bars.map(b => `
    <div class="u-legend-item">
      <div class="u-legend-swatch" style="background:${dsColor(b.dataset)}"></div>
      <span>${dsLabel(b.dataset)} <strong>${(b.fraction * 100).toFixed(0)}%</strong></span>
    </div>
  `).join("");

  const uScore = bars.length ? (1 - Math.max(...bars.map(b => b.fraction))).toFixed(3) : "—";

  el.innerHTML = `
    <div style="display:flex;justify-content:space-between;align-items:baseline;margin-bottom:10px">
      <span style="font-size:12px;color:var(--text-2)">Top-50 activating images by dataset</span>
      <span style="font-family:var(--font-mono);font-size:13px;font-weight:700;color:var(--accent)">u=${uScore}</span>
    </div>
    <div class="u-stacked-bar">${stackedHtml}</div>
    <div class="u-legend">${legendHtml}</div>
  `;
}

// ─── Top patches ──────────────────────────────────────────────────────────────

function renderTopPatches(key, featureData) {
  const el = document.getElementById("patches-container");
  if (!el) return;
  const patches = featureData.topPatches || [];

  if (patches.length === 0) {
    el.innerHTML = `<div class="empty-state">No activating patches found</div>`;
    return;
  }

  el.innerHTML = "";
  const grid = document.createElement("div");
  grid.className = "patches-grid";

  patches.forEach(p => {
    const cell = document.createElement("div");
    cell.className = "patch-cell";
    const color = dsColor(p.dataset);
    const label = dsLabel(p.dataset);
    cell.innerHTML = `
      <div class="patch-tooltip">
        F${key} · act=${fmt(p.act)} · img${p.image_idx} patch${p.patch_idx}
      </div>
      <img src="${p.img_b64}" alt="patch" loading="lazy" />
      <div class="patch-footer">
        <span class="patch-act">${fmt(p.act)}</span>
        <span class="patch-ds-badge" style="background:${color}">${label}</span>
      </div>
    `;
    grid.appendChild(cell);
  });

  el.appendChild(grid);
}

// ─── Co-activation / related features table ───────────────────────────────────

function renderCoactivationTable(featureData) {
  const el = document.getElementById("coact-container");
  if (!el) return;
  const rows = (featureData.featureTables || {}).correlatedFeatures || [];

  if (rows.length === 0) {
    el.innerHTML = `<div class="empty-state">No correlated features computed</div>`;
    return;
  }

  // Find max co-act value for bar scaling
  const maxCoact = Math.max(...rows.map(r => Math.abs(parseFloat(r.value)) || 0));

  const rowsHtml = rows.map(r => {
    const coact = parseFloat(r.value) || 0;
    const barPct = maxCoact > 0 ? (Math.abs(coact) / maxCoact * 100).toFixed(1) : 0;
    return `
      <tr>
        <td><span class="feat-link" onclick="jumpToFeature('${r.index}')">F${r.index}</span></td>
        <td class="r coact-bar-cell">
          <div class="coact-bar-bg" style="width:${barPct}%"></div>
          <span class="coact-bar-text">${r.value}</span>
        </td>
        <td class="r">${r.percentageL1}</td>
      </tr>
    `;
  }).join("");

  el.innerHTML = `
    <table class="coact-table">
      <thead>
        <tr>
          <th>Feature</th>
          <th class="r">Co-activation</th>
          <th class="r">Dec. cos-sim</th>
        </tr>
      </thead>
      <tbody>${rowsHtml}</tbody>
    </table>
  `;
}

function jumpToFeature(key) {
  if (!DATA[key]) return;
  setActiveItem(key);
  renderMainPanel(key);
  // Scroll sidebar item into view
  const item = document.querySelector(`.feature-item[data-key="${key}"]`);
  if (item) item.scrollIntoView({ behavior: "smooth", block: "nearest" });
}

// ─── Feature circuit graph ────────────────────────────────────────────────────

function renderCircuitGraph(key, featureData) {
  const container = document.getElementById("graph-container");
  if (!container) return;

  const graphList = featureData.attributionMiniGraph;
  if (!graphList || graphList.length === 0) {
    container.innerHTML = `<div class="empty-state" style="height:200px">
      Attribution graph unavailable (run with activations flag)</div>`;
    return;
  }

  container.innerHTML = `
    <div class="graph-controls">
      <label>Supernode threshold:</label>
      <input type="range" id="sn-slider" min="0" max="100" value="101" />
      <span id="sn-val" style="font-family:var(--font-mono);min-width:28px">OFF</span>
      <span style="color:var(--text-3);font-size:11px">← drag to merge similar features</span>
    </div>
    <div id="graph-svgs"></div>
    <div class="graph-legend">
      <div class="graph-legend-item">
        <div class="graph-legend-line" style="background:#3B82F6"></div>
        <span>Excitatory</span>
      </div>
      <div class="graph-legend-item">
        <div class="graph-legend-line" style="background:#EF4444"></div>
        <span>Inhibitory</span>
      </div>
      <div class="graph-legend-item">
        <div class="graph-legend-line" style="background:#D1D5DB"></div>
        <span>Weak (&lt;10%)</span>
      </div>
    </div>
  `;

  const svgContainer = document.getElementById("graph-svgs");
  const slider = document.getElementById("sn-slider");
  const snVal = document.getElementById("sn-val");
  const W = Math.max(380, container.clientWidth - 40);
  const H = 240;

  const svgRefs = [];

  graphList.forEach((graphData, i) => {
    const label = document.createElement("div");
    label.style.cssText = "font-size:11px;color:var(--text-3);margin:12px 0 4px;font-family:var(--font-mono)";
    label.textContent = `Image ${graphData.image_idx} · ${graphData.nodes.length} active features`;
    svgContainer.appendChild(label);

    const svgEl = d3.select(svgContainer)
      .append("svg")
      .attr("width", W).attr("height", H)
      .style("margin-bottom", "12px")
      .node();

    svgRefs.push({ svgEl, graphData });
    drawGraph(svgEl, graphData, W, H, 1.01);
  });

  slider.addEventListener("input", function () {
    const raw = +this.value;
    const threshold = raw > 100 ? 1.01 : raw / 100;
    snVal.textContent = raw > 100 ? "OFF" : threshold.toFixed(2);
    svgRefs.forEach(({ svgEl, graphData }) => drawGraph(svgEl, graphData, W, H, threshold));
  });
}

function drawGraph(svgEl, graphData, W, H, snThreshold) {
  const svg = d3.select(svgEl);
  svg.selectAll("*").remove();

  const cossimPairs = graphData.cossim_pairs || [];

  // Union-find for supernode merging
  const parent = {};
  graphData.nodes.forEach(n => { parent[n.id] = n.id; });
  function find(x) { return parent[x] === x ? x : (parent[x] = find(parent[x])); }
  function union(a, b) { parent[find(a)] = find(b); }

  if (snThreshold <= 1.0) {
    cossimPairs.forEach(p => { if (p.cossim >= snThreshold) union(p.i, p.j); });
  }

  // Build supernodes
  const groups = {};
  graphData.nodes.forEach(n => {
    const root = find(n.id);
    if (!groups[root]) groups[root] = { id: root, members: [], strength: 0, is_target: false, color: DATASET_COLORS.unknown };
    groups[root].members.push(n.id);
    groups[root].strength += n.strength;
    if (n.is_target) groups[root].is_target = true;
    if (n.dataset_color) groups[root].color = n.dataset_color;
  });
  const nodes = Object.values(groups);

  // Remap edges
  const edgeMap = {};
  graphData.edges.forEach(e => {
    const s = find(e.source), t = find(e.target);
    if (s === t) return;
    const k = `${s}-${t}`;
    if (!edgeMap[k]) edgeMap[k] = { source: s, target: t, weight: 0, excitatory: e.excitatory };
    edgeMap[k].weight += e.weight;
  });
  const edges = Object.values(edgeMap);

  // Arrow marker
  const defs = svg.append("defs");
  ["exc", "inh", "weak"].forEach(type => {
    const colors = { exc: "#3B82F6", inh: "#EF4444", weak: "#D1D5DB" };
    defs.append("marker")
      .attr("id", `arrow-${type}-${svgEl._uid || (svgEl._uid = Math.random().toString(36).slice(2))}`)
      .attr("viewBox", "0 -4 8 8").attr("refX", 14).attr("refY", 0)
      .attr("markerWidth", 5).attr("markerHeight", 5).attr("orient", "auto")
      .append("path").attr("d", "M0,-4L8,0L0,4Z").attr("fill", colors[type]);
  });
  const uid = svgEl._uid;

  const simulation = d3.forceSimulation(nodes)
    .force("link",      d3.forceLink(edges).id(d => d.id).distance(65).strength(0.45))
    .force("charge",    d3.forceManyBody().strength(-100))
    .force("center",    d3.forceCenter(W / 2, H / 2))
    .force("collision", d3.forceCollide(d => 10 + d.members.length * 3));

  const maxW = Math.max(...edges.map(e => e.weight), 1e-8);

  const link = svg.append("g").selectAll("line")
    .data(edges).enter().append("line")
    .attr("stroke-width", d => Math.min(4, 1.5 + (d.weight / maxW) * 3.5))
    .attr("stroke-opacity", d => 0.3 + (d.weight / maxW) * 0.6)
    .attr("stroke", d => {
      if (d.weight / maxW < 0.1) return "#D1D5DB";
      return d.excitatory ? "#3B82F6" : "#EF4444";
    })
    .attr("marker-end", d => {
      const t = d.weight / maxW < 0.1 ? "weak" : d.excitatory ? "exc" : "inh";
      return `url(#arrow-${t}-${uid})`;
    });

  const node = svg.append("g").selectAll("g")
    .data(nodes).enter().append("g")
    .attr("cursor", "pointer")
    .call(d3.drag()
      .on("start", (ev, d) => { if (!ev.active) simulation.alphaTarget(0.3).restart(); d.fx = d.x; d.fy = d.y; })
      .on("drag",  (ev, d) => { d.fx = ev.x; d.fy = ev.y; })
      .on("end",   (ev, d) => { if (!ev.active) simulation.alphaTarget(0); d.fx = null; d.fy = null; }));

  const r = d => d.is_target ? 12 : (d.members.length > 1 ? 9 + d.members.length * 1.2 : 7);

  node.append("circle")
    .attr("r", r)
    .attr("fill", d => d.color)
    .attr("stroke", d => d.is_target ? "#1a1f2e" : "white")
    .attr("stroke-width", d => d.is_target ? 2.5 : 1.5)
    .attr("stroke-dasharray", d => d.members.length > 1 && !d.is_target ? "4,2" : "none");

  node.append("text")
    .attr("dy", "0.32em").attr("text-anchor", "middle")
    .style("font-size", d => d.members.length > 1 ? "7px" : "6.5px")
    .style("font-family", "JetBrains Mono, monospace")
    .style("fill", "white").style("pointer-events", "none")
    .text(d => d.members.length > 1 ? `×${d.members.length}` : `F${d.id}`);

  node.append("title")
    .text(d => d.members.length > 1
      ? `Supernode: F${d.members.join(", F")} (${d.members.length} features)\nstr=${d.strength.toFixed(2)}`
      : `F${d.id}  strength=${d.strength.toFixed(2)}`);

  simulation.on("tick", () => {
    link.attr("x1", d => d.source.x).attr("y1", d => d.source.y)
        .attr("x2", d => d.target.x).attr("y2", d => d.target.y);
    node.attr("transform", d => `translate(${d.x},${d.y})`);
  });
}

// ─── Main panel renderer ──────────────────────────────────────────────────────

function renderMainPanel(key) {
  const featureData = DATA[key];
  if (!featureData) return;

  renderFeatureHeader(key, featureData);
  renderHistogram(featureData);
  renderUniversalityProfile(featureData);
  renderTopPatches(key, featureData);
  renderCoactivationTable(featureData);
  renderCircuitGraph(key, featureData);

  // Update URL hash for bookmarking
  history.replaceState(null, "", `#f${key}`);
}

// ─── Bootstrap ────────────────────────────────────────────────────────────────

document.addEventListener("DOMContentLoaded", function () {
  const allKeys = Object.keys(DATA).sort((a, b) => +a - +b);

  if (allKeys.length === 0) {
    document.body.innerHTML = "<p style='padding:40px;color:#888'>No feature data found.</p>";
    return;
  }

  buildSidebar(allKeys);

  // Honour URL hash or fall back to START_KEY
  let startKey = String(START_KEY);
  const hash = window.location.hash.replace(/^#f/, "");
  if (hash && DATA[hash]) startKey = hash;

  setActiveItem(startKey);
  renderMainPanel(startKey);

  // Scroll active sidebar item into view
  const activeItem = document.querySelector(".feature-item.active");
  if (activeItem) activeItem.scrollIntoView({ block: "nearest" });
});
