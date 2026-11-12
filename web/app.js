const scene = {
  attempts: 8,
  traffic: 600,
  probability: null,
  playing: !window.matchMedia("(prefers-reduced-motion: reduce)").matches,
  speed: 1,
  nodes: [],
  edges: [],
  particles: [],
  width: 0,
  height: 0,
  replay: null,
};

const canvas = document.querySelector("#net");
const ctx = canvas.getContext("2d");
const tooltip = document.querySelector("#tooltip");
let lastFrame = performance.now();
let seenRevision = -1;
let polling = false;
let bootstrapped = false;
let modelReady = false;

function resize() {
  const wrap = canvas.parentElement;
  const rect = wrap.getBoundingClientRect();
  if (rect.width < 20 || rect.height < 20) return;
  const dpr = Math.min(window.devicePixelRatio || 1, 2);
  canvas.width = Math.floor(rect.width * dpr);
  canvas.height = Math.floor(rect.height * dpr);
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  scene.width = rect.width;
  scene.height = rect.height;
  layoutNetwork();
}

function layoutNetwork() {
  const w = scene.width;
  const h = scene.height;
  const xs = [w * 0.1, w * 0.36, w * 0.64, w * 0.9];
  const counts = [2, 8, 4, 1];
  const names = ["Entrada", "Oculta · ReLU", "Oculta · ReLU", "Salida · sigmoide"];
  scene.nodes = counts.map((count, layer) => {
    const inset = count === 1 ? 0.5 : count === 2 ? 0.3 : count === 4 ? 0.2 : 0.08;
    const top = h * inset;
    const span = count === 1 ? 0 : h * (1 - inset * 2);
    const gap = count === 1 ? 0 : span / (count - 1);
    return Array.from({ length: count }, (_, index) => ({
      x: xs[layer],
      y: top + gap * index,
      layer,
      index,
      name: names[layer],
      r: layer === 3 ? 15 : layer === 0 ? 12 : 8,
    }));
  });
  scene.edges = [];
  for (let layer = 0; layer < 3; layer += 1) {
    scene.nodes[layer].forEach((from) => {
      scene.nodes[layer + 1].forEach((to) => {
        const mid = (from.x + to.x) / 2;
        const lift = (from.index - to.index) * 6;
        scene.edges.push({
          from,
          to,
          weight: null,
          c1: { x: mid, y: from.y + lift },
          c2: { x: mid, y: to.y - lift * 0.4 },
        });
      });
    });
  }
  if (!scene.particles.length) {
    scene.particles = Array.from({ length: 42 }, () => ({
      edge: Math.floor(Math.random() * scene.edges.length),
      t: Math.random(),
      speed: 0.00018 + Math.random() * 0.00035,
    }));
  } else {
    scene.particles.forEach((particle) => {
      if (particle.edge >= scene.edges.length) particle.edge = 0;
    });
  }
  stampWeights();
}

let weightMatrices = null;

function stampWeights() {
  if (!weightMatrices) return;
  scene.edges.forEach((edge) => {
    const matrix = weightMatrices[edge.from.layer];
    if (matrix && matrix[edge.from.index]) edge.weight = matrix[edge.from.index][edge.to.index];
  });
}

function signedColor(weight, alpha) {
  if (typeof weight !== "number") return `rgba(158, 186, 214, ${alpha * 0.45})`;
  const [red, green, blue] = weight >= 0 ? [61, 139, 255] : [255, 138, 48];
  return `rgba(${red}, ${green}, ${blue}, ${alpha})`;
}

function pointOnEdge(edge, t) {
  const u = 1 - t;
  const { from, to, c1, c2 } = edge;
  return {
    x: u ** 3 * from.x + 3 * u ** 2 * t * c1.x + 3 * u * t ** 2 * c2.x + t ** 3 * to.x,
    y: u ** 3 * from.y + 3 * u ** 2 * t * c1.y + 3 * u * t ** 2 * c2.y + t ** 3 * to.y,
  };
}

const PASS_BEATS = [1400, 1200, 1800, 1700, 1600, 1500];

function easeOut(value) {
  return 1 - (1 - Math.min(1, Math.max(0, value))) ** 3;
}

function formatNumber(value, digits = 2) {
  return Number(value).toLocaleString("es-CO", {
    minimumFractionDigits: digits,
    maximumFractionDigits: digits,
  });
}

function replayClock(now) {
  const replay = scene.replay;
  if (!replay) return null;
  if (replay.settled) return { index: 5, local: 1, finished: true };
  const elapsed = (now - replay.started) * scene.speed;
  let cursor = 0;
  const total = PASS_BEATS.reduce((sum, beat) => sum + beat, 0);
  for (let index = 0; index < PASS_BEATS.length; index += 1) {
    if (elapsed < cursor + PASS_BEATS[index]) {
      return { index, local: (elapsed - cursor) / PASS_BEATS[index], finished: false, progress: elapsed / total };
    }
    cursor += PASS_BEATS[index];
  }
  return { index: 5, local: 1, finished: true, progress: 1 };
}

function layerValues(node) {
  const replay = scene.replay;
  if (!replay?.trace) return null;
  if (node.layer === 0) {
    return [replay.payload.failed_attempts / 20, replay.payload.traffic_mb / 2000];
  }
  return replay.trace.layers?.[node.layer - 1]?.values || null;
}

function layerOpen(layer, phase) {
  return phase >= [0, 2, 3, 4][layer];
}

function activation(node) {
  const clock = scene.replay?.clock;
  if (scene.replay && clock) {
    if (!scene.replay.settled && !layerOpen(node.layer, clock.index)) return 0.04;
    const values = layerValues(node);
    if (!values) return 0.2;
    const raw = Math.abs(values[node.index] || 0);
    const peak = Math.max(1, ...values.map((value) => Math.abs(value)));
    const level = Math.min(1, raw / peak);
    const opens = [0, 2, 3, 4][node.layer];
    if (!scene.replay.settled && clock.index === opens) return 0.05 + level * easeOut(clock.local);
    return 0.1 + level * 0.9;
  }
  if (node.layer === 0) {
    return node.index === 0 ? scene.attempts / 20 : scene.traffic / 2000;
  }
  if (node.layer === 3 && scene.probability != null) return scene.probability;
  return 0.45 + 0.2 * Math.sin(performance.now() / 900 + node.index);
}

function drawAtmosphere(now) {
  const { width, height } = scene;
  const glow = ctx.createRadialGradient(width * 0.48, height * 0.48, 20, width * 0.5, height * 0.5, width * 0.42);
  glow.addColorStop(0, "rgba(46, 231, 255, 0.07)");
  glow.addColorStop(1, "rgba(46, 231, 255, 0)");
  ctx.fillStyle = glow;
  ctx.fillRect(0, 0, width, height);
  for (let band = 0; band < 4; band += 1) {
    ctx.beginPath();
    const base = height * (0.72 + band * 0.06);
    for (let x = 0; x <= width; x += 7) {
      const y = base + Math.sin(x * 0.011 + now / 1700 + band) * (7 + band * 3);
      if (x === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    }
    ctx.strokeStyle = `rgba(110, 200, 255, ${0.09 - band * 0.015})`;
    ctx.lineWidth = 1;
    ctx.stroke();
  }
}

function shownProbability() {
  const clock = scene.replay?.clock;
  if (scene.replay && !scene.replay.settled && clock && clock.index < 4) return null;
  return scene.replay?.payload?.probability ?? scene.probability;
}

function drawBeam(phase) {
  const focus = [0, 0, 1, 2, 3, 3][phase];
  const x = scene.nodes[focus]?.[0]?.x;
  if (x == null) return;
  const beam = ctx.createLinearGradient(x - 36, 0, x + 36, 0);
  beam.addColorStop(0, "rgba(46, 231, 255, 0)");
  beam.addColorStop(0.5, "rgba(46, 231, 255, 0.16)");
  beam.addColorStop(1, "rgba(46, 231, 255, 0)");
  ctx.fillStyle = beam;
  ctx.fillRect(x - 36, 0, 72, scene.height);
}

function drawSignalLabels() {
  const replay = scene.replay;
  if (!replay?.clock) return;
  const phase = replay.clock.index;
  ctx.font = "12px Segoe UI, sans-serif";
  ctx.textBaseline = "middle";
  scene.nodes.flat().forEach((node) => {
    if (!layerOpen(node.layer, phase) && !replay.settled) return;
    const values = layerValues(node);
    if (!values) return;
    let label = "";
    if (node.layer === 0) {
      label = node.index === 0 ? `${replay.payload.failed_attempts}` : `${Math.round(replay.payload.traffic_mb)} MB`;
      ctx.textAlign = "right";
      ctx.fillStyle = "rgba(231, 242, 255, 0.92)";
      ctx.fillText(label, node.x - node.r - 8, node.y);
      return;
    }
    const value = values[node.index] || 0;
    if (node.layer === 3) {
      label = `${formatNumber(value * 100, 1)} %`;
      ctx.textAlign = "center";
      ctx.fillStyle = value >= (replay.payload.threshold || 0.5) ? "#f5b942" : "#3ee0a0";
      ctx.fillText(label, node.x, node.y - node.r - 14);
      return;
    }
    if (value <= 0.01) return;
    label = formatNumber(value, 2);
    ctx.textAlign = "left";
    ctx.fillStyle = "rgba(231, 242, 255, 0.8)";
    ctx.fillText(label, node.x + node.r + 6, node.y);
  });
}

function drawNetwork(now) {
  const { width, height } = scene;
  ctx.clearRect(0, 0, width, height);
  drawAtmosphere(now);
  const live = scene.replay && !scene.replay.settled && scene.replay.clock;
  if (live) drawBeam(scene.replay.clock.index);
  const wave = live ? 0 : Math.sin(now / 1400);
  const hop = live ? [null, null, 0, 1, 2, 2][scene.replay.clock.index] : null;
  scene.edges.forEach((edge) => {
    const current = hop === edge.from.layer;
    const open = !live || scene.replay.clock.index >= [2, 3, 4][edge.from.layer];
    const glow = open ? (current ? 0.9 : 0.42 + activation(edge.from) * 0.4) : 0.16;
    const strength = Math.min(1, Math.abs(edge.weight || 0));
    ctx.beginPath();
    ctx.moveTo(edge.from.x, edge.from.y);
    ctx.bezierCurveTo(edge.c1.x, edge.c1.y + wave * 4, edge.c2.x, edge.c2.y, edge.to.x, edge.to.y);
    ctx.strokeStyle = signedColor(edge.weight, glow);
    ctx.lineWidth = (current ? 2.6 : 1.5) + strength;
    ctx.stroke();
  });
  const probability = shownProbability();
  scene.particles.forEach((particle) => {
    const edge = scene.edges[particle.edge];
    if (!edge) return;
    if (live && hop !== edge.from.layer && hop !== null) return;
    if (live && hop === null) return;
    const point = pointOnEdge(edge, particle.t % 1);
    ctx.beginPath();
    ctx.fillStyle = edge.weight < 0 ? "#ffb27a" : "#8eb8ff";
    ctx.arc(point.x, point.y, live ? 3 : 2.1, 0, Math.PI * 2);
    ctx.fill();
  });
  if (live && hop === null) {
    scene.nodes[0].forEach((node, index) => {
      const travel = (now / 700 + index * 0.35) % 1;
      ctx.beginPath();
      ctx.fillStyle = "#bff8ff";
      ctx.arc(node.x - 70 * (1 - travel), node.y, 2.4, 0, Math.PI * 2);
      ctx.fill();
    });
  }
  scene.nodes.flat().forEach((node) => {
    const level = activation(node);
    const color = node.layer === 3 && probability != null
      ? (probability >= 0.5 ? "#f5b942" : "#3ee0a0")
      : (node.layer === 1 ? "#9d8cff" : "#7eecff");
    ctx.beginPath();
    ctx.fillStyle = color;
    ctx.globalAlpha = 0.16 + level * 0.34;
    ctx.arc(node.x, node.y, node.r + 7 + level * 6, 0, Math.PI * 2);
    ctx.fill();
    ctx.globalAlpha = live && !layerOpen(node.layer, scene.replay.clock.index) ? 0.28 : 1;
    ctx.beginPath();
    ctx.fillStyle = color;
    ctx.arc(node.x, node.y, node.r, 0, Math.PI * 2);
    ctx.fill();
    ctx.globalAlpha = 1;
    ctx.beginPath();
    ctx.strokeStyle = "rgba(255,255,255,0.8)";
    ctx.lineWidth = 1.2;
    ctx.arc(node.x, node.y, node.r * 0.45, 0, Math.PI * 2);
    ctx.stroke();
  });
  if (scene.replay) drawSignalLabels();
}

function frame(now) {
  const elapsed = now - lastFrame;
  lastFrame = now;
  if (scene.replay && !scene.replay.settled) {
    scene.replay.clock = replayClock(now);
    paintPass(scene.replay.clock);
    if (scene.replay.clock.finished) settleReplay();
  }
  const live = scene.replay && !scene.replay.settled && scene.replay.clock;
  const hop = live ? [null, null, 0, 1, 2, 2][scene.replay.clock.index] : null;
  if ((scene.playing || live) && scene.edges.length) {
    const hopEdges = hop == null ? [] : scene.edges.reduce((indexes, edge, index) => {
      if (edge.from.layer === hop) indexes.push(index);
      return indexes;
    }, []);
    scene.particles.forEach((particle, index) => {
      if (live && hopEdges.length) particle.edge = hopEdges[index % hopEdges.length];
      particle.t += particle.speed * scene.speed * elapsed * (live ? 2.4 : 1);
      if (particle.t > 1) {
        particle.t %= 1;
        if (!live) particle.edge = Math.floor(Math.random() * scene.edges.length);
      }
    });
  }
  if (scene.width) drawNetwork(now);
  requestAnimationFrame(frame);
}

function showView(name) {
  document.querySelectorAll(".view").forEach((view) => {
    view.hidden = view.id !== `view-${name}`;
  });
  document.querySelectorAll(".nav button").forEach((button) => {
    button.classList.toggle("is-on", button.dataset.view === name);
  });
  if (name === "analizar") requestAnimationFrame(resize);
  if (name !== "guia") stopTutorial();
}

function setStep(step) {
  document.querySelectorAll(".stepper button").forEach((button) => {
    button.classList.toggle("is-on", button.dataset.step === String(step));
  });
  if (String(step) === "3") {
    showView("resultados");
    return;
  }
  showView("analizar");
  document.querySelector("#panel-train").hidden = String(step) !== "1";
  document.querySelector("#panel-analyze").hidden = String(step) !== "2";
}

function toast(message) {
  const node = document.querySelector("#toast");
  node.textContent = message;
  node.hidden = false;
  clearTimeout(toast.timer);
  toast.timer = setTimeout(() => { node.hidden = true; }, 6000);
}

function applyState(state) {
  document.querySelector("#banner").textContent = state.academic_notice;
  const title = document.querySelector("#status-title");
  const detail = document.querySelector("#status-detail");
  const dot = document.querySelector(".dot");
  const button = document.querySelector("#train-button");
  const warn = document.querySelector("#train-warn");
  const progress = document.querySelector("#train-progress");
  const chip = document.querySelector("#stage-chip");
  if (state.loading) {
    title.textContent = "Preparando";
    detail.textContent = "Cargando la red neuronal…";
    dot.className = "dot wait";
    button.disabled = true;
  } else if (state.training) {
    title.textContent = "Entrenando";
    const progressState = state.progress || {};
    const loss = progressState.loss == null ? "" : ` · pérdida ${progressState.loss.toFixed(4)}`;
    detail.textContent = progressState.epoch
      ? `Época ${progressState.epoch} de ${progressState.total}${loss}`
      : "Generando las conexiones…";
    dot.className = "dot wait";
    button.disabled = true;
    button.querySelector("span").textContent = "Entrenando…";
    warn.hidden = true;
    progress.hidden = false;
    document.querySelector("#train-fill").style.width = `${Math.round((progressState.fraction || 0) * 100)}%`;
    document.querySelector("#train-detail").textContent = detail.textContent;
  } else if (state.ready) {
    title.textContent = "Modelo listo";
    detail.textContent = state.source === "archivo" ? "Abierto desde el archivo guardado." : "Entrenado en esta sesión.";
    dot.className = "dot ready";
    button.disabled = false;
    button.querySelector("span").textContent = "Entrenar de nuevo";
    warn.hidden = true;
    progress.hidden = true;
    if (!scene.replay) chip.textContent = "Modelo activo";
  } else {
    title.textContent = "Sin modelo";
    detail.textContent = "Entrénela para clasificar conexiones.";
    dot.className = "dot";
    button.disabled = false;
    button.querySelector("span").textContent = "Entrenar red";
    warn.hidden = false;
    progress.hidden = true;
    chip.textContent = "Vista previa";
  }
  const error = document.querySelector("#train-error");
  error.hidden = !state.error || state.training;
  error.textContent = state.error || "";
  modelReady = Boolean(state.ready && !state.training);
  if (state.weights) {
    weightMatrices = state.weights;
    stampWeights();
  }
  if (pendingCustom && state.ready && !state.training) {
    const custom = pendingCustom;
    pendingCustom = null;
    attempts.value = custom.attempts;
    traffic.value = custom.traffic;
    syncInputs();
    setStep("2");
    predict(custom.attempts, custom.traffic);
  }
  if (state.revision !== seenRevision) {
    renderResults(state.report);
    const previous = seenRevision;
    seenRevision = state.revision;
    if (previous >= 0 && state.message) {
      toast(state.message);
      setStep("2");
    }
  }
  if (!bootstrapped && !state.loading) {
    bootstrapped = true;
    if (!location.hash && state.ready) setStep("2");
  }
  if (state.training && !polling) {
    polling = true;
    window.setTimeout(pollTraining, 400);
  }
}

async function refresh() {
  const response = await fetch("/api/state");
  applyState(await response.json());
}

async function pollTraining() {
  const response = await fetch("/api/state");
  const state = await response.json();
  applyState(state);
  if (state.training) window.setTimeout(pollTraining, 400);
  else polling = false;
}

function renderResults(report) {
  const root = document.querySelector("#results-root");
  if (!report) {
    root.innerHTML = "<h2>Resultados</h2><p class='muted'>Después de entrenar verá aquí el reparto de los datos, la pérdida, la exactitud, la precisión, el recall y las curvas.</p>";
    return;
  }
  const metrics = report.metrics || {};
  const baseline = report.baseline || {};
  const dataset = report.dataset || {};
  const cards = [
    ["Exactitud", metrics.accuracy, baseline.accuracy],
    ["Precisión", metrics.precision, baseline.precision],
    ["Recall", metrics.recall, baseline.recall],
    ["F1", metrics.f1, baseline.f1],
  ].map(([label, value, base]) => {
    const delta = value != null && base != null ? `<em>${value - base >= 0 ? "+" : ""}${((value - base) * 100).toFixed(2)}%</em>` : "";
    return `<article class="metric"><span>${label}</span><strong>${pct(value)}</strong>${delta}</article>`;
  }).join("");
  root.innerHTML = `
    <h2>Resultados de la prueba</h2>
    <p class="muted">${grouped(dataset.total || 0)} conexiones simuladas · ${grouped(report.test_count || 0)} apartadas para la prueba · semilla 42. La variación se mide contra la clase más frecuente.</p>
    <div class="metrics">${cards}</div>
    <p class="muted">Pérdida ${num(metrics.loss)} · umbral ${pct(metrics.threshold)} · ${grouped(metrics.normal_count || 0)} normales y ${grouped(metrics.suspicious_count || 0)} sospechosas en la prueba.</p>
    <div class="cm-wrap">
      <div><h3>Red neuronal</h3>${matrix(metrics.confusion_matrix)}</div>
      <div><h3>Referencia mayoritaria</h3>${matrix(baseline.confusion_matrix)}</div>
    </div>
    <div class="charts">
      ${chart("Pérdida", report.history?.loss, report.history?.val_loss)}
      ${chart("Exactitud", report.history?.accuracy, report.history?.val_accuracy, true)}
    </div>
    <p class="muted">La época ${report.best_epoch} tuvo la menor pérdida de validación. Esos pesos son los que quedaron. Una exactitud alta reconoce el patrón simulado; no detecta ataques reales.</p>
  `;
}

function matrix(value) {
  if (!value) return "<p class='muted'>Sin matriz.</p>";
  const [tn, fp] = value[0];
  const [fn, tp] = value[1];
  return `<div class="cm">
    <span></span><span>Predicha normal</span><span>Predicha sospechosa</span>
    <span>Real normal</span><div class="cell hit"><strong>${tn}</strong>correcta</div><div class="cell miss"><strong>${fp}</strong>falsa sospecha</div>
    <span>Real sospechosa</span><div class="cell miss"><strong>${fn}</strong>no vista</div><div class="cell hit"><strong>${tp}</strong>detectada</div>
  </div>`;
}

function chart(title, train, validation, clamp) {
  if (!train?.length) return "";
  const series = [train, validation].filter(Boolean);
  const max = clamp ? 1 : Math.max(...series.flat());
  const min = clamp ? 0 : Math.min(...series.flat());
  const line = (values, color) => {
    const d = values.map((value, index) => {
      const x = 20 + (index / Math.max(values.length - 1, 1)) * 360;
      const y = 150 - ((value - min) / (max - min || 1)) * 120;
      return `${index ? "L" : "M"}${x.toFixed(1)},${y.toFixed(1)}`;
    }).join(" ");
    return `<path d="${d}" fill="none" stroke="${color}" stroke-width="2"/>`;
  };
  return `<div class="chart"><strong>${title}</strong><svg viewBox="0 0 400 180">${line(train, "#2ee7ff")}${validation ? line(validation, "#8d7bff") : ""}</svg></div>`;
}

function pct(value) {
  return value == null ? "—" : `${(value * 100).toFixed(2)}%`;
}
function num(value) {
  return value == null ? "—" : Number(value).toFixed(4);
}
function grouped(value) {
  return String(value).replace(/\B(?=(\d{3})+(?!\d))/g, ".");
}

document.querySelectorAll(".nav button").forEach((button) => {
  button.addEventListener("click", () => {
    if (button.dataset.view === "resultados") setStep("3");
    else if (button.dataset.view === "analizar") setStep(modelReady ? "2" : "1");
    else showView("guia");
  });
});
document.querySelectorAll(".stepper button").forEach((button) => {
  button.addEventListener("click", () => setStep(button.dataset.step));
});

document.querySelector("#toggle-flow").addEventListener("click", (event) => {
  scene.playing = !scene.playing;
  event.currentTarget.setAttribute("aria-pressed", String(scene.playing));
  event.currentTarget.title = scene.playing ? "Pausar el flujo" : "Reanudar el flujo";
  document.querySelector("#icon-pause").hidden = !scene.playing;
  document.querySelector("#icon-play").hidden = scene.playing;
});
document.querySelector("#speed").addEventListener("change", (event) => {
  scene.speed = Number(event.target.value);
});

document.querySelector("#panel-train").addEventListener("submit", async (event) => {
  event.preventDefault();
  const button = document.querySelector("#train-button");
  button.disabled = true;
  const response = await fetch("/api/train", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      epochs: Number(document.querySelector("#epochs").value),
      save: document.querySelector("#save-model").checked,
    }),
  });
  const payload = await response.json();
  if (!response.ok) {
    document.querySelector("#train-error").hidden = false;
    document.querySelector("#train-error").textContent = payload.error;
    button.disabled = false;
    return;
  }
  pollTraining();
});

const attempts = document.querySelector("#attempts");
const traffic = document.querySelector("#traffic");
const attemptsNum = document.querySelector("#attempts-num");
const trafficNum = document.querySelector("#traffic-num");

function clampAttempts(value) {
  const number = Math.round(Number(value));
  if (!Number.isFinite(number)) return 0;
  return Math.min(20, Math.max(0, number));
}

function clampTraffic(value) {
  const number = Math.round(Number(value));
  if (!Number.isFinite(number)) return 0;
  return Math.min(2000, Math.max(0, number));
}

function syncInputs() {
  scene.attempts = Number(attempts.value);
  scene.traffic = Number(traffic.value);
  if (document.activeElement !== attemptsNum) attemptsNum.value = attempts.value;
  if (document.activeElement !== trafficNum) trafficNum.value = traffic.value;
}

function bindTyped(typed, slider, clamp) {
  typed.addEventListener("input", () => {
    if (typed.value === "") return;
    slider.value = clamp(typed.value);
    syncInputs();
  });
  typed.addEventListener("change", () => {
    const next = clamp(typed.value);
    typed.value = next;
    slider.value = next;
    syncInputs();
  });
}

attempts.addEventListener("input", syncInputs);
traffic.addEventListener("input", syncInputs);
bindTyped(attemptsNum, attempts, clampAttempts);
bindTyped(trafficNum, traffic, clampTraffic);

function customFromQuery() {
  const params = new URLSearchParams(location.search);
  if (!params.has("intentos") && !params.has("trafico")) return null;
  return {
    attempts: clampAttempts(params.get("intentos") ?? attempts.value),
    traffic: clampTraffic(params.get("trafico") ?? traffic.value),
  };
}

let pendingCustom = customFromQuery();

document.querySelector("#panel-analyze").addEventListener("submit", async (event) => {
  event.preventDefault();
  await predict(Number(attempts.value), Number(traffic.value));
});
document.querySelectorAll(".examples button").forEach((button) => {
  button.addEventListener("click", async () => {
    attempts.value = button.dataset.attempts;
    traffic.value = button.dataset.traffic;
    syncInputs();
    await predict(Number(button.dataset.attempts), Number(button.dataset.traffic));
  });
});

function setAnalyzeBusy(busy) {
  const button = document.querySelector("#panel-analyze .go");
  button.disabled = busy;
  button.textContent = busy ? "Recorriendo la red…" : "Analizar la conexión";
  document.querySelectorAll(".examples button").forEach((item) => { item.disabled = busy; });
}

function litCount(values) {
  return (values || []).filter((value) => value > 0.01).length;
}

function beatCopy(replay, index) {
  const payload = replay.payload;
  const scaled = replay.trace?.scaled;
  const hidden = replay.trace?.layers || [];
  const percent = formatNumber(payload.probability * 100, 1);
  const threshold = formatNumber(payload.threshold * 100, 0);
  const beats = [
    {
      kicker: "01 · Entrada",
      title: "Entran los dos números",
      text: `${payload.failed_attempts} intentos fallidos y ${Math.round(payload.traffic_mb)} MB entran juntos a la red.`,
    },
    {
      kicker: "02 · Normalizar",
      title: "Se ponen en la misma escala",
      text: scaled
        ? `El escalador del entrenamiento los deja en ${formatNumber(scaled[0])} y ${formatNumber(scaled[1])}. Así el tráfico no se impone solo por ser un número más grande.`
        : "El escalador del entrenamiento pone los dos números en la misma escala.",
    },
    {
      kicker: "03 · ReLU",
      title: "Ocho neuronas buscan el patrón",
      text: hidden[0]
        ? `Cada neurona suma pesos por entradas más el sesgo. ReLU apaga lo negativo: ${litCount(hidden[0].values)} de 8 siguen encendidas.`
        : "La señal cruza la primera capa oculta y ReLU deja pasar lo positivo.",
    },
    {
      kicker: "04 · ReLU",
      title: "Cuatro neuronas afinan la señal",
      text: hidden[1]
        ? `Esas ocho salidas se mezclan aquí. ${litCount(hidden[1].values)} de 4 quedan con activación.`
        : "La segunda capa mezcla lo que la primera dejó pasar.",
    },
    {
      kicker: "05 · Sigmoide",
      title: "Sale una probabilidad",
      text: `La última cuenta pasa por la sigmoide y se convierte en ${percent} %.`,
    },
    {
      kicker: "06 · Umbral",
      title: payload.label ? "Queda como sospechosa" : "Queda como normal",
      text: `${percent} % ${payload.label ? "llega al" : "queda por debajo del"} umbral del ${threshold} %.`,
    },
  ];
  return beats[index];
}

function paintPass(clock) {
  const replay = scene.replay;
  if (!replay || replay.painted === clock.index) return;
  replay.painted = clock.index;
  const beat = beatCopy(replay, clock.index);
  document.querySelector("#pass").hidden = false;
  document.querySelector("#film-corners").hidden = false;
  document.querySelector("#pass-kicker").textContent = beat.kicker;
  document.querySelector("#pass-title").textContent = beat.title;
  document.querySelector("#pass-text").textContent = beat.text;
  document.querySelectorAll("#pass-beats li").forEach((item, index) => {
    item.classList.toggle("is-on", index === clock.index);
    item.classList.toggle("is-done", index < clock.index);
  });
}

function fillResult(payload) {
  scene.probability = payload.probability;
  const card = document.querySelector("#result");
  card.hidden = false;
  card.classList.remove("is-reveal");
  void card.offsetWidth;
  card.classList.add("is-reveal");
  card.classList.toggle("is-suspicious", payload.label === 1);
  document.querySelector("#result-class").textContent = payload.label ? "Sospechosa" : "Normal";
  document.querySelector("#result-prob").textContent = `${formatNumber(payload.probability * 100, 1)}%`;
  document.querySelector("#result-fill").style.width = `${payload.probability * 100}%`;
  document.querySelector("#result-mark").style.left = `${payload.threshold * 100}%`;
  document.querySelector("#result-threshold").textContent = `Umbral utilizado: ${formatNumber(payload.threshold * 100, 0)}%.`;
  document.querySelector("#result-values").textContent = `Valores introducidos: ${payload.failed_attempts} intentos · ${Math.round(payload.traffic_mb)} MB.`;
  document.querySelector("#result-explain").textContent = `Explicación: ${payload.explanation}`;
}

function settleReplay() {
  const replay = scene.replay;
  if (!replay || replay.settled) return;
  replay.settled = true;
  replay.clock = { index: 5, local: 1, finished: true };
  paintPass(replay.clock);
  fillResult(replay.payload);
  setAnalyzeBusy(false);
  document.querySelector("#stage-chip").textContent = "Clasificación lista";
  clearTimeout(settleReplay.timer);
  settleReplay.timer = setTimeout(() => {
    if (scene.replay === replay) {
      document.querySelector("#pass").hidden = true;
      document.querySelector("#film-corners").hidden = true;
    }
  }, 2200);
}

function startReplay(payload) {
  const reduced = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
  scene.replay = {
    payload,
    trace: payload.trace || null,
    started: performance.now(),
    painted: -1,
    settled: false,
    clock: { index: 0, local: 0, finished: false },
  };
  document.querySelector("#result").hidden = true;
  document.querySelector("#stage-chip").textContent = "Recorrido en vivo";
  if (reduced) {
    scene.replay.settled = true;
    scene.replay.clock = { index: 5, local: 1, finished: true };
    fillResult(payload);
    setAnalyzeBusy(false);
    document.querySelector("#stage-chip").textContent = "Clasificación lista";
    return;
  }
  paintPass(scene.replay.clock);
}

async function predict(failedAttempts, trafficMb) {
  const error = document.querySelector("#predict-error");
  error.hidden = true;
  setAnalyzeBusy(true);
  const response = await fetch("/api/predict", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ failed_attempts: failedAttempts, traffic_mb: trafficMb }),
  });
  const payload = await response.json();
  if (!response.ok) {
    error.hidden = false;
    error.textContent = payload.error;
    setAnalyzeBusy(false);
    return;
  }
  startReplay(payload);
}

canvas.addEventListener("mousemove", (event) => {
  const rect = canvas.getBoundingClientRect();
  const x = event.clientX - rect.left;
  const y = event.clientY - rect.top;
  const node = scene.nodes.flat().find((item) => Math.hypot(item.x - x, item.y - y) < item.r + 8);
  tooltip.hidden = !node;
  if (!node) return;
  const values = layerValues(node);
  const extra = values ? ` · ${formatNumber(values[node.index], 2)}` : "";
  tooltip.textContent = `${node.name} · neurona ${node.index + 1}${extra}`;
  tooltip.style.left = `${node.x + 12}px`;
  tooltip.style.top = `${node.y - 10}px`;
});
canvas.addEventListener("mouseleave", () => { tooltip.hidden = true; });

window.addEventListener("resize", resize);
resize();
requestAnimationFrame(frame);
refresh();
window.setInterval(refresh, 4000);

function openFromHash() {
  if (location.hash === "#guia") showView("guia");
  else if (location.hash === "#resultados") setStep("3");
}
openFromHash();
window.addEventListener("hashchange", openFromHash);

const tutorialSteps = [
  {
    title: "Qué hace esta red",
    html: "<p>Clasifica una conexión <strong>simulada</strong> como normal, la clase 0, o sospechosa, la clase 1. El número de la pantalla lo calcula el modelo entrenado.</p>",
    speech: "Este tutorial recorre el experimento completo. La red tiene una sola tarea: mirar una conexión simulada y decir si se parece más a una conexión normal o a una sospechosa. Normal es la clase cero. Sospechosa es la clase uno. El número que ves no lo inventa la interfaz. Lo calcula el modelo después de entrenar.",
  },
  {
    title: "Las dos entradas",
    html: "<p><strong>x₁</strong> son los intentos fallidos de inicio de sesión, un entero de 0 a 20. <strong>x₂</strong> es el tráfico enviado, de 0 a 2.000 MB. La semilla 42 no es una entrada: solo permite repetir los mismos datos.</p>",
    speech: "Cada conexión entra con dos números, y solo con dos. Equís uno son los intentos fallidos de inicio de sesión, un entero entre cero y veinte. Equís dos es el tráfico enviado, entre cero y dos mil megabytes. La red no ve direcciones, puertos ni usuarios. La semilla cuarenta y dos no clasifica nada: si repites el experimento, vuelven a salir los mismos datos.",
  },
  {
    title: "De dónde sale la etiqueta",
    html: "<p>La etiqueta es la respuesta con la que la red aprende. No sale de un ataque real. El ruido evita que la frontera sea perfecta.</p><pre>puntaje = 0,55 × (intentos / 20) + 0,45 × (tráfico / 2000) + ruido\netiqueta = 1 si puntaje ≥ 0,55; si no, 0</pre>",
    speech: "La etiqueta es la respuesta correcta. No sale de un ataque real. Se calcula así: cero coma cincuenta y cinco por los intentos divididos entre veinte, más cero coma cuarenta y cinco por el tráfico dividido entre dos mil, más un poco de ruido. Si el puntaje llega a cero coma cincuenta y cinco, la etiqueta es uno, sospechosa. Si no, es cero, normal. El ruido existe para que la frontera no sea perfecta y la red tenga algo que aprender.",
  },
  {
    title: "La cuenta de una neurona",
    html: "<p>Antes de la cuenta, los datos se normalizan. Si no, el tráfico, que llega a 2.000, dominaría a los intentos, que solo llegan a 20. El escalador se ajusta solo con el entrenamiento.</p><pre>z = (w₁ × x₁) + (w₂ × x₂) + b</pre><p>El peso dice cuánto importa cada entrada. El sesgo desplaza la respuesta. Esos valores no vienen escritos a mano: el entrenamiento los ajusta.</p>",
    speech: "Antes de entrar, los dos números se normalizan. Si no se hiciera, el tráfico, que llega hasta dos mil, aplastaría a los intentos, que solo llegan a veinte. El normalizador se ajusta únicamente con el entrenamiento. Dentro de la neurona, zeta es el peso uno por equís uno, más el peso dos por equís dos, más el sesgo. El peso dice cuánto importa esa entrada. El sesgo desplaza la respuesta, incluso cuando las entradas son pequeñas. La red no trae esos números escritos: los ajusta al entrenar.",
  },
  {
    title: "ReLU, sigmoide y el umbral",
    html: "<pre>ReLU(z) = 0 si z &lt; 0;  z si z ≥ 0\np = 1 / (1 + e^(−z))</pre><p>ReLU, en las capas ocultas, apaga lo negativo y deja pasar lo positivo. Así aparecen relaciones que una recta no separa. La sigmoide de la salida convierte la cuenta en una probabilidad entre 0 y 1. Si llega a <strong>0,5</strong>, la clase es sospechosa.</p>",
    speech: "Las capas ocultas usan ReLU. Si zeta es negativo, ReLU lo convierte en cero. Si es positivo, lo deja pasar. Por eso la red puede separar patrones que una sola recta no separa. La salida usa la sigmoide: uno dividido entre uno más e elevado a menos zeta. El resultado es una probabilidad entre cero y uno. La probabilidad todavía no es la clase. El umbral del taller es cero coma cinco. Si la probabilidad llega a ese valor, la conexión se clasifica como sospechosa. Si queda por debajo, como normal.",
  },
  {
    title: "La arquitectura y los 65 parámetros",
    html: "<p>La red es <strong>2-8-4-1</strong>: 2 entradas, 8 neuronas ReLU, 4 neuronas ReLU y 1 salida sigmoide.</p><pre>entrada → oculta:  2 × 8 + 8 = 24\noculta → oculta:   8 × 4 + 4 = 36\noculta → salida:   4 × 1 + 1 =  5\ntotal                        65</pre>",
    speech: "La arquitectura es dos, ocho, cuatro, uno. Dos entradas, ocho neuronas ReLU, cuatro neuronas ReLU y una salida sigmoide. Hay sesenta y cinco parámetros entrenables. Se cuentan así. De la entrada a la primera capa oculta: dos por ocho pesos, más ocho sesgos, son veinticuatro. De esa capa a la siguiente: ocho por cuatro pesos, más cuatro sesgos, son treinta y seis. Y de la última capa oculta a la salida: cuatro pesos más un sesgo, son cinco. Veinticuatro más treinta y seis más cinco son sesenta y cinco.",
  },
  {
    title: "Dónde aprende",
    html: "<p>El aprendizaje ocurre en <code>modelo.fit()</code>: propagación hacia adelante, comparación con la etiqueta, pérdida y corrección de pesos y sesgos. El optimizador es Adam y la pérdida es la entropía cruzada binaria.</p><p>De las 2.000 conexiones, el <strong>64 %</strong> ajusta, el <strong>16 %</strong> vigila la validación y el <strong>20 %</strong> queda intacto para la prueba. Si la validación no mejora durante 12 épocas, el ajuste se detiene y se restauran los pesos de la mejor época.</p>",
    speech: "El aprendizaje ocurre en modelo punto fit. En cada época la red hace la propagación hacia adelante, compara su predicción con la etiqueta, calcula la pérdida y corrige pesos y sesgos hacia atrás. El optimizador es Adam y la pérdida es la entropía cruzada binaria. De las dos mil conexiones, el sesenta y cuatro por ciento ajusta los pesos, el dieciséis por ciento vigila la validación y el veinte por ciento queda intacto para el examen final. Si la pérdida de validación no mejora durante doce épocas, el entrenamiento se detiene y se restauran los pesos de la mejor época, no los de la última.",
  },
  {
    title: "Cómo leer el resultado",
    html: "<p>La exactitud es el porcentaje de aciertos. La precisión dice, de las marcadas como sospechosas, cuántas lo eran. El recall dice, de las sospechosas reales, cuántas detectó. Se comparan con una referencia que siempre predice la clase más frecuente.</p><p>Cada análisis llama a <code>ConnectionRiskService.predict</code>. El servicio valida el rango, normaliza con el escalador del entrenamiento y devuelve la probabilidad, el umbral, la clase y la explicación.</p><p>Una exactitud alta reconoce el patrón simulado. No detecta ataques reales. Demostración académica con datos simulados. Este resultado no confirma ni descarta un ataque informático real.</p>",
    speech: "En la prueba se leen la pérdida, la exactitud, la precisión y el recall. La exactitud es el porcentaje de aciertos. La precisión dice, de las que marcó como sospechosas, cuántas lo eran. El recall dice, de las sospechosas reales, cuántas detectó. Esas cifras se comparan con una referencia que siempre predice la clase más frecuente. La pantalla no recalcula el número por su cuenta: llama a Connection Risk Service punto predict. El servicio valida el rango, normaliza con el mismo escalador del entrenamiento y devuelve la probabilidad, el umbral, la clase y la explicación. Una exactitud alta significa que la red reconoció el patrón simulado. No significa que pueda detectar un ataque real. Esta es una demostración académica con datos simulados. El resultado no confirma ni descarta un ataque informático real.",
  },
];

const tutorial = { index: 0, playing: false, timer: 0, token: 0 };

function stopTutorial() {
  tutorial.playing = false;
  tutorial.token += 1;
  clearInterval(tutorial.timer);
  if (window.speechSynthesis) speechSynthesis.cancel();
  const play = document.querySelector("#tutorial-play");
  if (play && !play.disabled) play.textContent = tutorial.index === 0 ? "Empezar con voz" : "Escuchar este paso";
}

function maleVoice() {
  if (!window.speechSynthesis) return null;
  const voices = speechSynthesis.getVoices();
  const spanish = voices.filter((voice) => /^es([-_]|$)/i.test(voice.lang) || /spanish|español/i.test(voice.name));
  const male = /pablo|álvaro|alvaro|raul|raúl|jorge|diego|carlos|enrique|andrés|andres|miguel|juan|mateo|male|hombre/i;
  const female = /helena|sabina|laura|mónica|monica|paulina|elvira|dalia|female|mujer/i;
  return spanish.find((voice) => male.test(voice.name) && !female.test(voice.name))
    || spanish.find((voice) => !female.test(voice.name))
    || spanish[0]
    || null;
}

function voiceLabel() {
  const node = document.querySelector("#tutorial-voice");
  if (!window.speechSynthesis) {
    node.textContent = "Este navegador no ofrece lectura en voz alta. El texto del tutorial sigue disponible.";
    document.querySelector("#tutorial-play").disabled = true;
    return;
  }
  const voice = maleVoice();
  const male = voice && /pablo|álvaro|alvaro|raul|raúl|jorge|diego|carlos|enrique|andrés|andres|miguel|juan|mateo|male|hombre/i.test(voice.name);
  node.textContent = voice
    ? `Voz: ${voice.name}${male ? ", voz masculina en español." : "."}`
    : "Todavía no hay una voz en español. El texto se puede leer igual.";
}

function renderTutorial(index, speakNow) {
  tutorial.index = index;
  const step = tutorialSteps[index];
  document.querySelector("#tutorial-step").textContent = `Paso ${index + 1} de ${tutorialSteps.length}`;
  document.querySelector("#tutorial-title").textContent = step.title;
  document.querySelector("#tutorial-body").innerHTML = step.html;
  document.querySelector("#tutorial-prev").disabled = index === 0;
  document.querySelector("#tutorial-next").disabled = index === tutorialSteps.length - 1;
  document.querySelectorAll("#tutorial-index button").forEach((button, buttonIndex) => {
    button.classList.toggle("is-on", buttonIndex === index);
  });
  const play = document.querySelector("#tutorial-play");
  if (!tutorial.playing) play.textContent = index === 0 ? "Empezar con voz" : "Escuchar este paso";
  if (speakNow) speakTutorial(step.speech);
}

function speakTutorial(text) {
  if (!window.speechSynthesis) return;
  const token = tutorial.token + 1;
  tutorial.token = token;
  speechSynthesis.cancel();
  const utterance = new SpeechSynthesisUtterance(text);
  const voice = maleVoice();
  utterance.lang = voice?.lang || "es-ES";
  if (voice) utterance.voice = voice;
  utterance.rate = 0.96;
  utterance.onend = () => {
    if (token !== tutorial.token || !tutorial.playing) return;
    if (tutorial.index < tutorialSteps.length - 1) renderTutorial(tutorial.index + 1, true);
    else stopTutorial();
  };
  speechSynthesis.speak(utterance);
  clearInterval(tutorial.timer);
  tutorial.timer = window.setInterval(() => {
    if (!speechSynthesis.speaking) {
      clearInterval(tutorial.timer);
      return;
    }
    speechSynthesis.pause();
    speechSynthesis.resume();
  }, 8000);
}

function startTutorial() {
  if (!window.speechSynthesis) return;
  if (tutorial.playing) {
    stopTutorial();
    return;
  }
  tutorial.playing = true;
  document.querySelector("#tutorial-play").textContent = "Pausar la voz";
  const restart = tutorial.index === tutorialSteps.length - 1 && !speechSynthesis.speaking;
  renderTutorial(restart ? 0 : tutorial.index, true);
}

function initTutorial() {
  const list = document.querySelector("#tutorial-index");
  list.innerHTML = tutorialSteps.map((step, index) => `
    <li><button type="button" data-step="${index}"><small>Paso ${index + 1}</small>${step.title}</button></li>
  `).join("");
  list.querySelectorAll("button").forEach((button) => {
    button.addEventListener("click", () => renderTutorial(Number(button.dataset.step), tutorial.playing));
  });
  document.querySelector("#tutorial-prev").addEventListener("click", () => {
    if (tutorial.index > 0) renderTutorial(tutorial.index - 1, tutorial.playing);
  });
  document.querySelector("#tutorial-next").addEventListener("click", () => {
    if (tutorial.index < tutorialSteps.length - 1) renderTutorial(tutorial.index + 1, tutorial.playing);
  });
  document.querySelector("#tutorial-play").addEventListener("click", startTutorial);
  if (window.speechSynthesis) {
    speechSynthesis.onvoiceschanged = voiceLabel;
    voiceLabel();
  }
  renderTutorial(0, false);
}

initTutorial();
