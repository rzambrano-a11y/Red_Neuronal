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
}

function pointOnEdge(edge, t) {
  const u = 1 - t;
  const { from, to, c1, c2 } = edge;
  return {
    x: u ** 3 * from.x + 3 * u ** 2 * t * c1.x + 3 * u * t ** 2 * c2.x + t ** 3 * to.x,
    y: u ** 3 * from.y + 3 * u ** 2 * t * c1.y + 3 * u * t ** 2 * c2.y + t ** 3 * to.y,
  };
}

function activation(node) {
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

function drawNetwork(now) {
  const { width, height } = scene;
  ctx.clearRect(0, 0, width, height);
  drawAtmosphere(now);
  const wave = Math.sin(now / 1400);
  scene.edges.forEach((edge) => {
    const glow = 0.08 + activation(edge.from) * 0.22;
    ctx.beginPath();
    ctx.moveTo(edge.from.x, edge.from.y);
    ctx.bezierCurveTo(edge.c1.x, edge.c1.y + wave * 4, edge.c2.x, edge.c2.y, edge.to.x, edge.to.y);
    ctx.strokeStyle = `rgba(120, 210, 255, ${glow})`;
    ctx.lineWidth = 2.4;
    ctx.stroke();
    ctx.strokeStyle = `rgba(210, 245, 255, ${glow + 0.12})`;
    ctx.lineWidth = 1;
    ctx.stroke();
  });
  scene.particles.forEach((particle) => {
    const edge = scene.edges[particle.edge];
    if (!edge) return;
    const point = pointOnEdge(edge, particle.t % 1);
    ctx.beginPath();
    ctx.fillStyle = scene.probability != null && scene.probability >= 0.5 ? "#f5b942" : "#bff8ff";
    ctx.arc(point.x, point.y, 2.1, 0, Math.PI * 2);
    ctx.fill();
  });
  scene.nodes.flat().forEach((node) => {
    const level = activation(node);
    const color = node.layer === 3 && scene.probability != null
      ? (scene.probability >= 0.5 ? "#f5b942" : "#3ee0a0")
      : (node.layer === 1 ? "#9d8cff" : "#7eecff");
    ctx.beginPath();
    ctx.fillStyle = color;
    ctx.globalAlpha = 0.18 + level * 0.25;
    ctx.arc(node.x, node.y, node.r + 7 + level * 4, 0, Math.PI * 2);
    ctx.fill();
    ctx.globalAlpha = 1;
    ctx.beginPath();
    ctx.fillStyle = color;
    ctx.arc(node.x, node.y, node.r, 0, Math.PI * 2);
    ctx.fill();
    ctx.beginPath();
    ctx.strokeStyle = "rgba(255,255,255,0.8)";
    ctx.lineWidth = 1.2;
    ctx.arc(node.x, node.y, node.r * 0.45, 0, Math.PI * 2);
    ctx.stroke();
  });
}

function frame(now) {
  const elapsed = now - lastFrame;
  lastFrame = now;
  if (scene.playing && scene.edges.length) {
    scene.particles.forEach((particle) => {
      particle.t += particle.speed * scene.speed * elapsed;
      if (particle.t > 1) {
        particle.t %= 1;
        particle.edge = Math.floor(Math.random() * scene.edges.length);
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
    chip.textContent = "Modelo activo";
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
    if (state.ready) setStep("2");
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
function syncInputs() {
  scene.attempts = Number(attempts.value);
  scene.traffic = Number(traffic.value);
  document.querySelector("#attempts-out").textContent = attempts.value;
  document.querySelector("#traffic-out").textContent = `${traffic.value} MB`;
}
attempts.addEventListener("input", syncInputs);
traffic.addEventListener("input", syncInputs);

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

async function predict(failedAttempts, trafficMb) {
  const error = document.querySelector("#predict-error");
  error.hidden = true;
  const response = await fetch("/api/predict", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ failed_attempts: failedAttempts, traffic_mb: trafficMb }),
  });
  const payload = await response.json();
  if (!response.ok) {
    error.hidden = false;
    error.textContent = payload.error;
    return;
  }
  scene.probability = payload.probability;
  const card = document.querySelector("#result");
  card.hidden = false;
  card.classList.toggle("is-suspicious", payload.label === 1);
  document.querySelector("#result-class").textContent = payload.label ? "Sospechosa" : "Normal";
  document.querySelector("#result-prob").textContent = `${(payload.probability * 100).toFixed(1)}%`;
  document.querySelector("#result-fill").style.width = `${payload.probability * 100}%`;
  document.querySelector("#result-mark").style.left = `${payload.threshold * 100}%`;
  document.querySelector("#result-threshold").textContent = `Umbral utilizado: ${(payload.threshold * 100).toFixed(0)}%.`;
  document.querySelector("#result-values").textContent = `Valores introducidos: ${payload.failed_attempts} intentos · ${Math.round(payload.traffic_mb)} MB.`;
  document.querySelector("#result-explain").textContent = `Explicación: ${payload.explanation}`;
}

canvas.addEventListener("mousemove", (event) => {
  const rect = canvas.getBoundingClientRect();
  const x = event.clientX - rect.left;
  const y = event.clientY - rect.top;
  const node = scene.nodes.flat().find((item) => Math.hypot(item.x - x, item.y - y) < item.r + 8);
  tooltip.hidden = !node;
  if (!node) return;
  tooltip.textContent = `${node.name} · neurona ${node.index + 1}`;
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
