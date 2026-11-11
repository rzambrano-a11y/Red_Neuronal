const canvas = document.querySelector("#hero-net");
const ctx = canvas ? canvas.getContext("2d") : null;
const reduced = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
const scene = { width: 0, height: 0, nodes: [], edges: [], particles: [], playing: !reduced };

function resize() {
  if (!canvas) return;
  const rect = canvas.getBoundingClientRect();
  if (rect.width < 20 || rect.height < 20) return;
  const dpr = Math.min(window.devicePixelRatio || 1, 2);
  canvas.width = Math.floor(rect.width * dpr);
  canvas.height = Math.floor(rect.height * dpr);
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  scene.width = rect.width;
  scene.height = rect.height;
  layout();
}

function layout() {
  const { width: w, height: h } = scene;
  const xs = [w * 0.08, w * 0.38, w * 0.68, w * 0.92];
  const counts = [2, 8, 4, 1];
  scene.nodes = counts.map((count, layer) => {
    const inset = count === 1 ? 0.5 : count === 2 ? 0.28 : count === 4 ? 0.16 : 0.06;
    const top = h * inset;
    const gap = count === 1 ? 0 : (h * (1 - inset * 2)) / (count - 1);
    return Array.from({ length: count }, (_, index) => ({
      x: xs[layer],
      y: top + gap * index,
      layer,
      index,
      r: layer === 3 ? 13 : layer === 0 ? 11 : 7.5,
    }));
  });
  scene.edges = [];
  for (let layer = 0; layer < 3; layer += 1) {
    scene.nodes[layer].forEach((from) => {
      scene.nodes[layer + 1].forEach((to) => {
        const mid = (from.x + to.x) / 2;
        scene.edges.push({
          from,
          to,
          c1: { x: mid, y: from.y },
          c2: { x: mid, y: to.y },
        });
      });
    });
  }
  if (!scene.particles.length) {
    scene.particles = Array.from({ length: 36 }, () => ({
      edge: Math.floor(Math.random() * scene.edges.length),
      t: Math.random(),
      speed: 0.00016 + Math.random() * 0.00032,
    }));
  }
}

function pointOn(edge, t) {
  const u = 1 - t;
  const { from, to, c1, c2 } = edge;
  return {
    x: u ** 3 * from.x + 3 * u ** 2 * t * c1.x + 3 * u * t ** 2 * c2.x + t ** 3 * to.x,
    y: u ** 3 * from.y + 3 * u ** 2 * t * c1.y + 3 * u * t ** 2 * c2.y + t ** 3 * to.y,
  };
}

function draw(now) {
  const { width, height } = scene;
  ctx.clearRect(0, 0, width, height);
  const wash = ctx.createRadialGradient(width * 0.55, height * 0.5, 10, width * 0.55, height * 0.45, width * 0.48);
  wash.addColorStop(0, "rgba(120, 90, 255, 0.12)");
  wash.addColorStop(1, "rgba(120, 90, 255, 0)");
  ctx.fillStyle = wash;
  ctx.fillRect(0, 0, width, height);

  scene.edges.forEach((edge) => {
    const pulse = 0.16 + 0.08 * Math.sin(now / 700 + edge.from.index);
    ctx.beginPath();
    ctx.moveTo(edge.from.x, edge.from.y);
    ctx.bezierCurveTo(edge.c1.x, edge.c1.y, edge.c2.x, edge.c2.y, edge.to.x, edge.to.y);
    ctx.strokeStyle = `rgba(150, 190, 255, ${pulse})`;
    ctx.lineWidth = 1.15;
    ctx.stroke();
  });

  scene.particles.forEach((particle) => {
    const edge = scene.edges[particle.edge];
    if (!edge) return;
    const point = pointOn(edge, particle.t % 1);
    ctx.beginPath();
    ctx.fillStyle = "#d9fbff";
    ctx.arc(point.x, point.y, 1.8, 0, Math.PI * 2);
    ctx.fill();
  });

  scene.nodes.flat().forEach((node) => {
    const breathe = 0.5 + 0.5 * Math.sin(now / 800 + node.index + node.layer);
    const color = node.layer === 1 || node.layer === 2 ? "#b9a6ff" : "#7ef3ff";
    ctx.beginPath();
    ctx.fillStyle = color;
    ctx.globalAlpha = 0.16 + breathe * 0.18;
    ctx.arc(node.x, node.y, node.r + 8 + breathe * 3, 0, Math.PI * 2);
    ctx.fill();
    ctx.globalAlpha = 1;
    ctx.beginPath();
    ctx.fillStyle = color;
    ctx.arc(node.x, node.y, node.r, 0, Math.PI * 2);
    ctx.fill();
    ctx.beginPath();
    ctx.strokeStyle = "rgba(255,255,255,0.75)";
    ctx.lineWidth = 1.1;
    ctx.arc(node.x, node.y, node.r * 0.42, 0, Math.PI * 2);
    ctx.stroke();
  });
}

let last = performance.now();
function frame(now) {
  const elapsed = Math.min(40, now - last);
  last = now;
  if (scene.playing && scene.edges.length) {
    scene.particles.forEach((particle) => {
      particle.t += particle.speed * elapsed;
      if (particle.t > 1) {
        particle.t %= 1;
        particle.edge = Math.floor(Math.random() * scene.edges.length);
      }
    });
  }
  if (scene.width) draw(now);
  frameBuild(now, elapsed);
  if (scene.playing || !reduced) requestAnimationFrame(frame);
}

if (canvas) canvas.addEventListener("mousemove", (event) => {
  const rect = canvas.getBoundingClientRect();
  const x = event.clientX - rect.left;
  const y = event.clientY - rect.top;
  const hit = scene.nodes.flat().some((node) => Math.hypot(node.x - x, node.y - y) < node.r + 8);
  canvas.style.cursor = hit ? "pointer" : "default";
});

fetch("/api/state")
  .then((response) => response.json())
  .then((state) => {
    if (state.academic_notice) document.querySelector("#notice").textContent = state.academic_notice;
  })
  .catch(() => {});

const buildCanvas = document.querySelector("#build-net");
const buildCtx = buildCanvas.getContext("2d");
const buildSection = document.querySelector(".build");
const stories = [
  ["La capa de entrada", "Dos números entran a la red", "Intentos fallidos y tráfico enviado. Con eso empieza cada conexión simulada."],
  ["Primera capa oculta", "Ocho neuronas buscan el patrón", "ReLU combina los dos datos. Lo que no aporta se apaga, y la red deja de ser una recta."],
  ["Segunda capa oculta", "Cuatro neuronas afinan la señal", "Esta capa mezcla lo que vio la anterior. El patrón de la conexión queda más nítido."],
  ["La salida", "Una probabilidad cierra el camino", "La sigmoide responde entre 0 y 1. Desde 0,5, el modelo llama sospechosa a la conexión."],
];
const build = { width: 0, height: 0, nodes: [], edges: [], particles: [], stars: [], progress: reduced ? 1 : 0 };
let storyIndex = -1;

function resizeBuild() {
  const rect = buildCanvas.getBoundingClientRect();
  if (rect.width < 20 || rect.height < 20) return;
  const dpr = Math.min(window.devicePixelRatio || 1, 2);
  buildCanvas.width = Math.floor(rect.width * dpr);
  buildCanvas.height = Math.floor(rect.height * dpr);
  buildCtx.setTransform(dpr, 0, 0, dpr, 0, 0);
  build.width = rect.width;
  build.height = rect.height;
  const left = rect.width > 760 ? 0.48 : 0.1;
  const xs = [rect.width * left, rect.width * (left + 0.17), rect.width * (left + 0.34), rect.width * Math.min(left + 0.5, 0.93)];
  const counts = [2, 8, 4, 1];
  build.nodes = counts.map((count, layer) => {
    const inset = count === 1 ? 0.5 : count === 2 ? 0.3 : count === 4 ? 0.2 : 0.08;
    const top = rect.height * inset;
    const gap = count === 1 ? 0 : (rect.height * (1 - inset * 2)) / (count - 1);
    return Array.from({ length: count }, (_, index) => ({
      x: xs[layer],
      y: top + gap * index,
      layer,
      index,
      r: layer === 3 ? 14 : layer === 0 ? 12 : 8,
    }));
  });
  build.edges = [];
  for (let layer = 0; layer < 3; layer += 1) {
    build.nodes[layer].forEach((from) => {
      build.nodes[layer + 1].forEach((to) => {
        const mid = (from.x + to.x) / 2;
        build.edges.push({ from, to, layer, c1: { x: mid, y: from.y }, c2: { x: mid, y: to.y } });
      });
    });
  }
  if (!build.particles.length) {
    build.particles = Array.from({ length: 28 }, () => ({
      edge: Math.floor(Math.random() * build.edges.length),
      t: Math.random(),
      speed: 0.00018 + Math.random() * 0.00028,
    }));
  }
  build.stars = Array.from({ length: 36 }, (_, index) => ({
    x: ((index * 97) % 100) / 100,
    y: ((index * 53) % 100) / 100,
    r: index % 4 === 0 ? 1.6 : 0.8,
  }));
}

function clamp01(value) {
  return Math.min(1, Math.max(0, value));
}

function reveal(progress, start, end) {
  return clamp01((progress - start) / (end - start));
}

function paintStory(progress) {
  const index = Math.min(stories.length - 1, Math.floor(progress * stories.length * 0.999));
  if (index !== storyIndex) {
    storyIndex = index;
    const [kicker, title, text] = stories[index];
    document.querySelector("#story-kicker").textContent = kicker;
    document.querySelector("#story-title").textContent = title;
    document.querySelector("#story-text").textContent = text;
    document.querySelector("#story-count").textContent = `0${index + 1} / 04`;
  }
  for (let step = 0; step < 4; step += 1) {
    const local = clamp01(progress * 4 - step);
    document.querySelector(`#meter-${step}`).style.width = `${local * 100}%`;
  }
  document.querySelector("#story-hint").style.opacity = progress < 0.03 ? "1" : "0";
}

function readBuildProgress() {
  if (reduced) return 1;
  const rect = buildSection.getBoundingClientRect();
  const total = buildSection.offsetHeight - window.innerHeight;
  if (total <= 0) return 1;
  return clamp01(-rect.top / total);
}

function liveNode(node, now) {
  return {
    x: node.x + Math.sin(now / 1100 + node.index) * 3,
    y: node.y + Math.sin(now / 700 + node.layer * 1.4 + node.index) * 6,
  };
}

function liveEdge(edge, now) {
  const from = liveNode(edge.from, now);
  const to = liveNode(edge.to, now);
  const mid = (from.x + to.x) / 2;
  const wave = Math.sin(now / 900 + edge.layer) * 8;
  return { from, to, c1: { x: mid, y: from.y + wave }, c2: { x: mid, y: to.y - wave } };
}

function drawPartial(edge, amount) {
  if (amount <= 0.01) return;
  const steps = 22;
  const lastStep = Math.max(1, Math.ceil(steps * amount));
  buildCtx.beginPath();
  buildCtx.moveTo(edge.from.x, edge.from.y);
  for (let step = 1; step <= lastStep; step += 1) {
    const point = pointOn(edge, Math.min(1, step / steps));
    buildCtx.lineTo(point.x, point.y);
  }
  buildCtx.stroke();
}

function drawField(now) {
  const { width, height } = build;
  for (let band = 0; band < 4; band += 1) {
    buildCtx.beginPath();
    const base = height * (0.62 + band * 0.08);
    for (let x = 0; x <= width; x += 10) {
      const y = base + Math.sin(x * 0.008 + now / 900 + band) * (16 + band * 5);
      if (x === 0) buildCtx.moveTo(x, y);
      else buildCtx.lineTo(x, y);
    }
    buildCtx.strokeStyle = band % 2
      ? `rgba(160, 130, 255, ${0.28 - band * 0.04})`
      : `rgba(80, 220, 255, ${0.26 - band * 0.04})`;
    buildCtx.lineWidth = 1.4;
    buildCtx.stroke();
  }
}

function drawBuild(now) {
  if (!build.width) return;
  const progress = build.progress;
  buildCtx.clearRect(0, 0, build.width, build.height);
  drawField(now);
  build.stars.forEach((star, index) => {
    const drift = (now / 18000 + star.x) % 1;
    const twinkle = 0.35 + 0.45 * Math.sin(now / 500 + index);
    buildCtx.globalAlpha = twinkle;
    buildCtx.fillStyle = index % 2 ? "#9befff" : "#d7ccff";
    buildCtx.beginPath();
    buildCtx.arc(drift * build.width, star.y * build.height, star.r, 0, Math.PI * 2);
    buildCtx.fill();
  });
  buildCtx.globalAlpha = 1;
  const edgeWindows = [[0.08, 0.28], [0.34, 0.56], [0.62, 0.84]];
  const amounts = build.edges.map((edge) => reveal(progress, edgeWindows[edge.layer][0], edgeWindows[edge.layer][1]));
  build.edges.forEach((edge, index) => {
    const amount = amounts[index];
    const moving = liveEdge(edge, now);
    buildCtx.strokeStyle = edge.layer === 1 ? "rgba(190, 170, 255, 0.7)" : "rgba(130, 230, 255, 0.62)";
    buildCtx.lineWidth = 1.35;
    drawPartial(moving, amount);
  });
  build.particles.forEach((particle) => {
    const edge = build.edges[particle.edge];
    const amount = edge ? amounts[particle.edge] : 0;
    if (!edge || amount < 0.15) return;
    const point = pointOn(liveEdge(edge, now), Math.min(particle.t % 1, amount));
    buildCtx.beginPath();
    buildCtx.fillStyle = "#f4feff";
    buildCtx.arc(point.x, point.y, 2.1, 0, Math.PI * 2);
    buildCtx.fill();
  });
  const nodeWindows = [[-0.2, 0.08], [0.22, 0.4], [0.5, 0.68], [0.78, 0.94]];
  build.nodes.flat().forEach((node) => {
    const shown = reveal(progress, nodeWindows[node.layer][0], nodeWindows[node.layer][1]);
    if (shown <= 0.01) return;
    const scale = 1 - (1 - shown) ** 3;
    const place = liveNode(node, now);
    const breathe = 0.65 + 0.35 * Math.sin(now / 420 + node.index);
    const color = node.layer === 0 || node.layer === 3 ? "#7ef3ff" : "#c4b4ff";
    buildCtx.globalAlpha = 0.28 * scale * breathe;
    buildCtx.fillStyle = color;
    buildCtx.beginPath();
    buildCtx.arc(place.x, place.y, (node.r + 10) * scale * breathe, 0, Math.PI * 2);
    buildCtx.fill();
    buildCtx.globalAlpha = scale;
    buildCtx.beginPath();
    buildCtx.arc(place.x, place.y, node.r * scale, 0, Math.PI * 2);
    buildCtx.fill();
    buildCtx.strokeStyle = "rgba(255,255,255,0.85)";
    buildCtx.lineWidth = 1.1;
    buildCtx.beginPath();
    buildCtx.arc(place.x, place.y, node.r * 0.4 * scale, 0, Math.PI * 2);
    buildCtx.stroke();
    buildCtx.globalAlpha = 1;
  });
}

function frameBuild(now, elapsed) {
  build.progress = readBuildProgress();
  paintStory(build.progress);
  if (!reduced && build.edges.length) {
    build.particles.forEach((particle) => {
      particle.t += particle.speed * elapsed;
      if (particle.t > 1) {
        particle.t %= 1;
        const open = build.edges
          .map((edge, index) => index)
          .filter((index) => reveal(build.progress, [0.08, 0.34, 0.62][build.edges[index].layer], [0.28, 0.56, 0.84][build.edges[index].layer]) > 0.2);
        particle.edge = open.length ? open[Math.floor(Math.random() * open.length)] : particle.edge;
      }
    });
  }
  drawBuild(now);
}

window.addEventListener("resize", () => { resize(); resizeBuild(); });
window.addEventListener("scroll", () => { build.progress = readBuildProgress(); }, { passive: true });
resize();
resizeBuild();
requestAnimationFrame(frame);
