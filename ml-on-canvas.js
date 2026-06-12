/* ml-on-canvas — classic ML algorithms trained live on a vanilla <canvas>.
 *
 * Five classifiers cycle through a gallery, each trained in real time on a
 * freshly generated 2-D point cloud: k-means++, softmax regression, linear
 * SVM (one-vs-rest, Pegasos), k-NN, and a gini decision tree.
 *
 * Zero dependencies, no build step, one file.
 * https://github.com/funkekaiser/ml-on-canvas — MIT license.
 */

export const ALGORITHMS = ["kmeans", "softmax", "svm", "knn", "tree"];

/* Default palettes. `pal` entries are [r, g, b] cluster colors (one per
   cluster), brighter than what you would pick for text on the same bg. */
export const DEFAULT_THEMES = {
  light: {
    bg: "#f4f4f1", ink: "#1b1b19", ring: "27,27,25", gray: [158, 156, 148],
    pal: [[182, 92, 67], [63, 138, 98], [85, 115, 196], [168, 90, 147], [168, 133, 63]],
  },
  dark: {
    bg: "#131311", ink: "#eceae4", ring: "236,234,228", gray: [104, 102, 94],
    pal: [[207, 122, 94], [96, 168, 128], [122, 150, 224], [196, 122, 176], [194, 156, 85]],
  },
};

/**
 * Create a classifier gallery on a fullscreen background canvas.
 * The engine sizes the canvas to the window and re-rounds on real resizes.
 *
 * options:
 *   theme               "light" | "dark" (default "light")
 *   themes              palette overrides, same shape as DEFAULT_THEMES
 *   algorithms          subset/order of ALGORITHMS (default: all five)
 *   clusters            number of clusters K (default 5; palettes need K colors)
 *   pointsPerCluster    default 48
 *   respectReducedMotion  default true — render one static converged frame
 *                         under prefers-reduced-motion instead of animating
 *   placement(w, h)     -> { x: [lo, hi], y: [lo, hi], sep } in unit coords;
 *                         where cluster centers may spawn (default: centered)
 *   tapIgnore           CSS selector whose taps never flash a cluster
 *                         (default "a, button, input, select, label")
 *   onStatus(label)     fired when the readout text changes
 *   onHover(k | null)   fired when the hovered cluster changes
 *
 * returns: { start, stop, destroy, setTheme, setHover, setAlgorithm,
 *            regenerate, getAlgorithm }
 */
export function createGallery(canvas, options = {}) {
  if (!canvas || !canvas.getContext) {
    throw new Error("ml-on-canvas: createGallery needs a <canvas> element");
  }
  const ctx = canvas.getContext("2d");

  const themes = { ...DEFAULT_THEMES, ...(options.themes || {}) };
  let themeName = options.theme === "dark" ? "dark" : "light";
  const algos = (options.algorithms || ALGORITHMS).filter((a) => ALGORITHMS.includes(a));
  if (!algos.length) algos.push(...ALGORITHMS);
  const K = options.clusters || 5;
  const PPC = options.pointsPerCluster || 48;
  const onStatus = options.onStatus || null;
  const onHover = options.onHover || null;
  const tapIgnore = options.tapIgnore || "a, button, input, select, label";
  const placement = options.placement ||
    ((w, h) => ({ x: [0.12, 0.88], y: [0.16, 0.84], sep: w / h > 1.15 ? 0.17 : 0.21 }));
  const respectRM = options.respectReducedMotion !== false;

  const motionMq = window.matchMedia("(prefers-reduced-motion: reduce)");
  let reduced = respectRM && motionMq.matches;

  const cur = () => themes[themeName] || themes.light;
  const isDark = () => themeName === "dark";
  const mixc = (c1, c2, t) => [
    (c1[0] + (c2[0] - c1[0]) * t) | 0,
    (c1[1] + (c2[1] - c1[1]) * t) | 0,
    (c1[2] + (c2[2] - c1[2]) * t) | 0,
  ];
  const rnd = (a, b) => a + Math.random() * (b - a);
  const gauss = () => (Math.random() + Math.random() + Math.random() - 1.5) * 0.9;
  const pad = (n, d) => String(Math.min(n, Math.pow(10, d) - 1)).padStart(d, "0");
  const ease = (u) => (u < 0.5 ? 2 * u * u : 1 - Math.pow(-2 * u + 2, 2) / 2);

  /* ---- round / algorithm state ---- */
  let w = 0, h = 0;
  let algoIdx = 0, algo = algos[0];
  let pts = [], phase = "in", t0 = 0, alpha = 0;
  let cents = [], iter = 0, moved = 1e9;
  let W = null, epoch = 0, acc = 0, svCount = 0;
  let kNow = 1, kStepT = 0;
  let tree = null, depthNow = 1, depthT = 0;
  let layers = null, lastRegion = 0;
  let dataC = null, dataR = 0, regFade = 0;
  let scene = null; /* cluster anchors for hover hit-testing */
  let lastLabel = null, lastHover = null;

  const feat = (x, y) => {
    const s = 0.5 * Math.min(w, h);
    return [(x - w * 0.5) / s, (y - h * 0.5) / s, 1];
  };

  /* ---- models ---- */
  const predictLinear = (f) => {
    let bi = 0, bs = -1e18;
    for (let k = 0; k < W.length; k++) {
      const s = W[k][0] * f[0] + W[k][1] * f[1] + W[k][2];
      if (s > bs) { bs = s; bi = k; }
    }
    return bi;
  };
  const stepSoftmax = () => {
    const lr = 0.12, n = pts.length;
    const G = W.map(() => [0, 0, 0]);
    for (let i = 0; i < n; i++) {
      const p = pts[i];
      const f = feat(p.x, p.y);
      const s = W.map((wk) => wk[0] * f[0] + wk[1] * f[1] + wk[2]);
      const mx = Math.max.apply(null, s);
      const ex = s.map((v) => Math.exp(v - mx));
      const Z = ex.reduce((a, b) => a + b, 0);
      for (let k = 0; k < K; k++) {
        const e = ex[k] / Z - (p.g === k ? 1 : 0);
        G[k][0] += e * f[0]; G[k][1] += e * f[1]; G[k][2] += e * f[2];
      }
    }
    for (let k = 0; k < K; k++) for (let j = 0; j < 3; j++) W[k][j] -= (lr * G[k][j]) / n;
  };
  const stepSvm = () => {
    const lam = 0.0005, n = pts.length;
    const lr = Math.min(2, 1 / (lam * (epoch + 1))); /* Pegasos schedule */
    const G = W.map((wk) => wk.map((v) => lam * v));
    for (let i = 0; i < n; i++) {
      const p = pts[i];
      const f = feat(p.x, p.y);
      for (let k = 0; k < K; k++) {
        const yk = p.g === k ? 1 : -1;
        const s = W[k][0] * f[0] + W[k][1] * f[1] + W[k][2];
        if (yk * s < 1) { G[k][0] -= (yk * f[0]) / n; G[k][1] -= (yk * f[1]) / n; G[k][2] -= (yk * f[2]) / n; }
      }
    }
    for (let k = 0; k < K; k++) for (let j = 0; j < 3; j++) W[k][j] -= lr * G[k][j];
  };
  const calcAcc = () => {
    let c = 0;
    for (let i = 0; i < pts.length; i++) if (predictLinear(feat(pts[i].x, pts[i].y)) === pts[i].g) c++;
    return c / pts.length;
  };
  const markSVs = () => {
    svCount = 0;
    for (let i = 0; i < pts.length; i++) {
      const p = pts[i];
      const f = feat(p.x, p.y);
      const s = W[p.g][0] * f[0] + W[p.g][1] * f[1] + W[p.g][2];
      p.sv = s < 1.05;
      if (p.sv) svCount++;
    }
  };
  const predictKnn = (x, y) => {
    const best = [];
    for (let i = 0; i < pts.length; i++) {
      const p = pts[i];
      const d = (p.x - x) * (p.x - x) + (p.y - y) * (p.y - y);
      if (best.length < kNow) { best.push([d, p.g]); best.sort((a, b) => a[0] - b[0]); }
      else if (d < best[best.length - 1][0]) { best[best.length - 1] = [d, p.g]; best.sort((a, b) => a[0] - b[0]); }
    }
    const cnt = new Array(K).fill(0);
    for (let j = 0; j < best.length; j++) cnt[best[j][1]]++;
    let bi = 0;
    for (let k = 1; k < K; k++) if (cnt[k] > cnt[bi]) bi = k;
    return bi;
  };
  const buildTree = () => {
    const gini = (idxs) => {
      const c = new Array(K).fill(0);
      for (let i = 0; i < idxs.length; i++) c[pts[idxs[i]].g]++;
      const n = idxs.length || 1;
      let g = 1;
      for (let k = 0; k < K; k++) g -= (c[k] / n) * (c[k] / n);
      return g;
    };
    const majority = (idxs) => {
      const c = new Array(K).fill(0);
      for (let i = 0; i < idxs.length; i++) c[pts[idxs[i]].g]++;
      let bi = 0;
      for (let k = 1; k < K; k++) if (c[k] > c[bi]) bi = k;
      return bi;
    };
    const split = (idxs, depth) => {
      const node = { leaf: true, cls: majority(idxs), depth };
      if (depth >= 3 || idxs.length < 12 || gini(idxs) < 0.05) return node;
      let best = null;
      for (let ax = 0; ax <= 1; ax++) {
        let lo = 1e18, hi = -1e18;
        for (let i = 0; i < idxs.length; i++) {
          const v = ax ? pts[idxs[i]].y : pts[idxs[i]].x;
          if (v < lo) lo = v; if (v > hi) hi = v;
        }
        for (let q = 1; q < 16; q++) {
          const thr = lo + ((hi - lo) * q) / 16;
          const L = [], R = [];
          for (let i2 = 0; i2 < idxs.length; i2++) {
            (((ax ? pts[idxs[i2]].y : pts[idxs[i2]].x) < thr) ? L : R).push(idxs[i2]);
          }
          if (L.length < 6 || R.length < 6) continue;
          const sc = (L.length * gini(L) + R.length * gini(R)) / idxs.length;
          if (!best || sc < best.sc) best = { sc, ax, thr, L, R };
        }
      }
      if (!best || best.sc > gini(idxs) - 0.02) return node;
      node.leaf = false; node.ax = best.ax; node.thr = best.thr;
      node.l = split(best.L, depth + 1);
      node.r = split(best.R, depth + 1);
      return node;
    };
    return split(pts.map((_, i) => i), 0);
  };
  const predictTree = (x, y) => {
    let n = tree;
    while (n && !n.leaf && n.depth < depthNow) n = ((n.ax ? y : x) < n.thr) ? n.l : n.r;
    return n ? n.cls : 0;
  };

  /* ---- decision-region layers (one offscreen canvas per class, masked
     to a soft radial zone around the data so the page never tints) ---- */
  const renderLayers = () => {
    if (!layers || algo === "kmeans") return;
    const T = cur();
    const cell = Math.max(16, Math.round(Math.min(w, h) / 42));
    for (let l = 0; l < layers.length; l++) layers[l].ctx.clearRect(0, 0, w, h);
    for (let y = 0; y < h + cell; y += cell) {
      for (let x = 0; x < w + cell; x += cell) {
        let cls;
        if (algo === "softmax" || algo === "svm") cls = predictLinear(feat(x + cell / 2, y + cell / 2));
        else if (algo === "knn") cls = predictKnn(x + cell / 2, y + cell / 2);
        else cls = predictTree(x + cell / 2, y + cell / 2);
        const col = T.pal[cls];
        const lc = layers[cls].ctx;
        lc.fillStyle = "rgb(" + col[0] + "," + col[1] + "," + col[2] + ")";
        lc.fillRect(x, y, cell + 1, cell + 1);
      }
    }
    if (dataC) {
      for (let l2 = 0; l2 < layers.length; l2++) {
        const L = layers[l2];
        const g = L.ctx.createRadialGradient(dataC.x, dataC.y, dataR * 0.5, dataC.x, dataC.y, dataR);
        g.addColorStop(0, "rgba(0,0,0,1)");
        g.addColorStop(1, "rgba(0,0,0,0)");
        L.ctx.globalCompositeOperation = "destination-in";
        L.ctx.fillStyle = g;
        L.ctx.fillRect(0, 0, w, h);
        L.ctx.globalCompositeOperation = "source-over";
      }
    }
  };

  /* ---- round setup ---- */
  const newRound = () => {
    algo = algos[algoIdx % algos.length];
    algoIdx++;
    const ctr = [];
    const box = placement(w, h);
    const xLo = box.x[0], xHi = box.x[1];
    const yLo = (box.y && box.y[0]) != null ? box.y[0] : 0.16;
    const yHi = (box.y && box.y[1]) != null ? box.y[1] : 0.84;
    const sep = Math.min(w, h) * box.sep;
    let guard = 0;
    while (ctr.length < K && guard++ < 300) {
      const c = [rnd(xLo, xHi) * w, rnd(yLo, yHi) * h];
      let ok = true;
      for (let i = 0; i < ctr.length; i++) {
        if (Math.hypot(ctr[i][0] - c[0], ctr[i][1] - c[1]) <= sep) { ok = false; break; }
      }
      if (ok) ctr.push(c);
    }
    pts = [];
    ctr.forEach((c, gi) => {
      const s = Math.min(w, h) * rnd(0.085, 0.125);
      for (let i = 0; i < PPC; i++) {
        pts.push({ x: c[0] + gauss() * s, y: c[1] + gauss() * s, g: gi, c: algo === "kmeans" ? -1 : gi, mix: 0, sv: false });
      }
    });
    if (algo === "kmeans") {
      /* k-means++ seeding: spread initial centroids ∝ squared distance */
      cents = [];
      const first = pts[(Math.random() * pts.length) | 0];
      const seeds = [[first.x, first.y]];
      while (seeds.length < K) {
        const d2 = pts.map((p) => {
          let m = 1e18;
          for (let si = 0; si < seeds.length; si++) {
            const dx = p.x - seeds[si][0], dy = p.y - seeds[si][1];
            m = Math.min(m, dx * dx + dy * dy);
          }
          return m;
        });
        const total = d2.reduce((a, b) => a + b, 0);
        let r = Math.random() * total, pick = 0;
        for (let di = 0; di < d2.length; di++) { r -= d2[di]; if (r <= 0) { pick = di; break; } }
        seeds.push([pts[pick].x, pts[pick].y]);
      }
      seeds.forEach((s) => {
        cents.push({ x: s[0], y: s[1], fx: s[0], fy: s[1], tx: s[0], ty: s[1], trail: [] });
      });
      scene = { cents };
    } else {
      scene = { cents: ctr.map((c) => ({ x: c[0], y: c[1] })) };
    }
    if (algo === "softmax" || algo === "svm") {
      W = [];
      for (let k = 0; k < K; k++) W.push([rnd(-1, 1), rnd(-1, 1), rnd(-0.3, 0.3)]);
      epoch = 0; acc = 0; svCount = 0;
    }
    if (algo === "knn") { kNow = 1; kStepT = 0; }
    if (algo === "tree") { tree = buildTree(); depthNow = 1; depthT = 0; }
    iter = 0; moved = 1e9; regFade = 0;
    let mx = 0, my = 0;
    for (let ci = 0; ci < ctr.length; ci++) { mx += ctr[ci][0]; my += ctr[ci][1]; }
    dataC = { x: mx / ctr.length, y: my / ctr.length };
    dataR = 0;
    for (let pi = 0; pi < pts.length; pi++) {
      dataR = Math.max(dataR, Math.hypot(pts[pi].x - dataC.x, pts[pi].y - dataC.y));
    }
    dataR += Math.min(w, h) * 0.14;
    phase = "in"; t0 = performance.now();
  };

  const fit = () => {
    w = window.innerWidth; h = window.innerHeight;
    const DPR = Math.min(window.devicePixelRatio || 1, 2);
    canvas.width = Math.round(w * DPR); canvas.height = Math.round(h * DPR);
    ctx.setTransform(DPR, 0, 0, DPR, 0, 0);
    layers = [];
    for (let k = 0; k < K; k++) {
      const cv = document.createElement("canvas");
      cv.width = w; cv.height = h;
      layers.push({ cv, ctx: cv.getContext("2d") });
    }
  };

  /* ---- k-means helpers ---- */
  const assign = () => {
    for (let i = 0; i < pts.length; i++) {
      const p = pts[i];
      let bi = 0, bd = 1e18;
      for (let k = 0; k < cents.length; k++) {
        const d = (p.x - cents[k].x) * (p.x - cents[k].x) + (p.y - cents[k].y) * (p.y - cents[k].y);
        if (d < bd) { bd = d; bi = k; }
      }
      if (p.c !== bi) { p.c = bi; p.mix = 0; }
    }
  };
  const setTargets = () => {
    let mv = 0;
    for (let k = 0; k < cents.length; k++) {
      let sx = 0, sy = 0, n = 0;
      for (let i = 0; i < pts.length; i++) {
        if (pts[i].c === k) { sx += pts[i].x; sy += pts[i].y; n++; }
      }
      const c = cents[k];
      c.fx = c.x; c.fy = c.y;
      c.tx = n ? sx / n : c.x; c.ty = n ? sy / n : c.y;
      c.trail.push([c.x, c.y]);
      mv = Math.max(mv, Math.hypot(c.tx - c.fx, c.ty - c.fy));
    }
    return mv;
  };

  /* ---- hover: external source (setHover), canvas-side pointer hover
     (nearest cluster anchor within 170px), and mobile tap-flash ---- */
  let externalHover = null, canvasHover = null;
  let flashK = null, flashUntil = 0, tapX = 0, tapY = 0;
  const hoverMq = window.matchMedia("(hover: hover) and (pointer: fine)");

  const onPointerMove = (e) => {
    /* touch "moves" are taps/scrolls, not hover — they'd freeze the
       highlight at the last touched point */
    if (e.pointerType !== "mouse" || !hoverMq.matches) { canvasHover = null; return; }
    if (!scene || !scene.cents.length) return;
    const x = e.clientX, y = e.clientY;
    let best = null, bd = 1e18;
    for (let k = 0; k < scene.cents.length; k++) {
      const d = Math.hypot(scene.cents[k].x - x, scene.cents[k].y - y);
      if (d < bd) { bd = d; best = k; }
    }
    canvasHover = bd < 170 ? best : null;
  };
  const onPointerLeave = () => { canvasHover = null; };

  /* touch: tap the canvas to flash the nearest cluster. Taps on interactive
     elements (tapIgnore selector) are left alone, and a drag (scroll) is
     ignored. The flash rides the same hover pipeline. */
  const onPointerDown = (e) => {
    if (e.pointerType === "mouse") return;
    tapX = e.clientX; tapY = e.clientY;
  };
  const onPointerUp = (e) => {
    if (e.pointerType === "mouse") return;
    if (Math.hypot(e.clientX - tapX, e.clientY - tapY) > 12) return; /* scroll, not tap */
    if (e.target && e.target.closest && e.target.closest(tapIgnore)) return;
    if (!scene || !scene.cents.length) return;
    let best = null, bd = 1e18;
    for (let k = 0; k < scene.cents.length; k++) {
      const d = Math.hypot(scene.cents[k].x - e.clientX, scene.cents[k].y - e.clientY);
      if (d < bd) { bd = d; best = k; }
    }
    if (bd < 200) { flashK = best; flashUntil = performance.now() + 1400; }
    else flashUntil = 0; /* tap on empty space clears an active flash */
  };

  const hoverK = () => {
    if (externalHover != null) return externalHover;
    if (canvasHover != null) return canvasHover;
    return flashK != null && performance.now() < flashUntil ? flashK : null;
  };

  /* ---- drawing ---- */
  const drawFrame = () => {
    const T = cur();
    const hk = hoverK();
    ctx.fillStyle = T.bg;
    ctx.fillRect(0, 0, w, h);

    if (algo !== "kmeans" && layers && regFade > 0.015) {
      const RA = isDark() ? [0.20, 0.34, 0.10] : [0.15, 0.27, 0.07];
      for (let k = 0; k < K; k++) {
        ctx.globalAlpha = alpha * regFade * (hk == null ? RA[0] : hk === k ? RA[1] : RA[2]);
        ctx.drawImage(layers[k].cv, 0, 0, w, h);
      }
      ctx.globalAlpha = 1;
    }

    ctx.save();
    ctx.globalAlpha = alpha;
    for (let i = 0; i < pts.length; i++) {
      const p = pts[i];
      const col = p.c < 0 ? T.gray : mixc(T.gray, T.pal[p.c], p.mix);
      const lit = hk != null && p.c === hk;
      const dim = hk != null && p.c !== hk;
      ctx.fillStyle = "rgba(" + col[0] + "," + col[1] + "," + col[2] + "," + (dim ? 0.15 : lit ? 0.95 : 0.75) + ")";
      ctx.beginPath(); ctx.arc(p.x, p.y, lit ? 3.2 : 2.6, 0, 6.2832); ctx.fill();
      if (algo === "svm" && p.sv && (phase === "hold" || phase === "out")) {
        ctx.strokeStyle = "rgba(" + T.ring + "," + (dim ? 0.2 : 0.55) + ")";
        ctx.lineWidth = 1;
        ctx.beginPath(); ctx.arc(p.x, p.y, 5, 0, 6.2832); ctx.stroke();
      }
    }
    if (algo === "kmeans") {
      for (let k2 = 0; k2 < cents.length; k2++) {
        const c = cents[k2], ccol = T.pal[k2];
        const cdim = hk != null && k2 !== hk;
        if (c.trail.length) {
          ctx.strokeStyle = "rgba(" + ccol[0] + "," + ccol[1] + "," + ccol[2] + "," + (cdim ? 0.12 : 0.35) + ")";
          ctx.lineWidth = 1;
          ctx.setLineDash([3, 3]);
          ctx.beginPath();
          ctx.moveTo(c.trail[0][0], c.trail[0][1]);
          for (let q = 0; q < c.trail.length; q++) ctx.lineTo(c.trail[q][0], c.trail[q][1]);
          ctx.lineTo(c.x, c.y);
          ctx.stroke();
          ctx.setLineDash([]);
        }
        ctx.strokeStyle = "rgba(" + ccol[0] + "," + ccol[1] + "," + ccol[2] + "," + (cdim ? 0.30 : 0.95) + ")";
        ctx.lineWidth = hk === k2 ? 2.5 : 2;
        ctx.beginPath(); ctx.arc(c.x, c.y, hk === k2 ? 9 : 6.5, 0, 6.2832); ctx.stroke();
        ctx.fillStyle = T.ink;
        ctx.beginPath(); ctx.arc(c.x, c.y, 1.5, 0, 6.2832); ctx.fill();
      }
    }
    ctx.restore();
  };

  const setLabel = () => {
    let label;
    if (algo === "kmeans") label = "k-means++ · k = " + K + " · iter " + pad(iter, 2);
    else if (algo === "softmax") label = "softmax regression · epoch " + pad(epoch, 3) + " · acc " + acc.toFixed(2);
    else if (algo === "svm") label = "linear svm (ovr) · epoch " + pad(epoch, 4) + (svCount ? " · sv " + svCount : "");
    else if (algo === "knn") label = "k-nn · k = " + kNow + " · " + pts.length + " samples";
    else label = "decision tree · depth " + depthNow + " · gini";
    if (label !== lastLabel) {
      lastLabel = label;
      if (onStatus) onStatus(label);
    }
  };
  const notifyHover = () => {
    const hk = hoverK();
    if (onHover && hk !== lastHover) { lastHover = hk; onHover(hk); }
  };

  /* ---- main loop ---- */
  let rafId = null, last = 0;
  const tick = (now) => {
    const dt = Math.min(0.05, (now - last) / 1000); last = now;

    if (phase === "in") {
      alpha = Math.min(1, (now - t0) / 900);
      if (alpha >= 1) {
        t0 = now;
        if (algo === "kmeans") { assign(); phase = "assign"; }
        else {
          phase = "run";
          lastRegion = 0;
          if (algo === "knn") { renderLayers(); kStepT = now; }
          if (algo === "tree") { renderLayers(); depthT = now; }
        }
      }
    } else if (phase === "assign") {
      if (now - t0 > 900) { moved = setTargets(); phase = "move"; t0 = now; }
    } else if (phase === "move") {
      const u = Math.min(1, (now - t0) / 1000), e = ease(u);
      for (let ci = 0; ci < cents.length; ci++) {
        const c = cents[ci];
        c.x = c.fx + (c.tx - c.fx) * e; c.y = c.fy + (c.ty - c.fy) * e;
      }
      if (u >= 1) {
        iter++;
        if (iter >= 9 || moved < Math.min(w, h) * 0.004) { phase = "hold"; t0 = now; }
        else { assign(); phase = "assign"; t0 = now; }
      }
    } else if (phase === "run") {
      if (algo === "softmax" || algo === "svm") {
        (algo === "softmax" ? stepSoftmax : stepSvm)();
        epoch++;
        if (algo === "svm") { stepSvm(); epoch++; } /* 2 epochs/frame */
        if (now - lastRegion > 70) { renderLayers(); lastRegion = now; }
        if (epoch % 5 === 0) acc = calcAcc();
        const done = algo === "svm"
          ? epoch >= 1200 /* margin scale needs convergence, not argmax acc */
          : epoch >= 800 || (epoch > 200 && acc >= 0.99);
        if (done) {
          acc = calcAcc();
          if (algo === "svm") markSVs();
          renderLayers();
          phase = "hold"; t0 = now;
        }
      } else if (algo === "knn") {
        if (now - kStepT > 1200) {
          if (kNow >= 9) { phase = "hold"; t0 = now; }
          else { kNow += 2; renderLayers(); kStepT = now; }
        }
      } else {
        if (now - depthT > 1300) {
          if (depthNow >= 3) { phase = "hold"; t0 = now; }
          else { depthNow++; renderLayers(); depthT = now; }
        }
      }
    } else if (phase === "hold") {
      if (now - t0 > 3600) { phase = "out"; t0 = now; }
    } else {
      alpha = Math.max(0, 1 - (now - t0) / 600);
      if (alpha <= 0) newRound();
    }
    for (let pi = 0; pi < pts.length; pi++) pts[pi].mix = Math.min(1, pts[pi].mix + dt * 2.2);

    const regTarget = (algo !== "kmeans" && phase !== "in") ? 1 : 0;
    regFade += (regTarget - regFade) * Math.min(1, dt * 1.6);

    drawFrame();
    notifyHover();
    setLabel();
    rafId = requestAnimationFrame(tick);
  };
  const start = () => {
    if (rafId == null && !reduced) {
      last = performance.now();
      rafId = requestAnimationFrame(tick);
    }
  };
  const stop = () => {
    if (rafId != null) { cancelAnimationFrame(rafId); rafId = null; }
  };

  /* ---- reduced motion: one static, fully-converged frame of the given
     algorithm (defaults to the current one) instead of an animation ---- */
  const staticScene = (name) => {
    const i = algos.indexOf(name != null ? name : algo);
    algoIdx = i >= 0 ? i : 0;
    newRound();
    if (algo === "kmeans") {
      for (let it = 0; it < 9; it++) {
        assign();
        const mv = setTargets();
        for (let ci = 0; ci < cents.length; ci++) {
          cents[ci].x = cents[ci].tx; cents[ci].y = cents[ci].ty;
        }
        iter++;
        if (mv < Math.min(w, h) * 0.004) break;
      }
    } else {
      if (algo === "softmax") {
        while (epoch < 800) {
          stepSoftmax(); epoch++;
          if (epoch % 5 === 0 && epoch > 200 && (acc = calcAcc()) >= 0.99) break;
        }
        acc = calcAcc();
      } else if (algo === "svm") {
        while (epoch < 1200) { stepSvm(); epoch++; }
        acc = calcAcc(); markSVs();
      } else if (algo === "knn") {
        kNow = 9;
      } else {
        depthNow = 3;
      }
      renderLayers();
      regFade = 1;
    }
    for (let pi = 0; pi < pts.length; pi++) pts[pi].mix = 1;
    alpha = 1; phase = "hold";
    drawFrame();
    setLabel();
  };

  const onMotionChange = () => {
    reduced = respectRM && motionMq.matches;
    if (reduced) { stop(); staticScene(algos[0]); }
    else { algoIdx = 0; fit(); newRound(); start(); }
  };

  const onResize = () => {
    const nw = window.innerWidth, nh = window.innerHeight;
    /* ignore mobile URL-bar height jitter — only re-round on real resizes */
    if (nw === w && Math.abs(nh - h) < 120) return;
    fit();
    if (reduced) staticScene();
    else newRound();
  };

  window.addEventListener("pointermove", onPointerMove, { passive: true });
  document.addEventListener("pointerleave", onPointerLeave);
  window.addEventListener("pointerdown", onPointerDown, { passive: true });
  window.addEventListener("pointerup", onPointerUp, { passive: true });
  window.addEventListener("resize", onResize);
  if (motionMq.addEventListener) motionMq.addEventListener("change", onMotionChange);

  fit();
  if (reduced) staticScene(algos[0]);
  else { newRound(); start(); }

  /* ---- public controller ---- */
  return {
    start,
    stop,
    destroy() {
      stop();
      window.removeEventListener("pointermove", onPointerMove);
      document.removeEventListener("pointerleave", onPointerLeave);
      window.removeEventListener("pointerdown", onPointerDown);
      window.removeEventListener("pointerup", onPointerUp);
      window.removeEventListener("resize", onResize);
      if (motionMq.removeEventListener) motionMq.removeEventListener("change", onMotionChange);
      ctx.clearRect(0, 0, w, h);
    },
    setTheme(name) {
      if (!themes[name] || name === themeName) return;
      themeName = name;
      renderLayers();
      if (rafId == null) drawFrame();
    },
    setHover(k) {
      externalHover = k == null ? null : k;
      if (rafId == null) { drawFrame(); notifyHover(); }
    },
    setAlgorithm,
    regenerate: () => setAlgorithm(algo),
    getAlgorithm: () => algo,
  };

  function setAlgorithm(name) {
    if (!algos.includes(name)) return;
    if (reduced) { staticScene(name); return; }
    algoIdx = algos.indexOf(name);
    newRound();
    start();
  }
}
