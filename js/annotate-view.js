// Draws human-style annotations over the two images, the way a teacher marks
// up a handout: marker circles, numbered badges, curved arrows from A to B and
// short labels, with everything that is not annotated dimmed a little.
// Sizes are given in CSS pixels and scaled by `u` (canvas pixels per CSS px).

export const PALETTE = ['#16a34a', '#2563eb', '#c026d3', '#0891b2', '#ea580c', '#4f46e5'];
export const RED = '#ef233c';
export const AMBER = '#f59e0b';
const FONT = '-apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif';

export const KIND_TEXT = {
  added: 'added in B',
  removed: 'missing in B',
  recoloured: 'recoloured',
  changed: 'changed',
};

function rng(seed) {
  let s = (seed * 2654435761) >>> 0;
  return () => {
    s = (s + 0x6d2b79f5) | 0;
    let t = Math.imul(s ^ (s >>> 15), 1 | s);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

/**
 * Both images at a common height, side by side with a gap, sized to fit
 * `cssWidth` (and at most `maxCssHeight` tall).
 */
export function stage(bmA, bmB, { cssWidth, gap = 28, maxCssHeight = 560, margin = 14 } = {}) {
  const ra = bmA.width / bmA.height;
  const rb = bmB.width / bmB.height;
  // a margin all round, so circles at the very edge of an image stay whole
  const H = Math.max(120, Math.min(maxCssHeight, (cssWidth - gap - 2 * margin) / (ra + rb)));
  const u = Math.min(3, Math.max(1.5, globalThis.devicePixelRatio || 1));
  const h = Math.round(H * u);
  const m = Math.round(margin * u);
  const wa = Math.round(ra * h);
  const wb = Math.round(rb * h);
  const g0 = Math.round(gap * u);
  const canvas = document.createElement('canvas');
  canvas.width = m + wa + g0 + wb + m;
  canvas.height = h + 2 * m;
  canvas.style.width = `${canvas.width / u}px`;
  canvas.style.maxWidth = '100%';
  const g = canvas.getContext('2d');
  const p = {
    canvas,
    g,
    // canvas pixels per CSS pixel for marks and labels, a little smaller on small images
    u: u * Math.max(0.72, Math.min(1, H / 340)),
    a: { x: m, y: m, w: wa, h, s: h / bmA.height },
    b: { x: m + wa + g0, y: m, w: wb, h, s: h / bmB.height },
    m,
    placed: [],
    hits: [],
  };
  p.paint = () => {
    g.clearRect(0, 0, canvas.width, canvas.height);
    g.imageSmoothingQuality = 'high';
    g.drawImage(bmA, p.a.x, p.a.y, p.a.w, p.a.h);
    g.drawImage(bmB, p.b.x, p.b.y, p.b.w, p.b.h);
  };
  p.paint();
  return p;
}

/** Ellipse (canvas coordinates) around a box in the image's own pixels. */
export function ellipseFor(box, r, u, grow = 1.15) {
  const x1 = r.x + box[0] * r.s;
  const y1 = r.y + box[1] * r.s;
  const x2 = r.x + box[2] * r.s;
  const y2 = r.y + box[3] * r.s;
  const min = 13 * u;
  return {
    cx: (x1 + x2) / 2,
    cy: (y1 + y2) / 2,
    rx: Math.max(min, ((x2 - x1) / 2) * grow + 4 * u),
    ry: Math.max(min, ((y2 - y1) / 2) * grow + 4 * u),
  };
}

/** Dim the images except inside the given ellipses. */
export function spotlight(p, holes, alpha = 0.38) {
  if (!holes.length) return;
  const off = document.createElement('canvas');
  off.width = p.canvas.width;
  off.height = p.canvas.height;
  const o = off.getContext('2d');
  o.fillStyle = `rgba(8, 12, 22, ${alpha})`;
  for (const r of [p.a, p.b]) o.fillRect(r.x, r.y, r.w, r.h);
  o.globalCompositeOperation = 'destination-out';
  for (const e of holes) {
    if (e.rect) {
      o.fillRect(e.cx - e.rx, e.cy - e.ry, 2 * e.rx, 2 * e.ry);
      continue;
    }
    const grad = o.createRadialGradient(0, 0, 0.6, 0, 0, 1);
    grad.addColorStop(0, 'rgba(0,0,0,1)');
    grad.addColorStop(1, 'rgba(0,0,0,0)');
    o.save();
    o.translate(e.cx, e.cy);
    o.scale(e.rx * 1.12, e.ry * 1.12);
    o.fillStyle = grad;
    o.beginPath();
    o.arc(0, 0, 1, 0, Math.PI * 2);
    o.fill();
    o.restore();
  }
  p.g.drawImage(off, 0, 0);
}

/** Marker-pen circle: two slightly different passes that overshoot the start. */
export function markerEllipse(p, e, color, seed, { width = 3.2, alpha = 1 } = {}) {
  const { g, u } = p;
  const r = rng(seed + 1);
  g.save();
  g.globalAlpha = alpha;
  g.lineCap = 'round';
  for (let pass = 0; pass < 2; pass++) {
    const start = r() * Math.PI * 2;
    const rot = (r() - 0.5) * 0.12;
    const sx = 1 + (r() - 0.5) * 0.06;
    const sy = 1 + (r() - 0.5) * 0.06;
    const ox = (r() - 0.5) * 2.5 * u;
    const oy = (r() - 0.5) * 2.5 * u;
    g.beginPath();
    g.ellipse(e.cx + ox, e.cy + oy, e.rx * sx, e.ry * sy, rot, start, start + Math.PI * 2 + 0.5);
    if (pass === 0) {
      g.strokeStyle = 'rgba(255,255,255,0.9)';
      g.lineWidth = (width + 3) * u;
      g.stroke();
    }
    g.strokeStyle = color;
    g.lineWidth = (pass ? width * 0.6 : width) * u;
    g.stroke();
  }
  g.restore();
}

/** Marker-pen frame for a region that covers most of its image. */
export function markerRect(p, e, color, seed, { width = 3.2, alpha = 1 } = {}) {
  const { g, u } = p;
  const r = rng(seed + 7);
  const inset = 5 * u;
  const x = e.cx - e.rx + inset;
  const y = e.cy - e.ry + inset;
  const w = 2 * e.rx - 2 * inset;
  const h = 2 * e.ry - 2 * inset;
  g.save();
  g.globalAlpha = alpha;
  g.lineJoin = 'round';
  for (let pass = 0; pass < 2; pass++) {
    const j = () => (r() - 0.5) * 3 * u;
    g.beginPath();
    g.roundRect(x + j(), y + j(), w + j(), h + j(), 14 * u);
    if (pass === 0) {
      g.strokeStyle = 'rgba(255,255,255,0.9)';
      g.lineWidth = (width + 3) * u;
      g.stroke();
    }
    g.strokeStyle = color;
    g.lineWidth = (pass ? width * 0.6 : width) * u;
    g.stroke();
  }
  g.restore();
}

const overlaps = (a, b) => a.x < b.x + b.w && b.x < a.x + a.w && a.y < b.y + b.h && b.y < a.y + a.h;

/** Numbered badge on an ellipse's upper-left edge, kept inside the image. */
export function badge(p, e, rect, n, color, { size = 10.5, alpha = 1 } = {}) {
  const { g, u } = p;
  const R = size * u;
  const a = (-3 * Math.PI) / 4;
  const m = p.m || 0;
  const x = Math.max(rect.x - m + R + 2 * u, Math.min(rect.x + rect.w + m - R - 2 * u, e.cx + Math.cos(a) * e.rx));
  const y = Math.max(rect.y - m + R + 2 * u, Math.min(rect.y + rect.h + m - R - 2 * u, e.cy + Math.sin(a) * e.ry));
  g.save();
  g.globalAlpha = alpha;
  g.shadowColor = 'rgba(0,0,0,0.35)';
  g.shadowBlur = 4 * u;
  g.beginPath();
  g.arc(x, y, R, 0, Math.PI * 2);
  g.fillStyle = color;
  g.fill();
  g.shadowBlur = 0;
  g.lineWidth = 2 * u;
  g.strokeStyle = '#fff';
  g.stroke();
  g.fillStyle = '#fff';
  g.font = `800 ${Math.round(R * 1.15)}px ${FONT}`;
  g.textAlign = 'center';
  g.textBaseline = 'middle';
  g.fillText(String(n), x, y + 0.5 * u);
  g.restore();
  p.placed.push({ x: x - R, y: y - R, w: 2 * R, h: 2 * R });
  return [x, y];
}

/** Point on an ellipse's edge in the direction of (tx, ty). */
export function edgePoint(e, tx, ty) {
  const dx = tx - e.cx;
  const dy = ty - e.cy;
  const t = 1 / Math.sqrt((dx / e.rx) ** 2 + (dy / e.ry) ** 2 || 1);
  return [e.cx + dx * t, e.cy + dy * t];
}

function arrowHead(g, x, y, ux, uy, size, color, u) {
  g.beginPath();
  g.moveTo(x, y);
  g.lineTo(x - ux * size - uy * size * 0.6, y - uy * size + ux * size * 0.6);
  g.lineTo(x - ux * size * 0.7, y - uy * size * 0.7);
  g.lineTo(x - ux * size + uy * size * 0.6, y - uy * size - ux * size * 0.6);
  g.closePath();
  g.lineJoin = 'round';
  g.strokeStyle = 'rgba(255,255,255,0.95)';
  g.lineWidth = 2 * u;
  g.stroke();
  g.fillStyle = color;
  g.fill();
}

/** Curved arrow; returns the point halfway along the curve. */
export function curvedArrow(p, from, to, color, { bend = 0.2, width = 2.6, alpha = 1, dashed = false } = {}) {
  const { g, u } = p;
  const [x1, y1] = from;
  const [x2, y2] = to;
  const dx = x2 - x1;
  const dy = y2 - y1;
  const len = Math.hypot(dx, dy) || 1;
  const cx = (x1 + x2) / 2 - dy * bend;
  const cy = (y1 + y2) / 2 + dx * bend;
  const head = Math.max(9, width * 3.6) * u;
  // stop the line where the arrowhead begins
  const tx = x2 - cx;
  const ty = y2 - cy;
  const tl = Math.hypot(tx, ty) || 1;
  const ux = tx / tl;
  const uy = ty / tl;
  g.save();
  g.globalAlpha = alpha;
  g.lineCap = 'round';
  if (dashed) g.setLineDash([6 * u, 5 * u]);
  for (const [stroke, w] of [
    ['rgba(255,255,255,0.92)', width + 3],
    [color, width],
  ]) {
    g.beginPath();
    g.moveTo(x1, y1);
    g.quadraticCurveTo(cx, cy, x2 - ux * head * 0.6, y2 - uy * head * 0.6);
    g.strokeStyle = stroke;
    g.lineWidth = w * u;
    g.stroke();
  }
  g.setLineDash([]);
  // a small dot where the arrow starts
  g.beginPath();
  g.arc(x1, y1, 3.2 * u, 0, Math.PI * 2);
  g.fillStyle = color;
  g.fill();
  g.lineWidth = 1.5 * u;
  g.strokeStyle = '#fff';
  g.stroke();
  arrowHead(g, x2, y2, ux, uy, head, color, u);
  g.restore();
  return [0.25 * x1 + 0.5 * cx + 0.25 * x2, 0.25 * y1 + 0.5 * cy + 0.25 * y2];
}

/** Measure a label pill. */
function pillSize(p, text, size) {
  const { g, u } = p;
  g.save();
  g.font = `700 ${size * u}px ${FONT}`;
  const w = g.measureText(text).width + size * 1.3 * u;
  g.restore();
  return { w, h: size * 1.9 * u };
}

/** Draw a label pill at its top-left corner. */
function drawPill(p, x, y, text, color, size, { alpha = 1, solid = false } = {}) {
  const { g, u } = p;
  const { w, h } = pillSize(p, text, size);
  g.save();
  g.globalAlpha = alpha;
  g.shadowColor = 'rgba(0,0,0,0.25)';
  g.shadowBlur = 5 * u;
  g.beginPath();
  g.roundRect(x, y, w, h, h / 2);
  g.fillStyle = solid ? color : 'rgba(255,255,255,0.97)';
  g.fill();
  g.shadowBlur = 0;
  g.lineWidth = 2 * u;
  g.strokeStyle = solid ? '#fff' : color;
  g.stroke();
  g.fillStyle = solid ? '#fff' : '#111827';
  g.font = `700 ${size * u}px ${FONT}`;
  g.textBaseline = 'middle';
  g.fillText(text, x + size * 0.65 * u, y + h / 2 + 0.5 * u);
  g.restore();
  return { x, y, w, h };
}

/**
 * Place a label near an ellipse where it overlaps nothing placed so far,
 * then point at the ellipse with a short arrow.
 */
export function labelFor(p, e, rect, text, color, { size = 11.5, alpha = 1, avoid = [] } = {}) {
  const { u } = p;
  const { w, h } = pillSize(p, text, size);
  const m = 10 * u;
  const cands = [
    [e.cx - w / 2, e.cy + e.ry + m], // below
    [e.cx - w / 2, e.cy - e.ry - m - h], // above
    [e.cx + e.rx + m, e.cy - h / 2], // right
    [e.cx - e.rx - m - w, e.cy - h / 2], // left
    [e.cx + e.rx * 0.6, e.cy + e.ry * 0.8 + m], // below right
    [e.cx - e.rx * 0.6 - w, e.cy + e.ry * 0.8 + m], // below left
    [e.cx + e.rx * 0.6, e.cy - e.ry * 0.8 - m - h], // above right
    [e.cx - e.rx * 0.6 - w, e.cy - e.ry * 0.8 - m - h], // above left
  ];
  const bounds = { x: rect.x - p.m + 2 * u, y: rect.y - p.m + 2 * u, w: rect.w + 2 * p.m - 4 * u, h: rect.h + 2 * p.m - 4 * u };
  const clamp = ([x, y]) => [Math.max(bounds.x, Math.min(bounds.x + bounds.w - w, x)), Math.max(bounds.y, Math.min(bounds.y + bounds.h - h, y))];
  const others = [...p.placed, ...avoid];
  const cost = ([x, y]) => {
    const box = { x, y, w, h };
    let c = 0;
    for (const o of others) if (overlaps(box, o)) c += 1;
    // covering the circled area itself is worse than covering other image content
    const own = { x: e.cx - e.rx * 0.8, y: e.cy - e.ry * 0.8, w: e.rx * 1.6, h: e.ry * 1.6 };
    if (overlaps(box, own)) c += 2;
    return c;
  };
  let best = null;
  for (const c of cands) {
    const pos = clamp(c);
    const k = cost(pos);
    if (!best || k < best.k) best = { pos, k };
    if (k === 0) break;
  }
  const [x, y] = best.pos;
  // short leader arrow from the pill to the ellipse edge
  const px = Math.max(x, Math.min(x + w, e.cx));
  const py = Math.max(y, Math.min(y + h, e.cy));
  const [ex, ey] = edgePoint(e, px, py);
  const dist = Math.hypot(ex - px, ey - py);
  if (dist > 6 * u) {
    const { g } = p;
    g.save();
    g.globalAlpha = alpha;
    g.lineCap = 'round';
    for (const [stroke, lw] of [
      ['rgba(255,255,255,0.9)', 4.5],
      [color, 2],
    ]) {
      g.beginPath();
      g.moveTo(px, py);
      g.lineTo(ex, ey);
      g.strokeStyle = stroke;
      g.lineWidth = lw * u;
      g.stroke();
    }
    g.restore();
  }
  const placed = drawPill(p, x, y, text, color, size, { alpha });
  p.placed.push(placed);
  return placed;
}

/** "Nothing to circle" banner across the middle of one image. */
export function banner(p, rect, text, color) {
  const size = 13;
  const { w, h } = pillSize(p, text, size);
  const x = rect.x + Math.max(4, (rect.w - w) / 2);
  const y = rect.y + rect.h / 2 - h / 2;
  drawPill(p, x, y, text, color, size, { solid: true });
}

const diffText = (d) => (d.label ? `${d.label} · ${KIND_TEXT[d.kind] || 'changed'}` : KIND_TEXT[d.kind] || 'changed');

/**
 * Spot the difference: the same numbered red circle on both images. With
 * `focus`, one difference is singled out and joined across with an arrow.
 */
export function drawDifferences(bmA, bmB, ann, { cssWidth, focus = null, labels = 'auto' } = {}) {
  const p = stage(bmA, bmB, { cssWidth, gap: 34 });
  const items = ann.differences.map((d) => ({ d, ea: ellipseFor(d.a, p.a, p.u), eb: ellipseFor(d.b, p.b, p.u) }));
  if (!items.length) {
    banner(p, p.b, ann.global ? 'B differs almost everywhere' : 'No differences found', ann.global ? RED : '#16a34a');
    return p;
  }
  const lit = focus ? items.filter((i) => i.d.n === focus) : items;
  spotlight(p, lit.flatMap((i) => [i.ea, i.eb]), focus ? 0.5 : 0.24);
  for (const { d, ea, eb } of items) {
    const alpha = focus && d.n !== focus ? 0.35 : 1;
    markerEllipse(p, ea, RED, d.n * 7, { alpha });
    markerEllipse(p, eb, RED, d.n * 7 + 3, { alpha });
    p.hits.push({ n: d.n, e: ea }, { n: d.n, e: eb });
  }
  for (const { d, ea, eb } of items) {
    const alpha = focus && d.n !== focus ? 0.45 : 1;
    badge(p, ea, p.a, d.n, RED, { alpha });
    badge(p, eb, p.b, d.n, RED, { alpha });
  }
  const showLabels = labels === 'all' || (labels === 'auto' && items.length <= 8);
  if (focus) {
    const f = items.find((i) => i.d.n === focus);
    if (f) {
      const from = edgePoint(f.ea, f.eb.cx, f.eb.cy);
      const to = edgePoint(f.eb, f.ea.cx, f.ea.cy);
      curvedArrow(p, from, to, RED, { bend: f.ea.cy < p.a.y + p.a.h / 2 ? -0.18 : 0.18 });
      labelFor(p, f.eb, p.b, `${f.d.n} · ${diffText(f.d)}`, RED, { size: 12.5 });
    }
  } else if (showLabels) {
    for (const { d, eb } of items) labelFor(p, eb, p.b, diffText(d), RED);
  }
  return p;
}

/**
 * Matching regions: each pair circled in its own colour on both images and
 * joined by a curved arrow from A to B, labelled with what it is and how alike.
 */
export function drawRegions(bmA, bmB, ann, { cssWidth, focus = null } = {}) {
  const p = stage(bmA, bmB, { cssWidth, gap: 72 });
  const big = (box, img) => ((box[2] - box[0]) * (box[3] - box[1])) / (img.width * img.height) >= 0.45;
  const shapeFor = (box, rect, img) => {
    if (!big(box, img)) return { e: ellipseFor(box, rect, p.u, 1.04), frame: false };
    const x1 = rect.x + box[0] * rect.s;
    const y1 = rect.y + box[1] * rect.s;
    const x2 = rect.x + box[2] * rect.s;
    const y2 = rect.y + box[3] * rect.s;
    return { e: { cx: (x1 + x2) / 2, cy: (y1 + y2) / 2, rx: (x2 - x1) / 2, ry: (y2 - y1) / 2 }, frame: true };
  };
  const items = ann.regions.map((r, k) => {
    const sa = shapeFor(r.a, p.a, bmA);
    const sb = shapeFor(r.b, p.b, bmB);
    return { r, color: PALETTE[k % PALETTE.length], ea: sa.e, eb: sb.e, fa: sa.frame, fb: sb.frame };
  });
  if (!items.length) {
    banner(p, p.b, 'No part of B closely matches A', '#64748b');
    return p;
  }
  const lit = focus ? items.filter((i) => i.r.n === focus) : items;
  spotlight(
    p,
    lit.flatMap((i) => [i.fa ? { ...i.ea, rect: true } : i.ea, i.fb ? { ...i.eb, rect: true } : i.eb]),
    focus ? 0.55 : 0.4,
  );
  for (const { r, color, ea, eb, fa, fb } of items) {
    const alpha = focus && r.n !== focus ? 0.3 : 1;
    (fa ? markerRect : markerEllipse)(p, ea, color, r.n * 11, { alpha });
    (fb ? markerRect : markerEllipse)(p, eb, color, r.n * 11 + 5, { alpha });
    p.hits.push({ n: r.n, e: ea }, { n: r.n, e: eb });
  }
  const mids = [];
  // draw the focused arrow last so it sits on top
  const order = focus ? [...items.filter((i) => i.r.n !== focus), ...items.filter((i) => i.r.n === focus)] : items;
  for (const { r, color, ea, eb, fa, fb } of order) {
    const alpha = focus && r.n !== focus ? 0.3 : 1;
    // frames: arrow from the right edge of A's frame to the left edge of B's
    const from = fa ? [ea.cx + ea.rx - 5 * p.u, ea.cy] : edgePoint(ea, eb.cx, eb.cy);
    const to = fb ? [eb.cx - eb.rx + 5 * p.u, eb.cy] : edgePoint(eb, ea.cx, ea.cy);
    const bend = (ea.cy + eb.cy) / 2 < p.a.y + p.a.h / 2 ? -0.16 : 0.16;
    const mid = curvedArrow(p, from, to, color, { bend, alpha, width: focus === r.n ? 3.4 : 2.6 });
    mids.push({ r, color, mid, alpha, up: bend < 0 });
  }
  for (const { r, color, ea, eb } of items) {
    const alpha = focus && r.n !== focus ? 0.4 : 1;
    badge(p, ea, p.a, r.n, color, { alpha });
    badge(p, eb, p.b, r.n, color, { alpha });
  }
  // label each arrow just beside its middle, nudged away from other labels
  const W = p.canvas.width;
  const H = p.canvas.height;
  for (const { r, color, mid, alpha, up } of mids) {
    if (focus && r.n !== focus) continue;
    if (!focus && items.length > 6) continue;
    const text = `${r.n} · ${regionName(r)} · ${Math.round(r.sim * 100)}%`;
    const { w, h } = pillSize(p, text, 11);
    const off = h / 2 + 7 * p.u;
    const steps = up ? [-1, 1, -2.2, 2.2, -3.4, 3.4] : [1, -1, 2.2, -2.2, 3.4, -3.4];
    let best = null;
    for (const k of steps) {
      const x = Math.max(2 * p.u, Math.min(W - w - 2 * p.u, mid[0] - w / 2));
      const y = Math.max(2 * p.u, Math.min(H - h - 2 * p.u, mid[1] - h / 2 + k * off));
      const box = { x, y, w, h };
      const hitCount = p.placed.filter((o) => overlaps(box, o)).length;
      if (!best || hitCount < best.hitCount) best = { x, y, hitCount };
      if (!hitCount) break;
    }
    p.placed.push(drawPill(p, best.x, best.y, text, color, 11, { alpha }));
  }
  return p;
}

export function regionName(r) {
  if (r.labelA && r.labelB && r.labelA !== r.labelB) return `${r.labelA} ↔ ${r.labelB}`;
  if (r.labelA || r.labelB) return r.labelA || r.labelB;
  return r.share >= 0.5 ? 'most of the picture' : 'similar area';
}

/** Copy evidence: heat underneath (drawn by `underlay`), circles on the peaks. */
export function drawEvidence(bmA, bmB, ann, { cssWidth, underlay } = {}) {
  const p = stage(bmA, bmB, { cssWidth, gap: 34 });
  if (underlay) underlay(p);
  const pk = ann.peaks;
  const strength = Math.max(0.35, Math.min(1, (pk.score - 0.15) / 0.45));
  for (const [side, r] of [
    ['a', p.a],
    ['b', p.b],
  ]) {
    pk[side].forEach((q, k) => {
      const rad = Math.max(16 * p.u, q.r * r.s * 1.15);
      const e = { cx: r.x + q.x * r.s, cy: r.y + q.y * r.s, rx: rad, ry: rad };
      markerEllipse(p, e, AMBER, k * 13 + (side === 'a' ? 1 : 2), { alpha: k ? strength * 0.8 : strength });
      badge(p, e, r, k + 1, AMBER, { alpha: strength });
      if (k === 0) labelFor(p, e, r, `most evidence${q.label ? ` · ${q.label}` : ''}`, AMBER, { alpha: strength });
    });
  }
  return p;
}

/** Which annotation (by number) is under a click, if any. */
export function hitTest(p, evt) {
  const rect = p.canvas.getBoundingClientRect();
  const x = ((evt.clientX - rect.left) / rect.width) * p.canvas.width;
  const y = ((evt.clientY - rect.top) / rect.height) * p.canvas.height;
  let best = null;
  for (const h of p.hits) {
    const d = ((x - h.e.cx) / h.e.rx) ** 2 + ((y - h.e.cy) / h.e.ry) ** 2;
    if (d <= 1.2 && (!best || d < best.d)) best = { n: h.n, d };
  }
  return best ? best.n : null;
}
