// Draws human-style annotations over the two images, the way a teacher marks
// up a handout: marker circles, numbered badges, curved arrows from A to B and
// short labels, with everything that is not annotated dimmed a little.
// Sizes are given in CSS pixels and scaled by `u` (canvas pixels per CSS px).

import { lutColor } from './lib/colormap.js';

export const PALETTE = ['#16a34a', '#2563eb', '#c026d3', '#0891b2', '#ea580c', '#4f46e5', '#ca8a04', '#db2777'];
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

/**
 * When one image shows only part of the other (a crop, or a copy placed on a
 * larger canvas), outline that part and draw arrows from its corners to the
 * corners of the other image, like a zoom callout.
 */
function drawOverlap(p, ann) {
  const o = ann.overlap;
  if (!o) return null;
  const showA = o.coverA < 0.92;
  const showB = !showA && o.inB && o.coverB < 0.92;
  if (!showA && !showB) return null;
  const { g, u } = p;
  const host = showA ? p.a : p.b;
  const quad = (showA ? o.inA : o.inB).map(([x, y]) => [host.x + x * host.s, host.y + y * host.s]);
  const other = showA ? p.b : p.a;
  const corners = [
    [other.x, other.y],
    [other.x + other.w, other.y],
    [other.x + other.w, other.y + other.h],
    [other.x, other.y + other.h],
  ];
  // dim the part of the host image that the other image does not show
  g.save();
  g.beginPath();
  g.rect(host.x, host.y, host.w, host.h);
  g.moveTo(...quad[0]);
  for (let k = 1; k < 4; k++) g.lineTo(...quad[k]);
  g.closePath();
  g.clip('evenodd');
  g.fillStyle = 'rgba(8, 12, 22, 0.5)';
  g.fillRect(host.x, host.y, host.w, host.h);
  g.restore();
  // the outline, clipped to the host image
  g.save();
  g.beginPath();
  g.rect(host.x - 2 * u, host.y - 2 * u, host.w + 4 * u, host.h + 4 * u);
  g.clip();
  for (const [stroke, w, dash] of [
    ['rgba(255,255,255,0.95)', 6, []],
    ['#2563eb', 3, [9 * u, 6 * u]],
  ]) {
    g.beginPath();
    g.moveTo(...quad[0]);
    for (let k = 1; k < 4; k++) g.lineTo(...quad[k]);
    g.closePath();
    g.setLineDash(dash);
    g.strokeStyle = stroke;
    g.lineWidth = w * u;
    g.stroke();
  }
  g.restore();
  // corner arrows: this part of A -> all of B (or all of A -> this part of B)
  for (let k = 0; k < 4; k++) {
    const inside = (pt) => pt[0] >= host.x - 1 && pt[0] <= host.x + host.w + 1 && pt[1] >= host.y - 1 && pt[1] <= host.y + host.h + 1;
    if (!inside(quad[k])) continue;
    const [from, to] = showA ? [quad[k], corners[k]] : [corners[k], quad[k]];
    curvedArrow(p, from, to, '#2563eb', { bend: k < 2 ? -0.06 : 0.06, width: 1.8, alpha: 0.85, dashed: true });
  }
  const cx = quad.reduce((t, q) => t + q[0], 0) / 4;
  const top = Math.min(...quad.map((q) => q[1]));
  const zoom = showA ? o.zoom : 1 / o.zoom;
  const pct = Math.round((showA ? o.coverA : o.coverB) * 100);
  const text = showA ? `B shows this ${pct}% of A${zoom > 1.15 ? `, enlarged ${zoom.toFixed(1)}×` : ''}` : `A fills this ${pct}% of B`;
  const { w, h } = pillSize(p, text, 11.5);
  const x = Math.max(host.x + 2 * u, Math.min(host.x + host.w - w - 2 * u, cx - w / 2));
  const y = Math.max(host.y + 4 * u, Math.min(host.y + host.h - h - 4 * u, top - h - 6 * u >= host.y ? top - h - 6 * u : top + 6 * u));
  p.placed.push(drawPill(p, x, y, text, '#2563eb', 11.5, { solid: true }));
  return { showA, showB };
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
    drawOverlap(p, ann);
    banner(p, p.b, ann.global ? 'B differs almost everywhere' : 'No differences found', ann.global ? RED : '#16a34a');
    return p;
  }
  const lit = focus ? items.filter((i) => i.d.n === focus) : items;
  spotlight(p, lit.flatMap((i) => [i.ea, i.eb]), focus ? 0.5 : 0.24);
  if (!focus) drawOverlap(p, ann);
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

const hexRgb = (hex) => [1, 3, 5].map((i) => parseInt(hex.slice(i, i + 2), 16));

/** Soft-edged blob covering grid cells of an image, in a colour. */
function blob(p, rect, grid, cells, color, alpha) {
  const m = document.createElement('canvas');
  m.width = grid.gw;
  m.height = grid.gh;
  const mg = m.getContext('2d');
  const id = mg.createImageData(grid.gw, grid.gh);
  const [r, g, b] = hexRgb(color);
  for (const i of cells) {
    id.data[i * 4] = r;
    id.data[i * 4 + 1] = g;
    id.data[i * 4 + 2] = b;
    id.data[i * 4 + 3] = 255;
  }
  mg.putImageData(id, 0, 0);
  p.g.save();
  p.g.globalAlpha = alpha;
  p.g.imageSmoothingEnabled = true;
  p.g.imageSmoothingQuality = 'high';
  p.g.drawImage(m, rect.x, rect.y, rect.w, rect.h);
  p.g.restore();
}

/**
 * Where SSCD's copy score comes from: each link pairs a part of A with the
 * part of B it supports, drawn as matching colour patches joined by an arrow
 * whose width is that link's share of the score.
 */
export function drawCopyLinks(bmA, bmB, sscd, { cssWidth, focus = null, underlay = null } = {}) {
  const p = stage(bmA, bmB, { cssWidth, gap: 64 });
  const L = sscd.links;
  if (underlay) underlay(p);
  if (!L || !L.links.length) {
    banner(p, p.b, sscd.score < 0.1 ? 'No copy evidence' : 'Evidence too spread out to pair', '#64748b');
    return p;
  }
  const strength = Math.max(0.45, Math.min(1, (sscd.score - 0.1) / 0.5));
  const items = L.links.map((k, i) => ({ k, n: i + 1, color: PALETTE[i % PALETTE.length] }));
  const toA = ([x, y]) => [p.a.x + x * p.a.w, p.a.y + y * p.a.h];
  const toB = ([x, y]) => [p.b.x + x * p.b.w, p.b.y + y * p.b.h];
  // dim everything a little so the coloured parts stand out
  p.g.save();
  p.g.fillStyle = 'rgba(8, 12, 22, 0.28)';
  p.g.fillRect(p.a.x, p.a.y, p.a.w, p.a.h);
  p.g.fillRect(p.b.x, p.b.y, p.b.w, p.b.h);
  p.g.restore();
  for (const { k, n, color } of items) {
    // a light tint for every link; the one singled out gets a solid patch
    const a = focus ? (focus === n ? 0.6 : 0) : 0.26;
    if (!a) continue;
    blob(p, p.a, sscd.a, k.a, color, a * strength);
    blob(p, p.b, sscd.b, k.b, color, a * strength);
  }
  const maxW = Math.max(...items.map((i) => i.k.weight));
  const order = focus ? [...items.filter((i) => i.n !== focus), ...items.filter((i) => i.n === focus)] : items;
  const mids = [];
  for (const { k, n, color } of order) {
    const on = !focus || focus === n;
    const from = toA(k.ca);
    const to = toB(k.cb);
    const width = 1.8 + 4.2 * Math.sqrt(Math.max(0, k.weight) / maxW);
    const bend = (from[1] + to[1]) / 2 < p.a.y + p.a.h / 2 ? -0.12 : 0.12;
    const mid = curvedArrow(p, from, to, color, { bend, width, alpha: (on ? 1 : 0.25) * strength });
    mids.push({ k, n, color, mid, on });
    p.hits.push({ n, e: { cx: from[0], cy: from[1], rx: 22 * p.u, ry: 22 * p.u } }, { n, e: { cx: to[0], cy: to[1], rx: 22 * p.u, ry: 22 * p.u } });
  }
  for (const { k, n, color } of items) {
    const on = !focus || focus === n;
    const ea = { cx: toA(k.ca)[0], cy: toA(k.ca)[1], rx: 1, ry: 1 };
    const eb = { cx: toB(k.cb)[0], cy: toB(k.cb)[1], rx: 1, ry: 1 };
    badge(p, { ...ea, cx: ea.cx + 9 * p.u, cy: ea.cy + 9 * p.u }, p.a, n, color, { alpha: on ? 1 : 0.35, size: 9.5 });
    badge(p, { ...eb, cx: eb.cx + 9 * p.u, cy: eb.cy + 9 * p.u }, p.b, n, color, { alpha: on ? 1 : 0.35, size: 9.5 });
  }
  // label the strongest links (or the one singled out) with their share of the score
  const labelled = new Set(focus ? [focus] : items.slice(0, 3).map((i) => i.n));
  {
    for (const { k, n, color, mid, on } of mids) {
      if (!on || !labelled.has(n)) continue;
      const pct = sscd.score > 0.05 ? ` (${Math.round((100 * k.weight) / sscd.score)}%)` : '';
      const text = focus ? `+${k.weight.toFixed(2)} of ${sscd.score.toFixed(2)}${pct}` : `+${k.weight.toFixed(2)}`;
      const { w, h } = pillSize(p, text, 11);
      let best = null;
      for (const dy of [0, -1, 1, -2, 2]) {
        const x = Math.max(2, Math.min(p.canvas.width - w - 2, mid[0] - w / 2));
        const y = Math.max(2, Math.min(p.canvas.height - h - 2, mid[1] - h / 2 + dy * (h + 3 * p.u)));
        const hit = p.placed.filter((o) => overlaps({ x, y, w, h }, o)).length;
        if (!best || hit < best.hit) best = { x, y, hit };
        if (!hit) break;
      }
      p.placed.push(drawPill(p, best.x, best.y, text, color, 11, { alpha: strength }));
      void n;
    }
  }
  return p;
}

/**
 * Point-and-compare: hovering over one image lights up everything similar in
 * the other and draws an arrow to the closest match. `source` gives the
 * similarity of one cell to every cell of the other image.
 */
export function probeStage(bmA, bmB, source, { cssWidth } = {}) {
  const p = stage(bmA, bmB, { cssWidth, gap: 56 });
  const heat = (rect, grid, values, lo, hi) => {
    const c = document.createElement('canvas');
    c.width = grid.gw;
    c.height = grid.gh;
    const g = c.getContext('2d');
    const id = g.createImageData(grid.gw, grid.gh);
    for (let i = 0; i < grid.gw * grid.gh; i++) {
      const v = Math.max(0, Math.min(1, (values[i] - lo) / (hi - lo)));
      const [r, gg, b] = lutColor(0.25 + 0.75 * v);
      id.data[i * 4] = r;
      id.data[i * 4 + 1] = gg;
      id.data[i * 4 + 2] = b;
      id.data[i * 4 + 3] = Math.round(225 * v);
    }
    g.putImageData(id, 0, 0);
    p.g.save();
    p.g.imageSmoothingEnabled = true;
    p.g.imageSmoothingQuality = 'high';
    p.g.drawImage(c, rect.x, rect.y, rect.w, rect.h);
    p.g.restore();
  };
  const cellRect = (rect, grid, i) => {
    const cw = rect.w / grid.gw;
    const ch = rect.h / grid.gh;
    return { x: rect.x + (i % grid.gw) * cw, y: rect.y + Math.floor(i / grid.gw) * ch, w: cw, h: ch };
  };
  const draw = (pt) => {
    p.paint();
    p.placed = [];
    if (!pt) {
      const text = 'Point at any part of either image';
      const { w, h } = pillSize(p, text, 13);
      drawPill(p, (p.canvas.width - w) / 2, p.canvas.height - h - 8 * p.u, text, '#2563eb', 13, { solid: true });
      return null;
    }
    const from = pt.side === 'a' ? { rect: p.a, grid: source.ga } : { rect: p.b, grid: source.gb };
    const to = pt.side === 'a' ? { rect: p.b, grid: source.gb } : { rect: p.a, grid: source.ga };
    const gx = Math.min(from.grid.gw - 1, Math.max(0, Math.floor(pt.fx * from.grid.gw)));
    const gy = Math.min(from.grid.gh - 1, Math.max(0, Math.floor(pt.fy * from.grid.gh)));
    const cellIdx = gy * from.grid.gw + gx;
    const values = source.row(pt.side, cellIdx);
    let best = 0;
    for (let i = 1; i < values.length; i++) if (values[i] > values[best]) best = i;
    // dim the other image, then glow wherever it resembles the pointed-at spot
    p.g.save();
    p.g.fillStyle = 'rgba(8, 12, 22, 0.45)';
    p.g.fillRect(to.rect.x, to.rect.y, to.rect.w, to.rect.h);
    p.g.restore();
    heat(to.rect, to.grid, values, source.lo(values), source.hi(values));
    const src = cellRect(from.rect, from.grid, cellIdx);
    const dst = cellRect(to.rect, to.grid, best);
    const eSrc = { cx: src.x + src.w / 2, cy: src.y + src.h / 2, rx: Math.max(src.w, 16 * p.u) * 0.75, ry: Math.max(src.h, 16 * p.u) * 0.75 };
    const eDst = { cx: dst.x + dst.w / 2, cy: dst.y + dst.h / 2, rx: Math.max(dst.w, 16 * p.u) * 0.75, ry: Math.max(dst.h, 16 * p.u) * 0.75 };
    markerEllipse(p, eSrc, '#2563eb', 3, { width: 3 });
    markerEllipse(p, eDst, '#16a34a', 5, { width: 3.4 });
    const a0 = edgePoint(eSrc, eDst.cx, eDst.cy);
    const a1 = edgePoint(eDst, eSrc.cx, eSrc.cy);
    curvedArrow(p, a0, a1, '#16a34a', { bend: eSrc.cy < p.a.y + p.a.h / 2 ? -0.15 : 0.15, width: 3 });
    labelFor(p, eDst, to.rect, `${source.word} ${source.format(values[best])}`, '#16a34a', { size: 12 });
    return { value: values[best], side: pt.side };
  };
  draw(null);
  return { ...p, draw };
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
