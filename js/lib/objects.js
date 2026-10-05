// Object detection with D-FINE (Objects365 classes) and object-level
// comparison: find the objects in each image, pair them up, and measure how
// much each pair differs.

import { resize } from './resample.js';
import { LABELS } from './objects365.js';
import { cropImg, lumaFloat, resizeImg } from './pixels.js';
import { deltaE2000Map, ssim } from './iqa.js';
import { correlation, hsHistogram } from './histogram.js';

export const DETECT_SIZE = 640;

/** RT-DETR / D-FINE preprocessing: 640×640 bilinear resize, values in [0, 1]. */
export function detectorInput(img) {
  const r = resize(img.rgb, img.w, img.h, 3, DETECT_SIZE, DETECT_SIZE, 'bilinear');
  const n = DETECT_SIZE * DETECT_SIZE;
  const t = new Float32Array(3 * n);
  for (let i = 0; i < n; i++) {
    t[i] = r[i * 3] / 255;
    t[n + i] = r[i * 3 + 1] / 255;
    t[2 * n + i] = r[i * 3 + 2] / 255;
  }
  return t;
}

export function iou(a, b) {
  const w = Math.min(a[2], b[2]) - Math.max(a[0], b[0]);
  const h = Math.min(a[3], b[3]) - Math.max(a[1], b[1]);
  if (w <= 0 || h <= 0) return 0;
  const inter = w * h;
  return inter / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter);
}

const clamp = (v, lo, hi) => Math.min(hi, Math.max(lo, v));

/**
 * Scores are sigmoid(logits) over every (query, class) pair, as in
 * transformers' RTDetrImageProcessor.post_process_object_detection. Duplicates
 * are then suppressed so each object keeps its best label. Boxes are returned
 * as [x1, y1, x2, y2] in pixels of a w×h image.
 */
export function postprocess(logits, boxes, w, h, { threshold = 0.35, max = 30 } = {}) {
  const Q = boxes.length / 4;
  const C = logits.length / Q;
  const minLogit = Math.log(threshold / (1 - threshold));
  const cand = [];
  for (let q = 0; q < Q; q++) {
    for (let c = 1; c < C; c++) {
      const l = logits[q * C + c];
      if (l > minLogit) cand.push({ q, c, s: 1 / (1 + Math.exp(-l)) });
    }
  }
  cand.sort((a, b) => b.s - a.s);
  const out = [];
  for (const { q, c, s } of cand) {
    const [cx, cy, bw, bh] = boxes.subarray(q * 4, q * 4 + 4);
    const box = [clamp((cx - bw / 2) * w, 0, w), clamp((cy - bh / 2) * h, 0, h), clamp((cx + bw / 2) * w, 0, w), clamp((cy + bh / 2) * h, 0, h)];
    if (box[2] - box[0] < 2 || box[3] - box[1] < 2) continue;
    const dup = out.some((d) => {
      const o = iou(d.box, box);
      return o > 0.8 || (d.labelId === c && o > 0.5);
    });
    if (dup) continue;
    out.push({ labelId: c, label: LABELS[c], score: s, box });
    if (out.length >= max) break;
  }
  return out;
}

// Labels that may name the same object in two images.
const KINDS = [
  ['Desk', 'Dinning Table', 'Coffee Table', 'Side Table', 'Nightstand'],
  ['Hat', 'Helmet'],
  ['Picture/Frame', 'Mirror', 'Monitor/TV', 'Blackboard/Whiteboard'],
  ['Cup', 'Wine Glass', 'Mug', 'Jug', 'Tea pot', 'Kettle'],
  ['Chair', 'Stool', 'Bench', 'Couch'],
  ['Sneakers', 'Other Shoes', 'Leather Shoes', 'Boots', 'Sandals', 'High Heels', 'Slippers'],
  ['Handbag/Satchel', 'Backpack', 'Luggage', 'Briefcase', 'Wallet/Purse'],
  ['Car', 'SUV', 'Van', 'Pickup Truck', 'Sports Car'],
  ['Vase', 'Bottle', 'Potted Plant', 'Flower'],
];
const KIND_OF = new Map(KINDS.flatMap((group, i) => group.map((l) => [l, i])));

export function sameKind(a, b) {
  return a === b || (KIND_OF.has(a) && KIND_OF.get(a) === KIND_OF.get(b));
}

/** Greedy one-to-one assignment from scored candidate pairs. */
function assign(cands) {
  cands.sort((x, y) => y.s - x.s);
  const usedA = new Set();
  const usedB = new Set();
  const pairs = [];
  for (const c of cands) {
    if (usedA.has(c.i) || usedB.has(c.j)) continue;
    usedA.add(c.i);
    usedB.add(c.j);
    pairs.push(c);
  }
  return pairs;
}

/** Pair objects by position once B's boxes are mapped into A's frame. */
export function matchByPosition(detsA, boxesBinA, detsB) {
  const cands = [];
  detsA.forEach((a, i) => {
    boxesBinA.forEach((bb, j) => {
      const o = iou(a.box, bb);
      const kind = sameKind(a.label, detsB[j].label);
      if (o >= 0.3 && (kind || o >= 0.6)) cands.push({ i, j, s: o + (a.label === detsB[j].label ? 0.3 : kind ? 0.15 : 0) });
    });
  });
  return assign(cands);
}

/** Pair objects of the same kind by appearance (similarity function in 0..1). */
export function matchByAppearance(detsA, detsB, sim) {
  const cands = [];
  detsA.forEach((a, i) => {
    detsB.forEach((b, j) => {
      if (!sameKind(a.label, b.label)) return;
      const s = sim(i, j);
      if (s >= 0.25) cands.push({ i, j, s: s + (a.label === b.label ? 0.05 : 0) });
    });
  });
  return assign(cands);
}

/** Map a box through a 3×3 homography; returns the bounding box of the result. */
export function mapBox(H, box) {
  const pts = [
    [box[0], box[1]],
    [box[2], box[1]],
    [box[2], box[3]],
    [box[0], box[3]],
  ].map(([x, y]) => {
    const z = H[6] * x + H[7] * y + H[8];
    return [(H[0] * x + H[1] * y + H[2]) / z, (H[3] * x + H[4] * y + H[5]) / z];
  });
  const xs = pts.map((p) => p[0]);
  const ys = pts.map((p) => p[1]);
  return [Math.min(...xs), Math.min(...ys), Math.max(...xs), Math.max(...ys)];
}

/** Crop with padding (fraction of the box size), clipped to the image. */
export function cropBox(img, box, pad = 0.04) {
  const pw = (box[2] - box[0]) * pad;
  const ph = (box[3] - box[1]) * pad;
  const x1 = Math.max(0, Math.floor(box[0] - pw));
  const y1 = Math.max(0, Math.floor(box[1] - ph));
  const x2 = Math.min(img.w, Math.ceil(box[2] + pw));
  const y2 = Math.min(img.h, Math.ceil(box[3] + ph));
  return cropImg(img, x1, y1, Math.max(1, x2 - x1), Math.max(1, y2 - y1));
}

function maskCrop(mask, w, x1, y1, cw, ch) {
  const out = new Uint8Array(cw * ch);
  for (let y = 0; y < ch; y++) for (let x = 0; x < cw; x++) out[y * cw + x] = mask[(y1 + y) * w + x1 + x];
  return out;
}

/**
 * Compare the same region of two aligned images: SSIM, mean CIEDE2000 and the
 * share of pixels whose colour changed by more than ΔE 6. Pixels inside the
 * `exclude` boxes (smaller objects sitting on this one) are left out, so a
 * change is credited to the innermost object — unless that would leave less
 * than a quarter of the region.
 */
export function compareRegion(A, B, mask, box, exclude = []) {
  const x1 = Math.max(0, Math.floor(box[0]));
  const y1 = Math.max(0, Math.floor(box[1]));
  const x2 = Math.min(A.w, Math.ceil(box[2]));
  const y2 = Math.min(A.h, Math.ceil(box[3]));
  const cw = x2 - x1;
  const ch = y2 - y1;
  if (cw < 2 || ch < 2) return null;
  const ca = cropImg(A, x1, y1, cw, ch);
  const cb = cropImg(B, x1, y1, cw, ch);
  let m = mask ? maskCrop(mask, A.w, x1, y1, cw, ch) : null;
  if (exclude.length) {
    const ex = m ? m.slice() : new Uint8Array(cw * ch).fill(1);
    for (const e of exclude) {
      const ex1 = Math.max(0, Math.floor(e[0]) - x1);
      const ey1 = Math.max(0, Math.floor(e[1]) - y1);
      const ex2 = Math.min(cw, Math.ceil(e[2]) - x1);
      const ey2 = Math.min(ch, Math.ceil(e[3]) - y1);
      for (let y = ey1; y < ey2; y++) ex.fill(0, y * cw + ex1, Math.max(y * cw + ex1, y * cw + ex2));
    }
    let kept = 0;
    for (const v of ex) kept += v ? 1 : 0;
    if (kept >= 0.25 * cw * ch) m = ex;
  }
  const de = deltaE2000Map(ca.rgb, cb.rgb).map;
  let sum = 0;
  let n = 0;
  let changed = 0;
  for (let i = 0; i < de.length; i++) {
    if (m && !m[i]) continue;
    sum += de[i];
    n++;
    if (de[i] > 6) changed++;
  }
  if (!n) return null;
  const s = cw >= 16 && ch >= 16 ? ssim(lumaFloat(ca), lumaFloat(cb), cw, ch, m).value : NaN;
  return { ssim: s, deltaE: sum / n, changed: changed / n, crops: { a: ca, b: cb }, deltaEMap: de, mask: m, w: cw, h: ch };
}

/** Colour-and-structure similarity of two crops of possibly different sizes. */
export function cropSimilarity(ca, cb) {
  const size = 96;
  const ra = resizeImg(ca, size, size, 'bilinear');
  const rb = resizeImg(cb, size, size, 'bilinear');
  const hist = correlation(hsHistogram(ra.rgb, 16, 8), hsHistogram(rb.rgb, 16, 8));
  const s = ssim(lumaFloat(ra), lumaFloat(rb), size, size).value;
  return { hist, ssim: s, combined: Math.max(0, 0.5 * Math.max(0, hist) + 0.5 * Math.max(0, s)) };
}

/** Downscale an image for display (longest side `max`). */
export function thumb(img, max = 160) {
  const s = Math.min(1, max / Math.max(img.w, img.h));
  return resizeImg(img, Math.max(1, Math.round(img.w * s)), Math.max(1, Math.round(img.h * s)), 'bilinear');
}
