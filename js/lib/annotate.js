// Turn raw similarity signals into human-style annotations: circled
// differences (spot-the-difference), matching regions joined by arrows, and
// the peaks of SSCD's copy evidence.

import { cropImg, lumaFloat, resizeImg } from './pixels.js';
import { ssim } from './iqa.js';

export function invert3(m) {
  const [a, b, c, d, e, f, g, h, i] = m;
  const A = e * i - f * h;
  const B = -(d * i - f * g);
  const C = d * h - e * g;
  const det = a * A + b * B + c * C;
  if (!det) return null;
  return [A, -(b * i - c * h), b * f - c * e, B, a * i - c * g, -(a * f - c * d), C, -(a * h - b * g), a * e - b * d].map((v) => v / det);
}

const area = (b) => Math.max(0, b[2] - b[0]) * Math.max(0, b[3] - b[1]);
const inter = (a, b) => Math.max(0, Math.min(a[2], b[2]) - Math.max(a[0], b[0])) * Math.max(0, Math.min(a[3], b[3]) - Math.max(a[1], b[1]));

/**
 * The detected object that best coincides with a region (or null): it must
 * cover much of the region and lie mostly inside it, so a tie inside a
 * circled person does not name the person.
 */
export function labelFor(box, dets, imageArea = Infinity) {
  let best = null;
  for (const d of dets || []) {
    if (area(d.box) > 0.85 * imageArea) continue; // a frame around the whole picture says nothing
    const i = inter(box, d.box);
    const score = (i / Math.max(1, area(box))) * (i / Math.max(1, area(d.box)));
    if (score >= 0.2 && (!best || score > best.score)) best = { label: d.label, score };
  }
  return best ? best.label : null;
}

const rectOf = (b) => [b[0], b[1], b[2], b[3]];
const gapBetween = (a, b) => Math.max(0, Math.max(a[0], b[0]) - Math.min(a[2], b[2]), Math.max(a[1], b[1]) - Math.min(a[3], b[3]));
const unionBox = (a, b) => [Math.min(a[0], b[0]), Math.min(a[1], b[1]), Math.max(a[2], b[2]), Math.max(a[3], b[3])];

function rgbMat(cv, img) {
  const m = new cv.Mat(img.h, img.w, cv.CV_8UC3);
  m.data.set(img.rgb);
  return m;
}

/** Sample a coarse grid (values at tile centres) bilinearly at (x, y). */
function sampleGrid(grid, cx, cy, x, y) {
  const fx = Math.max(0, Math.min(cx.length - 1, interpIndex(cx, x)));
  const fy = Math.max(0, Math.min(cy.length - 1, interpIndex(cy, y)));
  const x0 = Math.floor(fx);
  const y0 = Math.floor(fy);
  const x1 = Math.min(cx.length - 1, x0 + 1);
  const y1 = Math.min(cy.length - 1, y0 + 1);
  const tx = fx - x0;
  const ty = fy - y0;
  const W = cx.length;
  return (grid[y0 * W + x0] * (1 - tx) + grid[y0 * W + x1] * tx) * (1 - ty) + (grid[y1 * W + x0] * (1 - tx) + grid[y1 * W + x1] * tx) * ty;
}

function interpIndex(centres, v) {
  if (v <= centres[0]) return 0;
  const n = centres.length - 1;
  if (v >= centres[n]) return n;
  let i = 0;
  while (centres[i + 1] < v) i++;
  return i + (v - centres[i]) / (centres[i + 1] - centres[i]);
}

/** Replace shifts that disagree with their neighbours, then fill gaps. */
function smoothShifts(g, cols, rows) {
  for (let pass = 0; pass < 2; pass++) {
    const out = g.slice();
    for (let i = 0; i < rows; i++) {
      for (let j = 0; j < cols; j++) {
        const nb = [];
        for (let di = -1; di <= 1; di++) {
          for (let dj = -1; dj <= 1; dj++) {
            const ii = i + di;
            const jj = j + dj;
            if (ii >= 0 && jj >= 0 && ii < rows && jj < cols && !Number.isNaN(g[ii * cols + jj])) nb.push(g[ii * cols + jj]);
          }
        }
        if (!nb.length) continue;
        nb.sort((p, q) => p - q);
        const med = nb[nb.length >> 1];
        const v = g[i * cols + j];
        if (Number.isNaN(v) || Math.abs(v - med) > 2.5) out[i * cols + j] = nb.length >= 3 ? med : NaN;
      }
    }
    g = out;
  }
  const known = g.filter((v) => !Number.isNaN(v)).sort((p, q) => p - q);
  const fill = known.length ? known[known.length >> 1] : 0;
  return g.map((v) => (Number.isNaN(v) ? fill : v));
}

const ncc = (a, b, mask) => {
  let n = 0;
  let sa = 0;
  let sb = 0;
  for (let i = 0; i < a.length; i++) {
    if (mask && !mask[i]) continue;
    n++;
    sa += a[i];
    sb += b[i];
  }
  const ma = sa / Math.max(1, n);
  const mb = sb / Math.max(1, n);
  let cab = 0;
  let caa = 0;
  let cbb = 0;
  for (let i = 0; i < a.length; i++) {
    if (mask && !mask[i]) continue;
    const x = a[i] - ma;
    const y = b[i] - mb;
    cab += x * y;
    caa += x * x;
    cbb += y * y;
  }
  return cab / Math.sqrt(Math.max(1e-9, caa * cbb));
};

/**
 * Bring B into A's w×h frame as exactly as possible: the global homography H
 * (B -> frame), then an ECC refinement for what keypoints got slightly wrong,
 * then a smooth field of local shifts for scans and redraws that are not
 * related by any single transform. Returns the aligned image, a validity
 * mask, and toB(x, y) mapping a frame point back to B's own pixels.
 */
export function alignDense(cv, A, B, H, w, h) {
  const del = [];
  const keep = (o) => (del.push(o), o);
  try {
    const mA = keep(rgbMat(cv, A));
    const mB = keep(rgbMat(cv, B));
    const Hm = keep(cv.matFromArray(3, 3, cv.CV_64F, H));
    const size = new cv.Size(w, h);
    let warped = keep(new cv.Mat());
    cv.warpPerspective(mB, warped, Hm, size, cv.INTER_LINEAR, cv.BORDER_CONSTANT, new cv.Scalar(0, 0, 0, 0));
    const ones = keep(new cv.Mat(B.h, B.w, cv.CV_8UC1, new cv.Scalar(255)));
    let mask = keep(new cv.Mat());
    cv.warpPerspective(ones, mask, Hm, size, cv.INTER_NEAREST, cv.BORDER_CONSTANT, new cv.Scalar(0));

    const gray = (m) => {
      const g = keep(new cv.Mat());
      cv.cvtColor(m, g, cv.COLOR_RGB2GRAY);
      cv.GaussianBlur(g, g, new cv.Size(0, 0), 1.5);
      return g;
    };
    const gA = gray(mA);
    let gB = gray(warped);
    let W = null; // ECC: aligned(x) = warped(W x)
    const eccInfo = { used: false };

    // 1. ECC homography refinement at <= 480 px; kept only if it helps.
    const s = Math.min(1, 480 / Math.max(w, h));
    const sw = Math.max(8, Math.round(w * s));
    const sh = Math.max(8, Math.round(h * s));
    const small = (g) => {
      const r = keep(new cv.Mat());
      cv.resize(g, r, new cv.Size(sw, sh), 0, 0, cv.INTER_AREA);
      const f = keep(new cv.Mat());
      r.convertTo(f, cv.CV_32F);
      return f;
    };
    try {
      const sA = small(gA);
      const sB = small(gB);
      const Wm = keep(cv.Mat.eye(3, 3, cv.CV_32F));
      const crit = new cv.TermCriteria(cv.TermCriteria_EPS + cv.TermCriteria_COUNT, 60, 1e-5);
      const none = keep(new cv.Mat());
      cv.findTransformECC(sA, sB, Wm, cv.MOTION_HOMOGRAPHY, crit, none, 5);
      const ws = Array.from(Wm.data32F);
      // back to full resolution: S^-1 W S
      const Wf = [ws[0], ws[1], ws[2] / s, ws[3], ws[4], ws[5] / s, ws[6] * s, ws[7] * s, ws[8]];
      const Wfm = keep(cv.matFromArray(3, 3, cv.CV_64F, Wf));
      const w2 = keep(new cv.Mat());
      cv.warpPerspective(warped, w2, Wfm, size, cv.INTER_LINEAR + cv.WARP_INVERSE_MAP, cv.BORDER_CONSTANT, new cv.Scalar(0, 0, 0, 0));
      const m2 = keep(new cv.Mat());
      cv.warpPerspective(mask, m2, Wfm, size, cv.INTER_NEAREST + cv.WARP_INVERSE_MAP, cv.BORDER_CONSTANT, new cv.Scalar(0));
      const g2 = gray(w2);
      const before = ncc(gA.data, gB.data, mask.data);
      const after = ncc(gA.data, g2.data, m2.data);
      let covered = 0;
      for (let i = 0; i < m2.data.length; i++) if (m2.data[i]) covered++;
      Object.assign(eccInfo, { before, after, covered: covered / (w * h) });
      if (after > before && covered > 0.5 * w * h) {
        eccInfo.used = true;
        warped = w2;
        mask = m2;
        gB = g2;
        W = Wf;
      }
    } catch (err) {
      // ECC did not converge: keep the keypoint alignment
      eccInfo.error = typeof err === 'number' ? 'did not converge' : String(err);
    }

    // 2. Local shifts: where does each tile of A sit in B (normalised cross-correlation)?
    const side = Math.max(w, h);
    const tile = Math.max(48, Math.round(side / 7));
    const R = Math.max(6, Math.round(side / 45));
    const step = tile >> 1;
    const xs = [];
    const ys = [];
    for (let x = 0; x <= Math.max(0, w - tile); x += step) xs.push(x);
    for (let y = 0; y <= Math.max(0, h - tile); y += step) ys.push(y);
    const cols = xs.length;
    const rows = ys.length;
    let gx = new Array(cols * rows).fill(NaN);
    let gy = new Array(cols * rows).fill(NaN);
    if (w >= tile && h >= tile) {
      const padded = keep(new cv.Mat());
      cv.copyMakeBorder(gB, padded, R, R, R, R, cv.BORDER_REPLICATE);
      const res = keep(new cv.Mat());
      const mean = keep(new cv.Mat());
      const sd = keep(new cv.Mat());
      const sub = (m, c, p) => {
        const d = m - 2 * c + p;
        return Math.abs(d) < 1e-9 ? 0 : (0.5 * (m - p)) / d;
      };
      for (let i = 0; i < rows; i++) {
        for (let j = 0; j < cols; j++) {
          const T = gA.roi(new cv.Rect(xs[j], ys[i], tile, tile));
          const S = padded.roi(new cv.Rect(xs[j], ys[i], tile + 2 * R, tile + 2 * R));
          try {
            cv.meanStdDev(T, mean, sd);
            if (sd.data64F[0] < 6) continue;
            cv.matchTemplate(S, T, res, cv.TM_CCOEFF_NORMED);
            const mm = cv.minMaxLoc(res);
            if (mm.maxVal < 0.45) continue;
            const { x, y } = mm.maxLoc;
            const at = (yy, xx) => res.floatAt(yy, xx);
            gx[i * cols + j] = x - R + (x > 0 && x < res.cols - 1 ? sub(at(y, x - 1), at(y, x), at(y, x + 1)) : 0);
            gy[i * cols + j] = y - R + (y > 0 && y < res.rows - 1 ? sub(at(y - 1, x), at(y, x), at(y + 1, x)) : 0);
          } finally {
            T.delete();
            S.delete();
          }
        }
      }
    }
    gx = smoothShifts(gx, cols, rows);
    gy = smoothShifts(gy, cols, rows);
    const cx = xs.map((x) => x + tile / 2);
    const cy = ys.map((y) => y + tile / 2);
    const local = gx.some((v) => Math.abs(v) > 0.25) || gy.some((v) => Math.abs(v) > 0.25);
    if (local && cols && rows) {
      const mapX = keep(new cv.Mat(h, w, cv.CV_32FC1));
      const mapY = keep(new cv.Mat(h, w, cv.CV_32FC1));
      const MX = mapX.data32F;
      const MY = mapY.data32F;
      // bilinear interpolation of the shift grid, one row/column of weights at a time
      const fx = new Float64Array(w);
      for (let x = 0; x < w; x++) fx[x] = interpIndex(cx, x);
      for (let y = 0; y < h; y++) {
        const fy = interpIndex(cy, y);
        const y0 = Math.floor(fy);
        const y1 = Math.min(rows - 1, y0 + 1);
        const ty = fy - y0;
        for (let x = 0; x < w; x++) {
          const x0 = Math.floor(fx[x]);
          const x1 = Math.min(cols - 1, x0 + 1);
          const tx = fx[x] - x0;
          const k00 = y0 * cols + x0;
          const k01 = y0 * cols + x1;
          const k10 = y1 * cols + x0;
          const k11 = y1 * cols + x1;
          const dx = (gx[k00] * (1 - tx) + gx[k01] * tx) * (1 - ty) + (gx[k10] * (1 - tx) + gx[k11] * tx) * ty;
          const dy = (gy[k00] * (1 - tx) + gy[k01] * tx) * (1 - ty) + (gy[k10] * (1 - tx) + gy[k11] * tx) * ty;
          MX[y * w + x] = x + dx;
          MY[y * w + x] = y + dy;
        }
      }
      const w3 = keep(new cv.Mat());
      cv.remap(warped, w3, mapX, mapY, cv.INTER_LINEAR, cv.BORDER_CONSTANT, new cv.Scalar(0, 0, 0, 0));
      const m3 = keep(new cv.Mat());
      cv.remap(mask, m3, mapX, mapY, cv.INTER_NEAREST, cv.BORDER_CONSTANT, new cv.Scalar(0));
      warped = w3;
      mask = m3;
    }

    const Hinv = invert3(H);
    const apply = (M, x, y) => {
      const z = M[6] * x + M[7] * y + M[8];
      return [(M[0] * x + M[1] * y + M[2]) / z, (M[3] * x + M[4] * y + M[5]) / z];
    };
    const toB = (x, y) => {
      let p = local ? [x + sampleGrid(gx, cx, cy, x, y), y + sampleGrid(gy, cx, cy, x, y)] : [x, y];
      if (W) p = apply(W, ...p);
      return Hinv ? apply(Hinv, ...p) : p;
    };
    const maxShift = Math.max(0, ...gx.map(Math.abs), ...gy.map(Math.abs));
    return { img: { w, h, rgb: new Uint8Array(warped.data) }, mask: new Uint8Array(mask.data), toB, ecc: eccInfo, maxShift };
  } finally {
    for (const o of del) o.delete();
  }
}

/** Map a frame box to B's pixels through toB (bounding box of its corners). */
export function boxToB(toB, box) {
  const pts = [
    [box[0], box[1]],
    [box[2], box[1]],
    [box[2], box[3]],
    [box[0], box[3]],
  ].map(([x, y]) => toB(x, y));
  const X = pts.map((p) => p[0]);
  const Y = pts.map((p) => p[1]);
  return [Math.min(...X), Math.min(...Y), Math.max(...X), Math.max(...Y)];
}

/**
 * Regions where two aligned images differ. A and B are { w, h, rgb } in the
 * same frame; `mask` marks pixels that B actually covers. A pixel only counts
 * as different when nothing within a few pixels of it in the other image is
 * close in colour, so leftover misalignment of lines and edges is forgiven
 * while things that were added, removed or recoloured stand out.
 */
export function findDifferences(cv, A, B, mask, { max = 15 } = {}) {
  const { w, h } = A;
  const del = [];
  const keep = (o) => (del.push(o), o);
  try {
    const side = Math.max(w, h);
    const sigma = Math.max(1, side / 600);
    const r = Math.max(2, Math.round(side / 220));
    const a = keep(rgbMat(cv, A));
    const b = keep(rgbMat(cv, B));
    cv.GaussianBlur(a, a, new cv.Size(0, 0), sigma);
    cv.GaussianBlur(b, b, new cv.Size(0, 0), sigma);
    const k = keep(cv.getStructuringElement(cv.MORPH_RECT, new cv.Size(2 * r + 1, 2 * r + 1)));
    const lo = keep(new cv.Mat());
    const hi = keep(new cv.Mat());
    const t1 = keep(new cv.Mat());
    const t2 = keep(new cv.Mat());
    const d = keep(new cv.Mat());
    // distance from each pixel of one image to the colour range around it in the other
    const towards = (x, y, out) => {
      cv.dilate(y, hi, k);
      cv.erode(y, lo, k);
      cv.subtract(x, hi, t1);
      cv.subtract(lo, x, t2);
      cv.max(t1, t2, out);
    };
    towards(a, b, d);
    const d2 = keep(new cv.Mat());
    towards(b, a, d2);
    cv.max(d, d2, d);
    // ignore a margin of r pixels around the area B covers
    let valid = null;
    if (mask) {
      const mm = keep(new cv.Mat(h, w, cv.CV_8UC1));
      mm.data.set(mask);
      cv.erode(mm, mm, k);
      valid = new Uint8Array(mm.data);
    }
    const src = d.data;
    const diff = new Uint8Array(w * h);
    let n = 0;
    let sum = 0;
    let sum2 = 0;
    for (let i = 0, j = 0; i < diff.length; i++, j += 3) {
      if (valid && !valid[i]) continue;
      const v = Math.max(src[j], src[j + 1], src[j + 2]);
      diff[i] = v;
      n++;
      sum += v;
      sum2 += v * v;
    }
    if (n < 0.2 * w * h) return { global: false, regions: [], threshold: null, covered: n / (w * h) };
    const mean = sum / n;
    const sd = Math.sqrt(Math.max(0, sum2 / n - mean * mean));
    const T = Math.min(90, Math.max(30, mean + 4 * sd));
    const bin = keep(new cv.Mat(h, w, cv.CV_8UC1));
    let hot = 0;
    for (let i = 0; i < diff.length; i++) {
      const v = diff[i] > T ? 255 : 0;
      bin.data[i] = v;
      if (v) hot++;
    }
    if (hot > 0.3 * n) return { global: true, regions: [], threshold: T };
    // drop specks, then join the strokes of each change into one blob
    const kOpen = keep(cv.getStructuringElement(cv.MORPH_ELLIPSE, new cv.Size(3, 3)));
    cv.morphologyEx(bin, bin, cv.MORPH_OPEN, kOpen);
    const kc = Math.max(5, Math.round(side / 60)) | 1;
    const kClose = keep(cv.getStructuringElement(cv.MORPH_ELLIPSE, new cv.Size(kc, kc)));
    cv.morphologyEx(bin, bin, cv.MORPH_CLOSE, kClose);
    const labels = keep(new cv.Mat());
    const stats = keep(new cv.Mat());
    const cents = keep(new cv.Mat());
    const count = cv.connectedComponentsWithStats(bin, labels, stats, cents, 8, cv.CV_32S);
    const S = stats.data32S;
    const L = labels.data32S;
    const score = new Float64Array(count);
    for (let i = 0; i < L.length; i++) if (L[i]) score[L[i]] += diff[i];
    const minArea = Math.max(24, 0.00012 * w * h);
    let regions = [];
    for (let l = 1; l < count; l++) {
      const [x, y, bw, bh, ar] = S.slice(l * 5, l * 5 + 5);
      if (ar < minArea || bw * bh > 0.35 * w * h) continue;
      regions.push({ box: [x, y, x + bw, y + bh], score: score[l] });
    }
    // merge pieces of the same change: boxes closer than a third of their size
    const span = (bx) => Math.max(bx[2] - bx[0], bx[3] - bx[1]);
    let merged = true;
    while (merged) {
      merged = false;
      for (let i = 0; i < regions.length && !merged; i++) {
        for (let j = i + 1; j < regions.length; j++) {
          const p = regions[i].box;
          const q = regions[j].box;
          const u = unionBox(p, q);
          if (area(u) > 0.15 * w * h || area(u) > 1.6 * (area(p) + area(q))) continue;
          if (gapBetween(p, q) <= Math.max(3, 0.35 * Math.min(span(p), span(q)))) {
            regions[i] = { box: u, score: regions[i].score + regions[j].score };
            regions.splice(j, 1);
            merged = true;
            break;
          }
        }
      }
    }
    // small fragments close together are pieces of one change (the strokes of a
    // redrawn figure in line art); one side must be small, so separate changes
    // that happen to sit side by side stay apart
    const small = 0.004 * w * h;
    merged = true;
    while (merged) {
      merged = false;
      for (let i = 0; i < regions.length && !merged; i++) {
        for (let j = i + 1; j < regions.length; j++) {
          const p = regions[i].box;
          const q = regions[j].box;
          if (Math.min(area(p), area(q)) > small) continue;
          const u = unionBox(p, q);
          if (area(u) > 0.025 * w * h || gapBetween(p, q) > 0.035 * side) continue;
          regions[i] = { box: u, score: regions[i].score + regions[j].score };
          regions.splice(j, 1);
          merged = true;
          break;
        }
      }
    }
    regions.sort((p, q) => q.score - p.score);
    // keep changes that carry a reasonable share of the strongest one
    const top = regions.length ? regions[0].score : 0;
    regions = regions.filter((g) => g.score >= 0.004 * top).slice(0, max);
    return { global: false, regions: regions.map((g) => ({ box: rectOf(g.box), score: g.score })), threshold: T };
  } finally {
    for (const o of del) o.delete();
  }
}

function edgeDensity(img) {
  const Y = lumaFloat(img);
  const { w, h } = img;
  let s = 0;
  for (let y = 1; y < h - 1; y++) {
    for (let x = 1; x < w - 1; x++) {
      const gx = Y[y * w + x + 1] - Y[y * w + x - 1];
      const gy = Y[(y + 1) * w + x] - Y[(y - 1) * w + x];
      s += Math.hypot(gx, gy);
    }
  }
  return s / Math.max(1, (w - 2) * (h - 2));
}

/** What kind of change a difference region shows (A -> B). */
export function describeChange(A, B, box) {
  const x1 = Math.max(0, Math.floor(box[0]));
  const y1 = Math.max(0, Math.floor(box[1]));
  const cw = Math.max(2, Math.min(A.w, Math.ceil(box[2])) - x1);
  const ch = Math.max(2, Math.min(A.h, Math.ceil(box[3])) - y1);
  const ca = cropImg(A, x1, y1, cw, ch);
  const cb = cropImg(B, x1, y1, cw, ch);
  const ea = edgeDensity(ca);
  const eb = edgeDensity(cb);
  if (eb > 1.6 * ea + 2) return 'added';
  if (ea > 1.6 * eb + 2) return 'removed';
  if (cw >= 16 && ch >= 16) {
    const size = 64;
    const ra = resizeImg(ca, size, size);
    const rb = resizeImg(cb, size, size);
    if (ssim(lumaFloat(ra), lumaFloat(rb), size, size).value >= 0.6) return 'recoloured';
  }
  return 'changed';
}

/**
 * Group mutual patch matches into regions that correspond as a whole: two
 * matches join when they are neighbours in A and their partners are
 * neighbours in B. Returns boxes as fractions of each image.
 */
export function matchRegions(mutual, ga, gb, { minSim = 0.5, minSize = 4, max = 6 } = {}) {
  const ms = mutual.filter((m) => m[2] >= minSim);
  const parent = ms.map((_, i) => i);
  const find = (i) => (parent[i] === i ? i : (parent[i] = find(parent[i])));
  const pos = ms.map(([i, j]) => [i % ga.gw, Math.floor(i / ga.gw), j % gb.gw, Math.floor(j / gb.gw)]);
  for (let p = 0; p < ms.length; p++) {
    for (let q = p + 1; q < ms.length; q++) {
      const [ax, ay, bx, by] = pos[p];
      const [cx, cy, dx, dy] = pos[q];
      if (Math.max(Math.abs(ax - cx), Math.abs(ay - cy)) <= 2 && Math.max(Math.abs(bx - dx), Math.abs(by - dy)) <= 3) {
        parent[find(p)] = find(q);
      }
    }
  }
  const groups = new Map();
  ms.forEach((m, k) => {
    const r = find(k);
    if (!groups.has(r)) groups.set(r, []);
    groups.get(r).push(k);
  });
  const out = [];
  for (const ks of groups.values()) {
    if (ks.length < minSize) continue;
    const ax = ks.map((k) => pos[k][0]);
    const ay = ks.map((k) => pos[k][1]);
    const bx = ks.map((k) => pos[k][2]);
    const by = ks.map((k) => pos[k][3]);
    const sim = ks.reduce((s, k) => s + ms[k][2], 0) / ks.length;
    out.push({
      a: [Math.min(...ax) / ga.gw, Math.min(...ay) / ga.gh, (Math.max(...ax) + 1) / ga.gw, (Math.max(...ay) + 1) / ga.gh],
      b: [Math.min(...bx) / gb.gw, Math.min(...by) / gb.gh, (Math.max(...bx) + 1) / gb.gw, (Math.max(...by) + 1) / gb.gh],
      size: ks.length,
      sim,
      share: ks.length / (ga.gw * ga.gh),
    });
  }
  return out.sort((p, q) => q.size * q.sim - p.size * p.sim).slice(0, max);
}

/**
 * Matching regions guided by detected objects: the patches inside each object
 * in A, and where their mutual matches land in B (preferably inside one
 * object there). Patches outside every accepted object fall back to plain
 * spatial grouping. Boxes are in each image's pixels.
 */
export function objectRegions(mutual, ga, gb, detsA, detsB, A, B, { minSim = 0.5, max = 6 } = {}) {
  const ms = mutual.filter((m) => m[2] >= minSim);
  const centre = (idx, g, img) => [(((idx % g.gw) + 0.5) / g.gw) * img.w, ((Math.floor(idx / g.gw) + 0.5) / g.gh) * img.h];
  const pa = ms.map(([i]) => centre(i, ga, A));
  const pb = ms.map(([, j]) => centre(j, gb, B));
  const inside = (p, b) => p[0] >= b[0] && p[0] <= b[2] && p[1] >= b[1] && p[1] <= b[3];
  const usable = (dets, img) => {
    const ok = (dets || []).filter((d) => area(d.box) <= 0.85 * img.w * img.h && area(d.box) >= 0.004 * img.w * img.h);
    // a box drawn around a group (two people) is not itself an object
    return ok.filter((d) => ok.filter((o) => o !== d && o.label === d.label && inter(o.box, d.box) >= 0.6 * area(o.box)).length < 2);
  };
  const candA = usable(detsA, A);
  const candB = usable(detsB, B);
  const cellB = [B.w / gb.gw, B.h / gb.gh];
  const spread = (pts, cell, img) => {
    const xs = pts.map((p) => p[0]).sort((x, y) => x - y);
    const ys = pts.map((p) => p[1]).sort((x, y) => x - y);
    const q = (arr, f) => arr[Math.min(arr.length - 1, Math.max(0, Math.round(f * (arr.length - 1))))];
    const lo = pts.length >= 8 ? 0.08 : 0;
    return [Math.max(0, q(xs, lo) - cell[0] / 2), Math.max(0, q(ys, lo) - cell[1] / 2), Math.min(img.w, q(xs, 1 - lo) + cell[0] / 2), Math.min(img.h, q(ys, 1 - lo) + cell[1] / 2)];
  };
  const found = [];
  for (const d of candA) {
    const ks = [];
    for (let k = 0; k < ms.length; k++) if (inside(pa[k], d.box)) ks.push(k);
    const small = area(d.box) < 0.02 * A.w * A.h;
    if (ks.length < (small ? 2 : 3)) continue;
    // where the object's patches land in B, then the B object that best coincides with that area
    const landed = spread(ks.map((k) => pb[k]), cellB, B);
    let bestB = null;
    for (const e of candB) {
      const i = inter(landed, e.box);
      const iou = i / Math.max(1, area(landed) + area(e.box) - i);
      const sc = iou + (e.label === d.label ? 0.1 : 0);
      if (iou >= 0.3 && (!bestB || sc > bestB.sc)) bestB = { e, sc };
    }
    const sim = ks.reduce((t, k) => t + ms[k][2], 0) / ks.length;
    const boxB = bestB ? bestB.e.box : landed;
    const labelB = bestB ? bestB.e.label : labelFor(landed, detsB, B.w * B.h);
    found.push({ a: d.box.slice(), b: boxB.slice(), ks: new Set(ks), size: ks.length, sim, share: ks.length / (ga.gw * ga.gh), labelA: d.label, labelB, conf: d.score ?? 0 });
  }
  // drop near-duplicates (the same patches claimed by overlapping detections),
  // trusting the more confident detection
  found.sort((p, q) => q.conf - p.conf);
  const kept = [];
  for (const f of found) {
    const dup = kept.some((k) => {
      let both = 0;
      for (const x of f.ks) if (k.ks.has(x)) both++;
      const union = f.ks.size + k.ks.size - both;
      return both / Math.max(1, union) >= 0.6;
    });
    if (!dup) kept.push(f);
  }
  // what no object explains: spatially coherent groups of the remaining matches
  const claimed = new Set();
  for (const k of kept) for (const x of k.ks) claimed.add(x);
  const rest = ms.filter((_, k) => !claimed.has(k));
  const groups = matchRegions(rest, ga, gb, { minSim, minSize: kept.length ? 6 : 4, max }).map((m) => ({
    a: [m.a[0] * A.w, m.a[1] * A.h, m.a[2] * A.w, m.a[3] * A.h],
    b: [m.b[0] * B.w, m.b[1] * B.h, m.b[2] * B.w, m.b[3] * B.h],
    size: m.size,
    sim: m.sim,
    share: m.share,
    labelA: null,
    labelB: null,
  }));
  const all = [...kept, ...groups].sort((p, q) => q.size * q.sim - p.size * p.sim).slice(0, max);
  return all.map(({ ks, conf, ...r }) => ({
    ...r,
    labelA: r.labelA || labelFor(r.a, detsA, A.w * A.h),
    labelB: r.labelB || labelFor(r.b, detsB, B.w * B.h),
  }));
}

/** Split grid cells into k spatial clusters (weighted k-means, deterministic start). */
function kmeansCells(cells, g, weights, k) {
  const pts = cells.map((i) => [(i % g.gw) + 0.5, Math.floor(i / g.gw) + 0.5, Math.max(1e-6, weights[i])]);
  // farthest-point initialisation from the heaviest cell
  const centres = [pts.reduce((b, p) => (p[2] > b[2] ? p : b), pts[0]).slice(0, 2)];
  while (centres.length < k) {
    let far = null;
    let farD = -1;
    for (const p of pts) {
      const d = Math.min(...centres.map((c) => (p[0] - c[0]) ** 2 + (p[1] - c[1]) ** 2));
      if (d > farD) {
        farD = d;
        far = p;
      }
    }
    centres.push(far.slice(0, 2));
  }
  let assign = new Array(pts.length).fill(0);
  for (let it = 0; it < 12; it++) {
    assign = pts.map((p) => {
      let best = 0;
      let bd = Infinity;
      centres.forEach((c, j) => {
        const d = (p[0] - c[0]) ** 2 + (p[1] - c[1]) ** 2;
        if (d < bd) {
          bd = d;
          best = j;
        }
      });
      return best;
    });
    centres.forEach((c, j) => {
      let sx = 0;
      let sy = 0;
      let sw = 0;
      pts.forEach((p, i) => {
        if (assign[i] !== j) return;
        sx += p[0] * p[2];
        sy += p[1] * p[2];
        sw += p[2];
      });
      if (sw) {
        c[0] = sx / sw;
        c[1] = sy / sw;
      }
    });
  }
  const out = centres.map(() => []);
  cells.forEach((c, i) => out[assign[i]].push(c));
  return out.filter((o) => o.length);
}

/**
 * SSCD's copy score, split by which part of A matched which part of B.
 *
 * The descriptor of each image is a sum of per-location terms (GeM pooling
 * written as a sum, pushed through the final linear layer), so the cosine
 * score is exactly a double sum over pairs of locations, R[l][m] (BiLRP-style
 * second-order explanation). Each cell l of A keeps its exact share of the
 * score, evA[l] = sum over B, and points at the cell of B it pairs with most
 * strongly. Neighbouring cells that point at neighbouring cells form one
 * link, so the links (plus what is left over) add up to the score.
 */
export function copyLinks(R, evA, ga, gb, score, { max = 8 } = {}) {
  const nA = ga.gw * ga.gh;
  const nB = gb.gw * gb.gh;
  const target = new Int32Array(nA);
  for (let l = 0; l < nA; l++) {
    let best = -Infinity;
    for (let m = 0; m < nB; m++) {
      const v = R[l * nB + m];
      if (v > best) {
        best = v;
        target[l] = m;
      }
    }
  }
  const cell = (i, g) => [i % g.gw, Math.floor(i / g.gw)];
  const pos = [];
  for (let l = 0; l < nA; l++) if (evA[l] > 0) pos.push(l);
  const parent = new Map(pos.map((l) => [l, l]));
  const find = (l) => {
    while (parent.get(l) !== l) {
      parent.set(l, parent.get(parent.get(l)));
      l = parent.get(l);
    }
    return l;
  };
  for (let i = 0; i < pos.length; i++) {
    const [ax, ay] = cell(pos[i], ga);
    const [bx, by] = cell(target[pos[i]], gb);
    for (let j = i + 1; j < pos.length; j++) {
      const [cx, cy] = cell(pos[j], ga);
      if (Math.max(Math.abs(ax - cx), Math.abs(ay - cy)) > 1) continue;
      const [dx, dy] = cell(target[pos[j]], gb);
      if (Math.max(Math.abs(bx - dx), Math.abs(by - dy)) > 2) continue;
      parent.set(find(pos[i]), find(pos[j]));
    }
  }
  const groups = new Map();
  for (const l of pos) {
    const r = find(l);
    if (!groups.has(r)) groups.set(r, []);
    groups.get(r).push(l);
  }
  let positive = 0;
  let negative = 0;
  for (let l = 0; l < nA; l++) {
    if (evA[l] > 0) positive += evA[l];
    else negative += evA[l];
  }
  const centroid = (cells, g, w) => {
    let sx = 0;
    let sy = 0;
    let sw = 0;
    cells.forEach((i, k) => {
      const [x, y] = cell(i, g);
      const wt = Math.max(1e-9, w ? w[k] : 1);
      sx += (x + 0.5) * wt;
      sy += (y + 0.5) * wt;
      sw += wt;
    });
    return [sx / sw / g.gw, sy / sw / g.gh];
  };
  const box = (cells, g) => {
    const xs = cells.map((i) => i % g.gw);
    const ys = cells.map((i) => Math.floor(i / g.gw));
    return [Math.min(...xs) / g.gw, Math.min(...ys) / g.gh, (Math.max(...xs) + 1) / g.gw, (Math.max(...ys) + 1) / g.gh];
  };
  // A large coherent group (a near copy) is split into a few parts, so the
  // arrows show how content moved: parallel for a copy, crossing for a
  // mirror image, fanning out for a crop.
  const parts = [];
  for (const cells of groups.values()) {
    const k = cells.length >= 16 ? Math.min(5, Math.max(2, Math.round(cells.length / 14))) : 1;
    if (k === 1) parts.push(cells);
    else parts.push(...kmeansCells(cells, ga, evA, k));
  }
  const all = parts
    .map((cells) => {
      const weights = cells.map((l) => evA[l]);
      const b = [...new Set(cells.map((l) => target[l]))];
      return {
        a: cells,
        b,
        weight: weights.reduce((t, v) => t + v, 0),
        ca: centroid(cells, ga, weights),
        cb: centroid(cells.map((l) => target[l]), gb, weights),
        boxA: box(cells, ga),
        boxB: box(b, gb),
      };
    })
    .sort((p, q) => q.weight - p.weight);
  const floor = Math.max(0.02, 0.04 * Math.max(0, score));
  const links = score < 0.1 ? [] : all.filter((g) => g.weight >= floor).slice(0, max);
  const shown = links.reduce((t, g) => t + g.weight, 0);
  return { links, rest: positive - shown, negative, positive, score, target: Array.from(target) };
}

/** Local maxima of an evidence map (cells as fractions of the image). */
export function evidencePeaks(map, { max = 3, rel = 0.45 } = {}) {
  const { gw, gh, values } = map;
  const top = Math.max(...values);
  if (!(top > 0)) return [];
  const peaks = [];
  for (let y = 0; y < gh; y++) {
    for (let x = 0; x < gw; x++) {
      const v = values[y * gw + x];
      if (v < rel * top) continue;
      let isMax = true;
      for (let dy = -1; dy <= 1 && isMax; dy++) {
        for (let dx = -1; dx <= 1; dx++) {
          const xx = x + dx;
          const yy = y + dy;
          if ((dx || dy) && xx >= 0 && yy >= 0 && xx < gw && yy < gh && values[yy * gw + xx] > v) isMax = false;
        }
      }
      if (isMax) peaks.push({ x: (x + 0.5) / gw, y: (y + 0.5) / gh, r: 1.25 / gw, v: v / top });
    }
  }
  return peaks.sort((p, q) => q.v - p.v).slice(0, max);
}
