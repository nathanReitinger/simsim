// Similarity by registration: can one image be lined up with the other by
// shifting it (phase correlation), by shifting, turning and scaling it
// (Fourier–Mellin), or found inside the other at some scale (multi-scale
// template search)? Phase correlation and the Hanning window follow
// OpenCV's own implementation (OpenCV.js omits them). An affine ECC score
// was tried and left out: it converged differently in the browser and in
// native OpenCV, and found spurious alignments between unrelated images.

import { fft2, goodSize } from './fft.js';
import { fitWithin, grayPIL, resizeImg } from './pixels.js';

/** cv::createHanningWindow (double precision), including OpenCV's final square root. */
export function hanningWindow(w, h) {
  const c0 = (2 * Math.PI) / (w - 1);
  const c1 = (2 * Math.PI) / (h - 1);
  const wc = Float64Array.from({ length: w }, (_, j) => 0.5 * (1 - Math.cos(c0 * j)));
  const out = new Float64Array(w * h);
  for (let i = 0; i < h; i++) {
    const wr = 0.5 * (1 - Math.cos(c1 * i));
    for (let j = 0; j < w; j++) out[i * w + j] = Math.sqrt(wr * wc[j]);
  }
  return out;
}

/** OpenCV's fftShift: swap quadrants (an odd last row/column stays put). */
function fftShift(a, w, h) {
  const xm = w >> 1;
  const ym = h >> 1;
  const out = Float64Array.from(a);
  for (let y = 0; y < ym; y++) {
    for (let x = 0; x < xm; x++) {
      out[y * w + x] = a[(y + ym) * w + x + xm];
      out[(y + ym) * w + x + xm] = a[y * w + x];
      out[y * w + x + xm] = a[(y + ym) * w + x];
      out[(y + ym) * w + x] = a[y * w + x + xm];
    }
  }
  return out;
}

/**
 * cv::phaseCorrelate: the shift that best aligns b with a and the
 * "response" — the share of the normalised cross-power that sits in the
 * 5×5 window around the peak (1 for a pure shift, near 0 for unrelated
 * images). a and b are w×h float planes; window is optional.
 */
export function phaseCorrelate(a, b, w, h, window = null) {
  const M = goodSize(h);
  const N = goodSize(w);
  const pa = new Float64Array(M * N);
  const pb = new Float64Array(M * N);
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      const k = window ? window[y * w + x] : 1;
      pa[y * N + x] = a[y * w + x] * k;
      pb[y * N + x] = b[y * w + x] * k;
    }
  }
  const ai = new Float64Array(M * N);
  const bi = new Float64Array(M * N);
  fft2(pa, ai, N, M);
  fft2(pb, bi, N, M);
  // P = A · conj(B), then P / |P| (OpenCV adds DBL_EPSILON to |P|²)
  const cr = new Float64Array(M * N);
  const ci = new Float64Array(M * N);
  for (let i = 0; i < M * N; i++) {
    const re = pa[i] * pb[i] + ai[i] * bi[i];
    const im = ai[i] * pb[i] - pa[i] * bi[i];
    const m = Math.sqrt(re * re + im * im);
    const d = m / (m * m + Number.EPSILON);
    cr[i] = re * d;
    ci[i] = im * d;
  }
  fft2(cr, ci, N, M, true);
  const C = fftShift(cr, N, M);
  let peak = 0;
  for (let i = 1; i < C.length; i++) if (C[i] > C[peak]) peak = i;
  const py = Math.floor(peak / N);
  const px = peak % N;
  let sx = 0;
  let sy = 0;
  let s = 0;
  for (let y = Math.max(0, py - 2); y <= Math.min(M - 1, py + 2); y++) {
    for (let x = Math.max(0, px - 2); x <= Math.min(N - 1, px + 2); x++) {
      sx += x * C[y * N + x];
      sy += y * C[y * N + x];
      s += C[y * N + x];
    }
  }
  const response = s;
  s += Number.EPSILON;
  return { dx: N / 2 - sx / s, dy: M / 2 - sy / s, response };
}

/** A grey n×n square holding the whole image (aspect kept), padded with its mean grey. */
export function letterbox(img, n) {
  const [w, h] = fitWithin(img.w, img.h, n);
  const g = grayPIL(resizeImg(img, w, h, 'bilinear'));
  let m = 0;
  for (const v of g) m += v;
  m /= g.length;
  const out = new Float64Array(n * n).fill(m);
  const ox = (n - w) >> 1;
  const oy = (n - h) >> 1;
  for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) out[(y + oy) * n + x + ox] = g[y * w + x];
  return out;
}

/**
 * Fourier–Mellin similarity (Reddy & Chatterji 1996): compare the two
 * images' amplitude spectra, which ignore where things are, after a
 * high-pass emphasis and a log-polar warp that turns rotation and scaling
 * into shifts; the phase-correlation response of those warps is the score.
 * Inputs: letterboxed square grey planes of side n. Uses OpenCV's warpPolar.
 */
export function fourierMellin(cv, a, b, n) {
  const win = hanningWindow(n, n);
  const spectrum = (p) => {
    const re = Float64Array.from(p, (v, i) => v * win[i]);
    const im = new Float64Array(n * n);
    fft2(re, im, n, n);
    const mag = new Float64Array(n * n);
    for (let y = 0; y < n; y++) {
      for (let x = 0; x < n; x++) {
        // centred spectrum, emphasised away from the centre (Reddy & Chatterji's high-pass)
        const sy = (y + n / 2) % n;
        const sx = (x + n / 2) % n;
        const fy = Math.cos((Math.PI * (y - n / 2)) / n);
        const fx = Math.cos((Math.PI * (x - n / 2)) / n);
        const t = fx * fy;
        mag[y * n + x] = Math.hypot(re[sy * n + sx], im[sy * n + sx]) * (1 - t) * (2 - t);
      }
    }
    return mag;
  };
  const toPolar = (mag) => {
    const src = cv.matFromArray(n, n, cv.CV_64F, Array.from(mag));
    const dst = new cv.Mat();
    try {
      cv.warpPolar(src, dst, new cv.Size(n, n), new cv.Point(n / 2, n / 2), n / 2, cv.INTER_LINEAR + cv.WARP_POLAR_LOG);
      return Float64Array.from(dst.data64F);
    } finally {
      src.delete();
      dst.delete();
    }
  };
  const la = toPolar(spectrum(a));
  const lb = toPolar(spectrum(b));
  const r = phaseCorrelate(la, lb, n, n, win);
  // rows of the polar image are angles (0..360° over n rows), columns log-radius
  const angle = (-r.dy * 360) / n;
  const scale = Math.exp((r.dx * Math.log(n / 2)) / n);
  return { response: r.response, angle, scale };
}

/**
 * Multi-scale template search: shrink one image step by step and look for it
 * inside the other (normalised cross-correlation, OpenCV matchTemplate), in
 * both directions. Inputs are { w, h, gray } planes; the best match wins.
 */
export function multiScaleTemplate(cv, A, B, { steps = 16, minSide = 24 } = {}) {
  const del = [];
  const keep = (o) => (del.push(o), o);
  const mat = (p) => {
    const m = keep(new cv.Mat(p.h, p.w, cv.CV_8UC1));
    m.data.set(p.gray);
    return m;
  };
  try {
    let best = { score: -1, scale: null, inside: null };
    for (const [outer, inner, name] of [
      [A, B, 'B inside A'],
      [B, A, 'A inside B'],
    ]) {
      const O = mat(outer);
      const I = mat(inner);
      // largest scale at which the inner image fits inside the outer one
      const top = Math.min(1, outer.w / inner.w, outer.h / inner.h);
      for (let k = 0; k < steps; k++) {
        const s = top * 0.25 ** (k / (steps - 1));
        const tw = Math.round(inner.w * s);
        const th = Math.round(inner.h * s);
        if (Math.min(tw, th) < minSide) break;
        const T = keep(new cv.Mat());
        cv.resize(I, T, new cv.Size(tw, th), 0, 0, cv.INTER_AREA);
        const out = keep(new cv.Mat());
        cv.matchTemplate(O, T, out, cv.TM_CCOEFF_NORMED);
        const mm = cv.minMaxLoc(out);
        if (mm.maxVal > best.score) best = { score: mm.maxVal, scale: s, inside: name, x: mm.maxLoc.x, y: mm.maxLoc.y, tw, th };
      }
    }
    return best.score >= -1 && best.scale !== null ? best : null;
  } finally {
    for (const o of del) o.delete();
  }
}
