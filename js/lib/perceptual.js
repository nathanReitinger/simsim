// Full-reference image-quality metrics from the perceptual-quality
// literature, ported from piq (PyTorch Image Quality) with its default
// settings: VIF, HaarPSI, MDSI, multi-scale GMSD, FSIM, DSS and VSI; plus
// normalised mutual information (as scikit-image). Inputs are two RGB images
// of the same size (A the reference, B the possibly altered copy); pixel
// values stay on the 0..255 scale, as piq uses internally.

import { fft2 } from './fft.js';
import { roundHalfEven } from './resample.js';

// ---------------------------------------------------------------- helpers

/** A plane: { w, h, d: Float64Array }. */
const plane = (w, h, d = new Float64Array(w * h)) => ({ w, h, d });

function channels(img) {
  const n = img.w * img.h;
  const r = new Float64Array(n);
  const g = new Float64Array(n);
  const b = new Float64Array(n);
  for (let i = 0; i < n; i++) {
    r[i] = img.rgb[i * 3];
    g[i] = img.rgb[i * 3 + 1];
    b[i] = img.rgb[i * 3 + 2];
  }
  return [r, g, b];
}

/** Linear colour transform of an RGB image: rows of the 3×3 matrix give the output channels. */
function transform(img, m) {
  const [r, g, b] = channels(img);
  return m.map(([a, c, e]) => plane(img.w, img.h, Float64Array.from(r, (v, i) => a * v + c * g[i] + e * b[i])));
}

const YIQ = [
  [0.299, 0.587, 0.114],
  [0.5959, -0.2746, -0.3213],
  [0.2115, -0.5227, 0.3112],
];
const LHM = [
  [0.2989, 0.587, 0.114],
  [0.3, 0.04, -0.35],
  [0.34, -0.6, 0.17],
];

/** Zero (or edge-replicating) padding: left, right, top, bottom. */
function pad(p, l, r, t, b, mode = 'zero') {
  const out = plane(p.w + l + r, p.h + t + b);
  if (mode === 'zero') {
    for (let y = 0; y < p.h; y++) out.d.set(p.d.subarray(y * p.w, (y + 1) * p.w), (y + t) * out.w + l);
    return out;
  }
  for (let y = 0; y < out.h; y++) {
    let sy = y - t;
    if (mode === 'replicate') sy = Math.min(p.h - 1, Math.max(0, sy));
    if (sy < 0 || sy >= p.h) continue;
    for (let x = 0; x < out.w; x++) {
      let sx = x - l;
      if (mode === 'replicate') sx = Math.min(p.w - 1, Math.max(0, sx));
      if (sx < 0 || sx >= p.w) continue;
      out.d[y * out.w + x] = p.d[sy * p.w + sx];
    }
  }
  return out;
}

/** torch avg_pool2d without padding (output size floors). */
function avgPool(p, k, stride = k) {
  const w = Math.floor((p.w - k) / stride) + 1;
  const h = Math.floor((p.h - k) / stride) + 1;
  const out = plane(w, h);
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      let s = 0;
      for (let dy = 0; dy < k; dy++) for (let dx = 0; dx < k; dx++) s += p.d[(y * stride + dy) * p.w + x * stride + dx];
      out.d[y * w + x] = s / (k * k);
    }
  }
  return out;
}

/** torch conv2d (cross-correlation), 'valid' region, kernel kh×kw. */
function conv(p, k, kw, kh) {
  const w = p.w - kw + 1;
  const h = p.h - kh + 1;
  const out = plane(w, h);
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      let s = 0;
      for (let j = 0; j < kh; j++) {
        const row = (y + j) * p.w + x;
        for (let i = 0; i < kw; i++) s += k[j * kw + i] * p.d[row + i];
      }
      out.d[y * w + x] = s;
    }
  }
  return out;
}

/** Separable cross-correlation, 'valid' region: column filter kc (length kh), then row filter kr (length kw). */
function convSep(p, kc, kr) {
  const kh = kc.length;
  const kw = kr.length;
  const h = p.h - kh + 1;
  const w = p.w - kw + 1;
  const tmp = new Float64Array(h * p.w);
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < p.w; x++) {
      let s = 0;
      for (let j = 0; j < kh; j++) s += kc[j] * p.d[(y + j) * p.w + x];
      tmp[y * p.w + x] = s;
    }
  }
  const out = plane(w, h);
  for (let y = 0; y < h; y++) {
    const row = y * p.w;
    for (let x = 0; x < w; x++) {
      let s = 0;
      for (let i = 0; i < kw; i++) s += kr[i] * tmp[row + x + i];
      out.d[y * w + x] = s;
    }
  }
  return out;
}

function map2(a, b, f) {
  const out = plane(a.w, a.h);
  for (let i = 0; i < out.d.length; i++) out.d[i] = f(a.d[i], b.d[i]);
  return out;
}

function map1(a, f) {
  const out = plane(a.w, a.h);
  for (let i = 0; i < out.d.length; i++) out.d[i] = f(a.d[i]);
  return out;
}
const sum = (a) => a.d.reduce((s, v) => s + v, 0);

/** piq.functional.gaussian_filter */
function gaussian(size, sigma) {
  const k = new Float64Array(size * size);
  const c = (size - 1) / 2;
  let s = 0;
  for (let y = 0; y < size; y++) {
    for (let x = 0; x < size; x++) {
      const v = Math.exp(-((y - c) ** 2 + (x - c) ** 2) / (2 * sigma * sigma));
      k[y * size + x] = v;
      s += v;
    }
  }
  return k.map((v) => v / s);
}

/** The 1-D factor of piq's normalised Gaussian kernel (its outer square is the 2-D kernel). */
function gaussian1(size, sigma) {
  const c = (size - 1) / 2;
  const g = Float64Array.from({ length: size }, (_, i) => Math.exp(-((i - c) ** 2) / (2 * sigma * sigma)));
  const s = g.reduce((a, b) => a + b, 0);
  return g.map((v) => v / s);
}

const PREWITT = [-1, 0, 1, -1, 0, 1, -1, 0, 1].map((v) => v / 3);
const SCHARR = [-3, 0, 3, -10, 0, 10, -3, 0, 3].map((v) => v / 16);
const transpose3 = (k) => [k[0], k[3], k[6], k[1], k[4], k[7], k[2], k[5], k[8]];

/** piq.functional.gradient_map: magnitude of two 3×3 filters, zero padding 1. */
function gradientMap(p, k) {
  const q = pad(p, 1, 1, 1, 1);
  const gx = conv(q, k, 3, 3);
  const gy = conv(q, transpose3(k), 3, 3);
  return map2(gx, gy, (a, b) => Math.sqrt(a * a + b * b));
}

/** piq.functional.similarity_map */
function similarity(a, b, c, alpha = 0) {
  const out = plane(a.w, a.h);
  for (let i = 0; i < out.d.length; i++) {
    const x = a.d[i];
    const y = b.d[i];
    out.d[i] = (2 * x * y - alpha * x * y + c) / (x * x + y * y - alpha * x * y + c);
  }
  return out;
}

/** piq's kernel size for its internal downsampling: max(1, round(min side / 256)), Python rounding. */
const downFactor = (w, h) => Math.max(1, roundHalfEven(Math.min(w, h) / 256));

// ---------------------------------------------------------------- VIF

/**
 * Visual Information Fidelity in the pixel domain (Sheikh & Bovik 2006), as
 * piq.vif_p(x = B, y = A): how much of the information in A survives in B,
 * across four scales. 1 for identical images; can exceed 1 for B with more
 * contrast than A.
 */
export function vifp(A, B, sigmaNsq = 2) {
  if (A.w < 41 || A.h < 41) return null;
  const luma = (img) => transform(img, [[0.299, 0.587, 0.114]])[0];
  let x = luma(B);
  let y = luma(A);
  const EPS = 1e-8;
  let num = 0;
  let den = 0;
  for (let scale = 0; scale < 4; scale++) {
    const ks = 2 ** (4 - scale) + 1;
    const k1 = gaussian1(ks, ks / 5);
    const blur = (p) => convSep(p, k1, k1);
    if (scale > 0) {
      const sub = (p) => {
        const c = blur(p);
        const out = plane(Math.ceil(c.w / 2), Math.ceil(c.h / 2));
        for (let yy = 0; yy < out.h; yy++) for (let xx = 0; xx < out.w; xx++) out.d[yy * out.w + xx] = c.d[2 * yy * c.w + 2 * xx];
        return out;
      };
      x = sub(x);
      y = sub(y);
    }
    if (x.w < ks || x.h < ks) return null;
    const mx = blur(x);
    const my = blur(y);
    const sxx = blur(map1(x, (v) => v * v));
    const syy = blur(map1(y, (v) => v * v));
    const sxy = blur(map2(x, y, (a, b) => a * b));
    for (let i = 0; i < mx.d.length; i++) {
      const sx2 = Math.max(0, sxx.d[i] - mx.d[i] * mx.d[i]);
      let sy2 = Math.max(0, syy.d[i] - my.d[i] * my.d[i]);
      const s12 = sxy.d[i] - mx.d[i] * my.d[i];
      let g = s12 / (sy2 + EPS);
      let sv2 = sx2 - g * s12;
      if (!(sy2 >= EPS)) {
        g = 0;
        sv2 = sx2;
        sy2 = 0;
      }
      if (!(sx2 >= EPS)) {
        g = 0;
        sv2 = 0;
      }
      if (!(g >= 0)) sv2 = sx2;
      g = Math.max(0, g);
      if (!(sv2 > EPS)) sv2 = EPS;
      num += Math.log10(1 + (g * g * sy2) / (sv2 + sigmaNsq));
      den += Math.log10(1 + sy2 / sigmaNsq);
    }
  }
  return (num + EPS) / (den + EPS);
}

// ---------------------------------------------------------------- HaarPSI

/** Haar wavelet-based perceptual similarity (Reisenhofer et al. 2018), as piq.haarpsi. 1 = identical. */
export function haarpsi(A, B, { scales = 3, c = 30, alpha = 4.2 } = {}) {
  if (A.w < 2 ** (scales + 1) || A.h < 2 ** (scales + 1)) return null;
  const prep = (img) => {
    let [Y, I, Q] = transform(img, YIQ);
    const d = Math.max(img.h % 2, img.w % 2);
    [Y, I, Q] = [Y, I, Q].map((p) => avgPool(pad(p, 0, d, 0, d), 2));
    return [Y, I, Q];
  };
  const [Yx, Ix, Qx] = prep(B);
  const [Yy, Iy, Qy] = prep(A);
  const coeffs = (Y) => {
    const out = [];
    for (let s = 0; s < scales; s++) {
      // the Haar kernel is (±1 down the rows) × (1/k across), and its transpose
      const k = 2 ** (s + 1);
      const sign = Float64Array.from({ length: k }, (_, j) => (j < k / 2 ? 1 : -1));
      const flat = new Float64Array(k).fill(1 / k);
      const up = k / 2 - 1;
      const down = k / 2;
      const p = pad(Y, up, down, up, down);
      out.push(convSep(p, sign, flat), convSep(p, flat, sign));
    }
    return out;
  };
  const cx = coeffs(Yx);
  const cy = coeffs(Yy);
  const n = Yx.d.length;
  const w0 = map2(cx[4], cy[4], (a, b) => Math.max(Math.abs(a), Math.abs(b)));
  const w1 = map2(cx[5], cy[5], (a, b) => Math.max(Math.abs(a), Math.abs(b)));
  const sims = [0, 1].map((o) => {
    const s0 = similarity(map1(cx[o], Math.abs), map1(cy[o], Math.abs), c);
    const s1 = similarity(map1(cx[o + 2], Math.abs), map1(cy[o + 2], Math.abs), c);
    return map2(s0, s1, (a, b) => (a + b) / 2);
  });
  const weights = [w0, w1];
  // chromatic part: 2×2 means of I and Q (stride 1, zero row/column added)
  const iq = (p) => map1(avgPool(pad(p, 0, 1, 0, 1), 2, 1), Math.abs);
  const sI = similarity(iq(Ix), iq(Iy), c);
  const sQ = similarity(iq(Qx), iq(Qy), c);
  sims.push(map2(sI, sQ, (a, b) => (a + b) / 2));
  weights.push(map2(w0, w1, (a, b) => (a + b) / 2));
  const sigmoid = (v) => 1 / (1 + Math.exp(-v));
  let top = 0;
  let bottom = 0;
  for (let k = 0; k < 3; k++) {
    for (let i = 0; i < n; i++) {
      top += sigmoid(sims[k].d[i] * alpha) * weights[k].d[i];
      bottom += weights[k].d[i];
    }
  }
  const eps = Number.EPSILON;
  const score = (top + eps) / (bottom + eps);
  return (Math.log(score / (1 - score)) / alpha) ** 2;
}

// ---------------------------------------------------------------- MDSI

/** Mean Deviation Similarity Index (Nafchi et al. 2016), as piq.mdsi(x = B, y = A). 0 = identical; lower is more similar. */
export function mdsi(A, B, { c1 = 140, c2 = 55, c3 = 550, alpha = 0.6, q = 0.25, o = 0.25 } = {}) {
  const k = downFactor(A.w, A.h);
  const prep = (img) => {
    let ps = channels(img).map((d) => plane(img.w, img.h, d));
    if (Math.floor(k / 2)) ps = ps.map((p) => pad(p, Math.floor((k - 1) / 2), Math.floor(k / 2), Math.floor((k - 1) / 2), Math.floor(k / 2)));
    ps = ps.map((p) => avgPool(p, k));
    const n = ps[0].d.length;
    return LHM.map(([a, b, c]) => plane(ps[0].w, ps[0].h, Float64Array.from({ length: n }, (_, i) => a * ps[0].d[i] + b * ps[1].d[i] + c * ps[2].d[i])));
  };
  const [Lx, Hx, Mx] = prep(B);
  const [Ly, Hy, My] = prep(A);
  const gx = gradientMap(Lx, PREWITT);
  const gy = gradientMap(Ly, PREWITT);
  const ga = gradientMap(map2(Lx, Ly, (a, b) => (a + b) / 2), PREWITT);
  const gsXY = similarity(gx, gy, c1);
  const gsXA = similarity(gx, ga, c2);
  const gsYA = similarity(gy, ga, c2);
  const n = Lx.d.length;
  // gcs may be negative: raise to the power q as a complex number
  const re = new Float64Array(n);
  const im = new Float64Array(n);
  let mr = 0;
  let mi = 0;
  for (let i = 0; i < n; i++) {
    const gs = gsXY.d[i] + gsXA.d[i] - gsYA.d[i];
    const cs = (2 * (Hx.d[i] * Hy.d[i] + Mx.d[i] * My.d[i]) + c3) / (Hx.d[i] ** 2 + Hy.d[i] ** 2 + Mx.d[i] ** 2 + My.d[i] ** 2 + c3);
    const gcs = alpha * gs + (1 - alpha) * cs;
    const r = Math.abs(gcs) ** q;
    const phi = (gcs < 0 || Object.is(gcs, -0) ? Math.PI : 0) * q;
    re[i] = r * Math.cos(phi);
    im[i] = r * Math.sin(phi);
    mr += re[i];
    mi += im[i];
  }
  mr /= n;
  mi /= n;
  let s = 0;
  for (let i = 0; i < n; i++) s += Math.hypot(re[i] - mr, im[i] - mi);
  return (s / n) ** o;
}

// ---------------------------------------------------------------- MS-GMSD

function gmsdOne(x, y, t, alpha) {
  const gms = similarity(gradientMap(x, PREWITT), gradientMap(y, PREWITT), t, alpha);
  const m = sum(gms) / gms.d.length;
  let s = 0;
  for (const v of gms.d) s += (v - m) ** 2;
  return Math.sqrt(s / gms.d.length);
}

/** Multi-scale gradient magnitude similarity deviation (Zhang et al. 2017), as piq.multi_scale_gmsd. 0 = identical. */
export function msGmsd(A, B, { t = 170, alpha = 0.5 } = {}) {
  const weights = [0.096, 0.596, 0.289, 0.019];
  if (A.w < 2 ** weights.length + 1 || A.h < 2 ** weights.length + 1) return null;
  let x = transform(B, [YIQ[0]])[0];
  let y = transform(A, [YIQ[0]])[0];
  let total = 0;
  for (let s = 0; s < weights.length; s++) {
    if (s > 0) {
      const d = Math.max(x.h % 2, x.w % 2);
      x = avgPool(pad(x, 0, d, 0, d), 2);
      y = avgPool(pad(y, 0, d, 0, d), 2);
    }
    total += weights[s] * gmsdOne(x, y, t, alpha) ** 2;
  }
  return Math.sqrt(total);
}

// ---------------------------------------------------------------- FSIM

/** piq.functional.get_meshgrid for (H, W) */
function meshgrid(H, W) {
  const ax = (n) => Array.from({ length: n }, (_, i) => (n % 2 ? (i - (n - 1) / 2) / (n - 1) : (i - n / 2) / n));
  return [ax(H), ax(W)];
}

/** ifftshift of an H×W array (roll by −floor(n/2) on each axis). */
function ifftshift(a, H, W) {
  const out = new Float64Array(H * W);
  const sy = Math.floor(H / 2);
  const sx = Math.floor(W / 2);
  for (let y = 0; y < H; y++) for (let x = 0; x < W; x++) out[y * W + x] = a[((y + sy) % H) * W + ((x + sx) % W)];
  return out;
}

const fsimFilters = new Map();

function logGaborBank(H, W, scales = 4, orientations = 4, minLength = 6, mult = 2, sigmaF = 0.55, deltaTheta = 1.2) {
  const key = `${H}x${W}`;
  if (fsimFilters.has(key)) return fsimFilters.get(key);
  const thetaSigma = Math.PI / (orientations * deltaTheta);
  const [gx, gy] = meshgrid(H, W);
  const radius0 = new Float64Array(H * W);
  const theta0 = new Float64Array(H * W);
  const lp0 = new Float64Array(H * W);
  for (let y = 0; y < H; y++) {
    for (let x = 0; x < W; x++) {
      const r = Math.sqrt(gx[y] ** 2 + gy[x] ** 2);
      radius0[y * W + x] = r;
      theta0[y * W + x] = Math.atan2(-gy[x], gx[y]);
      lp0[y * W + x] = 1 / (1 + (r / 0.45) ** 30);
    }
  }
  const lp = ifftshift(lp0, H, W);
  const radius = ifftshift(radius0, H, W);
  const theta = ifftshift(theta0, H, W);
  radius[0] = 1;
  const logGabor = [];
  for (let s = 0; s < scales; s++) {
    const omega0 = 1 / (minLength * mult ** s);
    const g = Float64Array.from(radius, (r, i) => Math.exp(-(Math.log(r / omega0) ** 2) / (2 * Math.log(sigmaF) ** 2)) * lp[i]);
    g[0] = 0;
    logGabor.push(g);
  }
  const filters = [];
  for (let o = 0; o < orientations; o++) {
    const angl = (o * Math.PI) / orientations;
    const spread = Float64Array.from(theta, (t) => {
      const ds = Math.sin(t) * Math.cos(angl) - Math.cos(t) * Math.sin(angl);
      const dc = Math.cos(t) * Math.cos(angl) + Math.sin(t) * Math.sin(angl);
      const dt = Math.abs(Math.atan2(ds, dc));
      return Math.exp(-(dt * dt) / (2 * thetaSigma * thetaSigma));
    });
    for (let s = 0; s < scales; s++) filters.push(Float64Array.from(spread, (v, i) => v * logGabor[s][i]));
  }
  // spatial filters (real part of their inverse FFT, times √(HW)) for the noise model
  const spatial = filters.map((f) => {
    const re = Float64Array.from(f);
    const im = new Float64Array(H * W);
    fft2(re, im, W, H, true);
    return re.map((v) => v * Math.sqrt(H * W));
  });
  const bank = { filters, spatial, scales, orientations };
  fsimFilters.set(key, bank);
  return bank;
}

function torchMedianLower(values) {
  const a = Float64Array.from(values).sort();
  return a[(a.length - 1) >> 1];
}

/** Phase congruency (Kovesi), as piq's _phase_congruency. */
function phaseCongruency(lum, bank, k = 2) {
  const { w: W, h: H } = lum;
  const { filters, spatial, scales, orientations } = bank;
  const EPS = Number.EPSILON;
  const fr = Float64Array.from(lum.d);
  const fi = new Float64Array(H * W);
  fft2(fr, fi, W, H);
  const n = H * W;
  const energyAll = new Float64Array(n);
  const anAll = new Float64Array(n);
  const er = new Float64Array(n);
  const ei = new Float64Array(n);
  for (let o = 0; o < orientations; o++) {
    const even = [];
    const odd = [];
    for (let s = 0; s < scales; s++) {
      const f = filters[o * scales + s];
      for (let i = 0; i < n; i++) {
        er[i] = fr[i] * f[i];
        ei[i] = fi[i] * f[i];
      }
      fft2(er, ei, W, H, true);
      even.push(Float64Array.from(er));
      odd.push(Float64Array.from(ei));
    }
    const sumE = new Float64Array(n);
    const sumO = new Float64Array(n);
    for (let s = 0; s < scales; s++) {
      for (let i = 0; i < n; i++) {
        sumE[i] += even[s][i];
        sumO[i] += odd[s][i];
        anAll[i] += Math.hypot(even[s][i], odd[s][i]);
      }
    }
    const energy = new Float64Array(n);
    for (let i = 0; i < n; i++) {
      const xe = Math.sqrt(sumE[i] ** 2 + sumO[i] ** 2) + EPS;
      const me = sumE[i] / xe;
      const mo = sumO[i] / xe;
      for (let s = 0; s < scales; s++) energy[i] += even[s][i] * me + odd[s][i] * mo - Math.abs(even[s][i] * mo - odd[s][i] * me);
    }
    // noise threshold from the smallest scale
    const e2 = Float64Array.from({ length: n }, (_, i) => even[0][i] ** 2 + odd[0][i] ** 2);
    const meanE2n = -torchMedianLower(e2) / Math.log(0.5);
    let emN = 0;
    for (const v of filters[o * scales]) emN += v * v;
    const noisePower = meanE2n / emN;
    let sumAn2 = 0;
    let sumAiAj = 0;
    for (let s = 0; s < scales; s++) for (const v of spatial[o * scales + s]) sumAn2 += v * v;
    for (let s = 0; s < scales - 1; s++) {
      for (let t = s + 1; t < scales; t++) {
        const a = spatial[o * scales + s];
        const b = spatial[o * scales + t];
        for (let i = 0; i < n; i++) sumAiAj += a[i] * b[i];
      }
    }
    const noiseEnergy2 = 2 * noisePower * sumAn2 + 4 * noisePower * sumAiAj;
    const tau = Math.sqrt(noiseEnergy2 / 2);
    const T = (tau * Math.sqrt(Math.PI / 2) + k * Math.sqrt((2 - Math.PI / 2) * tau * tau)) / 1.7;
    for (let i = 0; i < n; i++) energyAll[i] += Math.max(energy[i] - T, 0);
  }
  return plane(W, H, Float64Array.from(energyAll, (v, i) => (v + EPS) / (anAll[i] + EPS)));
}

/**
 * Feature Similarity Index with colour (FSIMc; Zhang, Zhang, Mou & Zhang
 * 2011), as piq.fsim: phase congruency and gradient magnitude compared
 * point by point, weighted by how much structure each point has. 1 = identical.
 */
export function fsim(A, B, { withMap = false } = {}) {
  const k = downFactor(A.w, A.h);
  const prep = (img) => transform(img, YIQ).map((p) => avgPool(p, k));
  const [Yx, Ix, Qx] = prep(B);
  const [Yy, Iy, Qy] = prep(A);
  if (Yx.w < 8 || Yx.h < 8) return null;
  const bank = logGaborBank(Yx.h, Yx.w);
  const pcx = phaseCongruency(Yx, bank);
  const pcy = phaseCongruency(Yy, bank);
  const gmx = gradientMap(Yx, SCHARR);
  const gmy = gradientMap(Yy, SCHARR);
  const PC = similarity(pcx, pcy, 0.85);
  const GM = similarity(gmx, gmy, 160);
  const SI = similarity(Ix, Iy, 200);
  const SQ = similarity(Qx, Qy, 200);
  let top = 0;
  let bottom = 0;
  const map = withMap ? new Float64Array(Yx.d.length) : null;
  for (let i = 0; i < Yx.d.length; i++) {
    const pcMax = pcx.d[i] > pcy.d[i] ? pcx.d[i] : pcy.d[i];
    const local = GM.d[i] * PC.d[i] * Math.abs(SI.d[i] * SQ.d[i]) ** 0.03;
    top += local * pcMax;
    bottom += pcMax;
    if (map) map[i] = local;
  }
  const value = top / bottom;
  return withMap ? { value, map, mapW: Yx.w, mapH: Yx.h } : value;
}

// ---------------------------------------------------------------- DSS

/** Orthonormal DCT-II matrix, as piq's _dct_matrix. */
function dctMatrix(n) {
  const m = new Float64Array(n * n);
  for (let j = 0; j < n; j++) m[j] = Math.sqrt(1 / n);
  for (let p = 1; p < n; p++) for (let j = 0; j < n; j++) m[p * n + j] = Math.sqrt(2 / n) * Math.cos((Math.PI / (2 * n)) * p * (2 * j + 1));
  return m;
}

/** 8×8 block DCT of a plane; coefficient (m, n) of each block stays at that position inside the block. */
function blockDct(p, n) {
  const C = dctMatrix(n);
  const out = plane(p.w, p.h);
  const t = new Float64Array(n * n);
  for (let by = 0; by < p.h; by += n) {
    for (let bx = 0; bx < p.w; bx += n) {
      // t = C · block, then out = t · Cᵀ
      for (let i = 0; i < n; i++) {
        for (let j = 0; j < n; j++) {
          let s = 0;
          for (let k = 0; k < n; k++) s += C[i * n + k] * p.d[(by + k) * p.w + bx + j];
          t[i * n + j] = s;
        }
      }
      for (let i = 0; i < n; i++) {
        for (let j = 0; j < n; j++) {
          let s = 0;
          for (let k = 0; k < n; k++) s += t[i * n + k] * C[j * n + k];
          out.d[(by + i) * p.w + bx + j] = s;
        }
      }
    }
  }
  return out;
}

function subbandSimilarity(x, y, first, kernel, percentile) {
  const c = first ? 1000 : 300;
  const blur = (q) => conv(pad(q, 1, 1, 1, 1), kernel, 3, 3);
  const mx = blur(x);
  const my = blur(y);
  const sxx = blur(map1(x, (v) => v * v));
  const syy = blur(map1(y, (v) => v * v));
  const n = x.d.length;
  const k = roundHalfEven(percentile * n);
  const left = new Float64Array(n);
  const vxx = new Float64Array(n);
  const vyy = new Float64Array(n);
  for (let i = 0; i < n; i++) {
    vxx[i] = Math.max(0, sxx.d[i] - mx.d[i] ** 2);
    vyy[i] = Math.max(0, syy.d[i] - my.d[i] ** 2);
    left[i] = (2 * Math.sqrt(vxx[i] * vyy[i]) + c) / (vxx[i] + vyy[i] + c);
  }
  const lowMean = (a) => {
    a.sort();
    let s = 0;
    for (let i = 0; i < k; i++) s += a[i];
    return s / k;
  };
  let sim = lowMean(left);
  if (first) {
    const sxy = blur(map2(x, y, (a, b) => a * b));
    const right = Float64Array.from({ length: n }, (_, i) => (sxy.d[i] - mx.d[i] * my.d[i] + c) / (Math.sqrt(vxx[i] * vyy[i]) + c));
    sim *= lowMean(right);
  }
  return sim;
}

/**
 * DCT Subbands Similarity (Balanov, Schwartz, Moshe & Peleg 2015), as piq.dss:
 * compares the two images band by band in the 8×8 DCT used by JPEG, focusing on
 * the worst-matching 5% of places in each band. 1 = identical.
 */
export function dss(A, B, { size = 8, sigmaWeight = 1.55, percentile = 0.05 } = {}) {
  let x = transform(B, [YIQ[0]])[0];
  let y = transform(A, [YIQ[0]])[0];
  const rows = size * Math.floor(x.h / size);
  const cols = size * Math.floor(x.w / size);
  // each DCT band must have enough points for its worst 5% to hold at least one
  if (rows < 3 * size || cols < 3 * size || roundHalfEven(percentile * (rows / size) * (cols / size)) < 1) return null;
  const crop = (p) => {
    const out = plane(cols, rows);
    for (let r = 0; r < rows; r++) out.d.set(p.d.subarray(r * p.w, r * p.w + cols), r * cols);
    return out;
  };
  x = blockDct(crop(x), size);
  y = blockDct(crop(y), size);
  const kernel = gaussian(3, 1.5);
  const weight = new Float64Array(size * size);
  for (let m = 0; m < size; m++) for (let n = 0; n < size; n++) weight[m * size + n] = Math.exp(-((m + 0.5) ** 2 + (n + 0.5) ** 2) / (2 * sigmaWeight ** 2));
  const sims = new Float64Array(size * size);
  const band = (p, m, n) => {
    const out = plane(cols / size, rows / size);
    for (let r = 0; r < out.h; r++) for (let c = 0; c < out.w; c++) out.d[r * out.w + c] = p.d[(r * size + m) * cols + c * size + n];
    return out;
  };
  for (let m = 0; m < size; m++) {
    for (let n = 0; n < size; n++) {
      if (weight[m * size + n] < 1e-2) {
        weight[m * size + n] = 0;
        continue;
      }
      sims[m * size + n] = subbandSimilarity(band(x, m, n), band(y, m, n), m === 0 && n === 0, kernel, percentile);
    }
  }
  const total = weight.reduce((a, b) => a + b, 0);
  let score = 0;
  for (let i = 0; i < weight.length; i++) score += sims[i] * (weight[i] / total + Number.EPSILON);
  return score;
}

// ---------------------------------------------------------------- VSI

/** torch.nn.functional.interpolate(mode='bilinear') of a plane, either corner convention, no antialiasing. */
function bilinear(p, W, H, alignCorners) {
  const src = (i, inLen, outLen) => {
    if (alignCorners) return outLen > 1 ? (i * (inLen - 1)) / (outLen - 1) : 0;
    return Math.max(0, (i + 0.5) * (inLen / outLen) - 0.5);
  };
  const out = plane(W, H);
  for (let y = 0; y < H; y++) {
    const sy = src(y, p.h, H);
    const y0 = Math.floor(sy);
    const y1 = Math.min(y0 + 1, p.h - 1);
    const ly = sy - y0;
    for (let x = 0; x < W; x++) {
      const sx = src(x, p.w, W);
      const x0 = Math.floor(sx);
      const x1 = Math.min(x0 + 1, p.w - 1);
      const lx = sx - x0;
      const top = (1 - lx) * p.d[y0 * p.w + x0] + lx * p.d[y0 * p.w + x1];
      const bottom = (1 - lx) * p.d[y1 * p.w + x0] + lx * p.d[y1 * p.w + x1];
      out.d[y * W + x] = (1 - ly) * top + ly * bottom;
    }
  }
  return out;
}

/** piq's rgb2lab (sRGB → XYZ, then L*a*b* against the D50 white, as piq does by default) of three 0..255 planes. */
function piqLab([r, g, b]) {
  const n = r.d.length;
  const lin = (v) => {
    const t = v / 255;
    return t <= 0.04045 ? t / 12.92 : ((t + 0.055) / 1.055) ** 2.4;
  };
  const white = [0.9642119944211994, 1, 0.8251882845188288];
  const f = (t) => (t > 0.008856 ? Math.cbrt(t) : (903.3 * t + 16) / 116);
  const L = plane(r.w, r.h);
  const A = plane(r.w, r.h);
  const Bp = plane(r.w, r.h);
  for (let i = 0; i < n; i++) {
    const R = lin(r.d[i]);
    const G = lin(g.d[i]);
    const B = lin(b.d[i]);
    const X = f((0.4124564 * R + 0.3575761 * G + 0.1804375 * B) / white[0]);
    const Y = f((0.2126729 * R + 0.7151522 * G + 0.072175 * B) / white[1]);
    const Z = f((0.0193339 * R + 0.119192 * G + 0.9503041 * B) / white[2]);
    L.d[i] = 116 * Y - 16;
    A.d[i] = 500 * (X - Y);
    Bp.d[i] = 200 * (Y - Z);
  }
  return [L, A, Bp];
}

/** SDSP visual saliency (Zhang, Gong, Zhang & Zhang 2013) as piq.functional's sdsp, back at the image's size. */
function sdsp(rgbPlanes, { omega0 = 0.021, sigmaF = 1.34, sigmaD = 145, sigmaC = 0.001 } = {}) {
  const S = 256;
  const small = rgbPlanes.map((p) => bilinear(p, S, S, false));
  const lab = piqLab(small);
  const [gx, gy] = meshgrid(S, S);
  const r0 = new Float64Array(S * S);
  for (let y = 0; y < S; y++) for (let x = 0; x < S; x++) {
    const r = Math.sqrt(gx[y] ** 2 + gy[x] ** 2);
    r0[y * S + x] = r <= 0.5 ? r : 0;
  }
  const r = ifftshift(r0, S, S);
  r[0] = 1;
  const lg = Float64Array.from(r, (v) => Math.exp(-(Math.log(v / omega0) ** 2) / (2 * sigmaF * sigmaF)));
  lg[0] = 0;
  const sf = new Float64Array(S * S);
  for (const ch of lab) {
    const re = Float64Array.from(ch.d);
    const im = new Float64Array(S * S);
    fft2(re, im, S, S);
    for (let i = 0; i < S * S; i++) {
      re[i] *= lg[i];
      im[i] *= lg[i];
    }
    fft2(re, im, S, S, true);
    for (let i = 0; i < S * S; i++) sf[i] += re[i] * re[i];
  }
  const eps = Number.EPSILON;
  const range = (d) => {
    let lo = Infinity;
    let hi = -Infinity;
    for (const v of d) {
      lo = Math.min(lo, v);
      hi = Math.max(hi, v);
    }
    return [lo, hi];
  };
  const [aLo, aHi] = range(lab[1].d);
  const [bLo, bHi] = range(lab[2].d);
  const vs = plane(S, S);
  for (let y = 0; y < S; y++) {
    for (let x = 0; x < S; x++) {
      const i = y * S + x;
      const cx = gx[y] * S + 1;
      const cy = gy[x] * S + 1;
      const sd = Math.exp(-(cx * cx + cy * cy) / (sigmaD * sigmaD));
      const na = (lab[1].d[i] - aLo) / (aHi - aLo + eps);
      const nb = (lab[2].d[i] - bLo) / (bHi - bLo + eps);
      const sc = 1 - Math.exp(-(na * na + nb * nb) / (sigmaC * sigmaC));
      vs.d[i] = Math.sqrt(sf[i]) * sd * sc;
    }
  }
  const back = bilinear(vs, rgbPlanes[0].w, rgbPlanes[0].h, true);
  const [lo, hi] = range(back.d);
  return map1(back, (v) => (v - lo) / (hi - lo + eps));
}

/**
 * Visual Saliency-induced Index (Zhang, Shen & Li 2014), as piq.vsi: where
 * people would look (SDSP saliency) weights how alike gradients and colours
 * are. 1 = identical.
 */
export function vsi(A, B, { c1 = 1.27, c2 = 386, c3 = 130, alpha = 0.4, beta = 0.02 } = {}) {
  if (A.w < 2 || A.h < 2) return null;
  const px = channels(B).map((d) => plane(B.w, B.h, d));
  const py = channels(A).map((d) => plane(A.w, A.h, d));
  let vsx = sdsp(px);
  let vsy = sdsp(py);
  const LMN = [
    [0.06, 0.63, 0.27],
    [0.3, 0.04, -0.35],
    [0.34, -0.6, 0.17],
  ];
  let lx = transform(B, LMN);
  let ly = transform(A, LMN);
  const k = downFactor(A.w, A.h);
  const padK = Math.floor(k / 2);
  const prep = (p) => avgPool(padK ? pad(p, padK, Math.floor((k - 1) / 2), padK, Math.floor((k - 1) / 2), 'replicate') : p, k);
  vsx = prep(vsx);
  vsy = prep(vsy);
  lx = lx.map(prep);
  ly = ly.map(prep);
  const sVs = similarity(vsx, vsy, c1);
  const sGm = similarity(gradientMap(lx[0], SCHARR), gradientMap(ly[0], SCHARR), c2);
  const sM = similarity(lx[1], ly[1], c3);
  const sN = similarity(lx[2], ly[2], c3);
  let top = 0;
  let bottom = 0;
  for (let i = 0; i < sVs.d.length; i++) {
    const sc = sM.d[i] * sN.d[i];
    const scPow = Math.abs(sc) ** beta * Math.cos((sc < 0 ? Math.PI : 0) * beta);
    const vmax = Math.max(vsx.d[i], vsy.d[i]);
    top += sVs.d[i] * sGm.d[i] ** alpha * scPow * vmax;
    bottom += vmax;
  }
  return (top + Number.EPSILON) / (bottom + Number.EPSILON);
}

// ---------------------------------------------------------------- NMI

/**
 * Normalised mutual information (Studholme et al. 1999) of two grey planes,
 * as skimage.metrics.normalized_mutual_information(bins=100):
 * (H(A) + H(B)) / H(A, B), from 1 (unrelated) to 2 (one determines the other).
 */
export function nmi(ga, gb, bins = 100) {
  const edges = (arr) => {
    let lo = Infinity;
    let hi = -Infinity;
    for (const v of arr) {
      lo = Math.min(lo, v);
      hi = Math.max(hi, v);
    }
    if (lo === hi) {
      lo -= 0.5;
      hi += 0.5;
    }
    const step = (hi - lo) / bins;
    const e = Float64Array.from({ length: bins + 1 }, (_, k) => k * step + lo);
    e[bins] = hi;
    return e;
  };
  const binOf = (e) => {
    const cache = new Int32Array(256).fill(-1);
    return (v) => {
      if (cache[v] >= 0) return cache[v];
      let lo = 0;
      let hi = e.length;
      // numpy searchsorted(side='right') − 1, with the last edge folded into the last bin
      while (lo < hi) {
        const mid = (lo + hi) >> 1;
        if (e[mid] <= v) lo = mid + 1;
        else hi = mid;
      }
      let b = lo - 1;
      if (v === e[e.length - 1]) b = bins - 1;
      cache[v] = b;
      return b;
    };
  };
  const ba = binOf(edges(ga));
  const bb = binOf(edges(gb));
  const joint = new Float64Array(bins * bins);
  for (let i = 0; i < ga.length; i++) joint[ba(ga[i]) * bins + bb(gb[i])]++;
  const entropy = (h) => {
    let t = 0;
    for (const v of h) t += v;
    let e = 0;
    for (const v of h) if (v > 0) e -= (v / t) * Math.log(v / t);
    return e;
  };
  const ma = new Float64Array(bins);
  const mb = new Float64Array(bins);
  for (let i = 0; i < bins; i++) {
    for (let j = 0; j < bins; j++) {
      ma[i] += joint[i * bins + j];
      mb[j] += joint[i * bins + j];
    }
  }
  return (entropy(ma) + entropy(mb)) / entropy(joint);
}
