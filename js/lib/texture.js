// Texture and shape descriptors, each computed on one image on its own and
// then compared: local binary patterns, histograms of oriented gradients,
// grey-level co-occurrence (Haralick) statistics, Gabor filter energies and
// the GIST scene descriptor. LBP, HOG and GLCM follow scikit-image exactly;
// GIST follows Oliva & Torralba's LMgist.m.

import { fft2, goodSize } from './fft.js';
import { cropImg, resizeImg } from './pixels.js';

// ---------------------------------------------------------------- LBP

/**
 * Histogram of "uniform" local binary patterns (Ojala et al. 2002) with 8
 * neighbours at radius 1, as skimage.feature.local_binary_pattern(..., 'uniform'):
 * each pixel is coded by which of its neighbours are at least as bright,
 * and the 10 codes are counted. Input: 8-bit grey plane.
 */
export function lbpHistogram(gray, w, h, P = 8, R = 1) {
  const rp = [];
  const cp = [];
  for (let i = 0; i < P; i++) {
    rp.push(Math.round(-R * Math.sin((2 * Math.PI * i) / P) * 1e5) / 1e5);
    cp.push(Math.round(R * Math.cos((2 * Math.PI * i) / P) * 1e5) / 1e5);
  }
  const at = (r, c) => (r >= 0 && c >= 0 && r < h && c < w ? gray[r * w + c] : 0);
  const sample = (r, c) => {
    const r0 = Math.floor(r);
    const c0 = Math.floor(c);
    const r1 = Math.ceil(r);
    const c1 = Math.ceil(c);
    const dr = r - r0;
    const dc = c - c0;
    const top = (1 - dc) * at(r0, c0) + dc * at(r0, c1);
    const bottom = (1 - dc) * at(r1, c0) + dc * at(r1, c1);
    return (1 - dr) * top + dr * bottom;
  };
  const hist = new Float64Array(P + 2);
  const bits = new Uint8Array(P);
  for (let r = 0; r < h; r++) {
    for (let c = 0; c < w; c++) {
      const centre = gray[r * w + c];
      for (let i = 0; i < P; i++) bits[i] = sample(r + rp[i], c + cp[i]) - centre >= 0 ? 1 : 0;
      // skimage counts changes between consecutive neighbours, not around the circle
      let changes = 0;
      for (let i = 0; i < P - 1; i++) if (bits[i] !== bits[i + 1]) changes++;
      let code = P + 1;
      if (changes <= 2) {
        code = 0;
        for (let i = 0; i < P; i++) code += bits[i];
      }
      hist[code]++;
    }
  }
  for (let i = 0; i < hist.length; i++) hist[i] /= w * h;
  return hist;
}

/** Chi-square distance between two normalised histograms, 0 (same) .. 1 (disjoint). */
export function chiSquare(a, b) {
  let s = 0;
  for (let i = 0; i < a.length; i++) if (a[i] + b[i] > 0) s += (a[i] - b[i]) ** 2 / (a[i] + b[i]);
  return s / 2;
}

// ---------------------------------------------------------------- HOG

/**
 * Histogram of oriented gradients (Dalal & Triggs 2005) as skimage.feature.hog
 * with 9 orientations, 16×16-pixel cells, 2×2-cell blocks and L2-Hys
 * normalisation. Input: 8-bit grey plane (the worker uses 128×128).
 */
export function hogDescriptor(gray, w, h, { orientations = 9, cell = 16, block = 2 } = {}) {
  const gr = new Float64Array(w * h);
  const gc = new Float64Array(w * h);
  for (let y = 1; y < h - 1; y++) for (let x = 0; x < w; x++) gr[y * w + x] = gray[(y + 1) * w + x] - gray[(y - 1) * w + x];
  for (let y = 0; y < h; y++) for (let x = 1; x < w - 1; x++) gc[y * w + x] = gray[y * w + x + 1] - gray[y * w + x - 1];
  const mag = new Float64Array(w * h);
  const ori = new Float64Array(w * h);
  for (let i = 0; i < w * h; i++) {
    mag[i] = Math.hypot(gc[i], gr[i]);
    let a = (Math.atan2(gr[i], gc[i]) * 180) / Math.PI;
    a %= 180;
    if (a < 0) a += 180;
    ori[i] = a;
  }
  const nr = Math.floor(h / cell);
  const nc = Math.floor(w / cell);
  const per = 180 / orientations;
  const hist = new Float64Array(nr * nc * orientations);
  const half = Math.floor(cell / 2);
  for (let o = 0; o < orientations; o++) {
    const start = per * (o + 1);
    const end = per * o;
    for (let ri = 0; ri < nr; ri++) {
      for (let ci = 0; ci < nc; ci++) {
        const r = half + ri * cell;
        const c = half + ci * cell;
        // skimage accumulates each cell in single precision
        let total = 0;
        for (let dy = -half; dy < Math.floor((cell + 1) / 2); dy++) {
          const y = r + dy;
          if (y < 0 || y >= h) continue;
          for (let dx = -half; dx < Math.floor((cell + 1) / 2); dx++) {
            const x = c + dx;
            if (x < 0 || x >= w) continue;
            const a = ori[y * w + x];
            if (a >= start || a < end) continue;
            total = Math.fround(total + mag[y * w + x]);
          }
        }
        hist[(ri * nc + ci) * orientations + o] = Math.fround(total / (cell * cell));
      }
    }
  }
  const br = nr - block + 1;
  const bc = nc - block + 1;
  if (br < 1 || bc < 1) return null;
  const out = new Float64Array(br * bc * block * block * orientations);
  const eps = 1e-5;
  let k = 0;
  const v = new Float64Array(block * block * orientations);
  for (let r = 0; r < br; r++) {
    for (let c = 0; c < bc; c++) {
      let j = 0;
      for (let y = 0; y < block; y++) for (let x = 0; x < block; x++) for (let o = 0; o < orientations; o++) v[j++] = hist[((r + y) * nc + c + x) * orientations + o];
      let s = 0;
      for (const t of v) s += t * t;
      let n = Math.sqrt(s + eps * eps);
      for (let i = 0; i < v.length; i++) v[i] = Math.min(v[i] / n, 0.2);
      s = 0;
      for (const t of v) s += t * t;
      n = Math.sqrt(s + eps * eps);
      for (let i = 0; i < v.length; i++) out[k++] = v[i] / n;
    }
  }
  return out;
}

export function cosine(a, b) {
  let ab = 0;
  let aa = 0;
  let bb = 0;
  for (let i = 0; i < a.length; i++) {
    ab += a[i] * b[i];
    aa += a[i] * a[i];
    bb += b[i] * b[i];
  }
  return aa && bb ? ab / Math.sqrt(aa * bb) : 0;
}

// ---------------------------------------------------------------- GLCM

/**
 * Haralick texture statistics (Haralick, Shanmugam & Dinstein 1973) from
 * grey-level co-occurrence matrices, as skimage's graycomatrix (32 levels,
 * distances 1 and 2, four directions, symmetric, normalised) and
 * graycoprops: contrast, dissimilarity, homogeneity, energy, correlation and
 * ASM for each distance, averaged over directions (12 numbers).
 */
export function glcmFeatures(gray, w, h, levels = 32) {
  const shift = Math.log2(256 / levels);
  const q = Uint8Array.from(gray, (v) => v >> shift);
  const angles = [0, Math.PI / 4, Math.PI / 2, (3 * Math.PI) / 4];
  const distances = [1, 2];
  const props = ['contrast', 'dissimilarity', 'homogeneity', 'energy', 'correlation', 'ASM'];
  const out = new Float64Array(props.length * distances.length);
  const cRound = (v) => Math.sign(v) * Math.floor(Math.abs(v) + 0.5); // C round(): half away from zero
  distances.forEach((d, di) => {
    const acc = new Float64Array(props.length);
    for (const angle of angles) {
      const orow = cRound(Math.sin(angle) * d);
      const ocol = cRound(Math.cos(angle) * d);
      const P = new Float64Array(levels * levels);
      for (let r = Math.max(0, -orow); r < Math.min(h, h - orow); r++) {
        for (let c = Math.max(0, -ocol); c < Math.min(w, w - ocol); c++) {
          const i = q[r * w + c];
          const j = q[(r + orow) * w + c + ocol];
          P[i * levels + j]++;
          P[j * levels + i]++; // symmetric
        }
      }
      let sum = 0;
      for (const v of P) sum += v;
      if (sum === 0) sum = 1;
      for (let i = 0; i < P.length; i++) P[i] /= sum;
      let contrast = 0;
      let dissim = 0;
      let homog = 0;
      let asm = 0;
      let mi = 0;
      let mj = 0;
      for (let i = 0; i < levels; i++) {
        for (let j = 0; j < levels; j++) {
          const p = P[i * levels + j];
          contrast += p * (i - j) ** 2;
          dissim += p * Math.abs(i - j);
          homog += p / (1 + (i - j) ** 2);
          asm += p * p;
          mi += i * p;
          mj += j * p;
        }
      }
      let vi = 0;
      let vj = 0;
      let cov = 0;
      for (let i = 0; i < levels; i++) {
        for (let j = 0; j < levels; j++) {
          const p = P[i * levels + j];
          vi += p * (i - mi) ** 2;
          vj += p * (j - mj) ** 2;
          cov += p * (i - mi) * (j - mj);
        }
      }
      const si = Math.sqrt(vi);
      const sj = Math.sqrt(vj);
      const corr = si < 1e-15 || sj < 1e-15 ? 1 : cov / (si * sj);
      [contrast, dissim, homog, Math.sqrt(asm), corr, asm].forEach((v, k) => (acc[k] += v / angles.length));
    }
    acc.forEach((v, k) => (out[k * distances.length + di] = v));
  });
  return out;
}

/** Mean Canberra distance |a−b| / (|a|+|b|): scale-free, 0 (same) .. 1. */
export function canberra(a, b) {
  let s = 0;
  for (let i = 0; i < a.length; i++) {
    const d = Math.abs(a[i]) + Math.abs(b[i]);
    if (d > 0) s += Math.abs(a[i] - b[i]) / d;
  }
  return s / a.length;
}

// ---------------------------------------------------------------- Gabor

/** skimage.filters.gabor_kernel(frequency, theta) with bandwidth 1: [re, im, size]. */
export function gaborKernel(frequency, theta, nStds = 3) {
  const b = 1;
  const sigma = ((1 / Math.PI) * Math.sqrt(Math.log(2) / 2) * (2 ** b + 1)) / (2 ** b - 1) / frequency;
  const ct = Math.cos(theta);
  const st = Math.sin(theta);
  const x0 = Math.ceil(Math.max(Math.abs(nStds * sigma * ct), Math.abs(nStds * sigma * st), 1));
  const y0 = Math.ceil(Math.max(Math.abs(nStds * sigma * ct), Math.abs(nStds * sigma * st), 1));
  const kw = 2 * x0 + 1;
  const kh = 2 * y0 + 1;
  const re = new Float64Array(kw * kh);
  const im = new Float64Array(kw * kh);
  for (let y = -y0; y <= y0; y++) {
    for (let x = -x0; x <= x0; x++) {
      const rotx = x * ct + y * st;
      const roty = -x * st + y * ct;
      const g = Math.exp(-0.5 * ((rotx * rotx) / (sigma * sigma) + (roty * roty) / (sigma * sigma))) / (2 * Math.PI * sigma * sigma);
      const a = 2 * Math.PI * frequency * rotx;
      re[(y + y0) * kw + x + x0] = g * Math.cos(a);
      im[(y + y0) * kw + x + x0] = g * Math.sin(a);
    }
  }
  return { re, im, kw, kh, x0, y0 };
}

const GABOR_BANK = (() => {
  const bank = [];
  for (const f of [0.05, 0.1, 0.2, 0.4]) for (let o = 0; o < 6; o++) bank.push(gaborKernel(f, (o * Math.PI) / 6));
  return bank;
})();

/**
 * Gabor texture energy (Manjunath & Ma 1996): the mean and spread of the
 * response of 24 Gabor filters (4 frequencies × 6 orientations) — 48 numbers.
 * Convolution with mirrored edges, as scipy.ndimage.convolve(mode='reflect').
 * Input: grey plane scaled to 0..1.
 */
export function gaborFeatures(gray, w, h) {
  // mirror the image outwards by at least the kernel radius, to a size the FFT handles fast
  const K = Math.max(...GABOR_BANK.map((k) => Math.max(k.x0, k.y0)));
  const W = goodSize(w + 2 * K);
  const H = goodSize(h + 2 * K);
  const refl = (i, n) => {
    // half-sample symmetric: d c b a | a b c d | d c b a
    const period = 2 * n;
    i = ((i % period) + period) % period;
    return i < n ? i : period - 1 - i;
  };
  const ir = new Float64Array(W * H);
  const ii = new Float64Array(W * H);
  for (let y = 0; y < H; y++) for (let x = 0; x < W; x++) ir[y * W + x] = gray[refl(y - K, h) * w + refl(x - K, w)];
  fft2(ir, ii, W, H);
  const out = [];
  const kr = new Float64Array(W * H);
  const ki = new Float64Array(W * H);
  const pr = new Float64Array(W * H);
  const pi = new Float64Array(W * H);
  for (const k of GABOR_BANK) {
    kr.fill(0);
    ki.fill(0);
    for (let y = -k.y0; y <= k.y0; y++) {
      for (let x = -k.x0; x <= k.x0; x++) {
        const o = ((y + H) % H) * W + ((x + W) % W);
        kr[o] = k.re[(y + k.y0) * k.kw + x + k.x0];
        ki[o] = k.im[(y + k.y0) * k.kw + x + k.x0];
      }
    }
    fft2(kr, ki, W, H);
    for (let i = 0; i < W * H; i++) {
      pr[i] = ir[i] * kr[i] - ii[i] * ki[i];
      pi[i] = ir[i] * ki[i] + ii[i] * kr[i];
    }
    fft2(pr, pi, W, H, true);
    let s = 0;
    let s2 = 0;
    for (let y = K; y < K + h; y++) {
      for (let x = K; x < K + w; x++) {
        const m = Math.hypot(pr[y * W + x], pi[y * W + x]);
        s += m;
        s2 += m * m;
      }
    }
    const n = w * h;
    const mean = s / n;
    out.push(mean, Math.sqrt(Math.max(0, s2 / n - mean * mean)));
  }
  return Float64Array.from(out);
}

// ---------------------------------------------------------------- GIST

/** LMgist.m's createGabor: transfer functions for orientationsPerScale on an n×n grid. */
function createGabor(orientationsPerScale, n) {
  const params = [];
  orientationsPerScale.forEach((or, i) => {
    for (let j = 0; j < or; j++) params.push([0.35, 0.3 / 1.85 ** i, (16 * or * or) / 32 ** 2, (Math.PI / or) * j]);
  });
  const fr = new Float64Array(n * n);
  const t = new Float64Array(n * n);
  for (let y = 0; y < n; y++) {
    for (let x = 0; x < n; x++) {
      // meshgrid(-n/2:n/2-1), then fftshift
      const fx = ((x + n / 2) % n) - n / 2;
      const fy = ((y + n / 2) % n) - n / 2;
      fr[y * n + x] = Math.hypot(fx, fy);
      t[y * n + x] = Math.atan2(fy, fx);
    }
  }
  return params.map(([a, b, c, d]) => {
    const G = new Float64Array(n * n);
    for (let i = 0; i < n * n; i++) {
      let tr = t[i] + d;
      tr += tr < -Math.PI ? 2 * Math.PI : tr > Math.PI ? -2 * Math.PI : 0;
      G[i] = Math.exp(-10 * a * (fr[i] / n / b - 1) ** 2 - 2 * c * Math.PI * tr * tr);
    }
    return G;
  });
}

let GIST_FILTERS = null;

/** MATLAB padarray(..., 'symmetric') index. */
const sym = (i, n) => {
  const period = 2 * n;
  i = ((i % period) + period) % period;
  return i < n ? i : period - 1 - i;
};

/**
 * The GIST descriptor (Oliva & Torralba 2001), as LMgist.m with its defaults:
 * a 256×256 grey image, whitened and contrast-normalised ("prefilt", fc = 4),
 * filtered by 32 Gabor filters (8 orientations × 4 scales), each response
 * averaged over a 4×4 grid — 512 numbers. Input: 256×256 grey plane (0..255,
 * any range; it is stretched to 0..255 as LMgist does).
 */
export function gistDescriptor(gray256) {
  const N = 256;
  const be = 32;
  const S = N + 2 * be;
  GIST_FILTERS ??= createGabor([8, 8, 8, 8], S);
  // stretch to 0..255
  let lo = Infinity;
  let hi = -Infinity;
  for (const v of gray256) {
    lo = Math.min(lo, v);
    hi = Math.max(hi, v);
  }
  const span = hi - lo || 1;
  const img = Float64Array.from(gray256, (v) => (255 * (v - lo)) / span);
  // prefilt(img, 4)
  const w = 5;
  const s1 = 4 / Math.sqrt(Math.log(2));
  const sn = N + 2 * w;
  const n = sn + (sn % 2);
  const pr = new Float64Array(n * n);
  const pim = new Float64Array(n * n);
  // pad 5 symmetric on all sides, then symmetric 'post' padding up to n×n
  const padded = (y, x) => {
    const yy = y < sn ? y : 2 * sn - 1 - y;
    const xx = x < sn ? x : 2 * sn - 1 - x;
    return Math.log(img[sym(yy - w, N) * N + sym(xx - w, N)] + 1);
  };
  for (let y = 0; y < n; y++) for (let x = 0; x < n; x++) pr[y * n + x] = padded(y, x);
  const gf = new Float64Array(n * n);
  for (let y = 0; y < n; y++) {
    for (let x = 0; x < n; x++) {
      const fx = ((x + n / 2) % n) - n / 2;
      const fy = ((y + n / 2) % n) - n / 2;
      gf[y * n + x] = Math.exp(-(fx * fx + fy * fy) / (s1 * s1));
    }
  }
  const src = Float64Array.from(pr);
  fft2(pr, pim, n, n);
  for (let i = 0; i < n * n; i++) {
    pr[i] *= gf[i];
    pim[i] *= gf[i];
  }
  fft2(pr, pim, n, n, true);
  const out = new Float64Array(n * n);
  for (let i = 0; i < n * n; i++) out[i] = src[i] - pr[i];
  const lr = Float64Array.from(out, (v) => v * v);
  const li = new Float64Array(n * n);
  fft2(lr, li, n, n);
  for (let i = 0; i < n * n; i++) {
    lr[i] *= gf[i];
    li[i] *= gf[i];
  }
  fft2(lr, li, n, n, true);
  const pre = new Float64Array(N * N);
  for (let y = 0; y < N; y++) {
    for (let x = 0; x < N; x++) {
      const i = (y + w) * n + x + w;
      pre[y * N + x] = out[i] / (0.2 + Math.sqrt(Math.hypot(lr[i], li[i])));
    }
  }
  // gistGabor: symmetric padding by 32, 32 filters, 4×4 block averages
  const fr = new Float64Array(S * S);
  const fi = new Float64Array(S * S);
  for (let y = 0; y < S; y++) for (let x = 0; x < S; x++) fr[y * S + x] = pre[sym(y - be, N) * N + sym(x - be, N)];
  fft2(fr, fi, S, S);
  const edges = Array.from({ length: 5 }, (_, k) => Math.trunc((k * N) / 4));
  const g = new Float64Array(GIST_FILTERS.length * 16);
  const zr = new Float64Array(S * S);
  const zi = new Float64Array(S * S);
  GIST_FILTERS.forEach((G, f) => {
    for (let i = 0; i < S * S; i++) {
      zr[i] = fr[i] * G[i];
      zi[i] = fi[i] * G[i];
    }
    fft2(zr, zi, S, S, true);
    // MATLAB order: blocks run down the rows first (column-major)
    for (let bx = 0; bx < 4; bx++) {
      for (let by = 0; by < 4; by++) {
        let s = 0;
        let cnt = 0;
        for (let y = edges[by]; y < edges[by + 1]; y++) {
          for (let x = edges[bx]; x < edges[bx + 1]; x++) {
            const i = (y + be) * S + x + be;
            s += Math.hypot(zr[i], zi[i]);
            cnt++;
          }
        }
        g[f * 16 + bx * 4 + by] = s / cnt;
      }
    }
  });
  return g;
}

/**
 * LMgist's input: the image scaled (bilinear) so that it covers 256×256,
 * centre-cropped, as the mean of its three channels.
 */
export function gistInput(img, size = 256) {
  const scale = Math.max(size / img.h, size / img.w);
  const nw = Math.max(size, Math.round(img.w * scale));
  const nh = Math.max(size, Math.round(img.h * scale));
  const r = cropImg(resizeImg(img, nw, nh, 'bilinear'), Math.floor((nw - size) / 2), Math.floor((nh - size) / 2), size, size);
  const out = new Float64Array(size * size);
  for (let i = 0; i < out.length; i++) out[i] = (r.rgb[i * 3] + r.rgb[i * 3 + 1] + r.rgb[i * 3 + 2]) / 3;
  return out;
}

// ---------------------------------------------------------------- Zernike

const FACT = [1, 1, 2, 6, 24, 120, 720, 5040, 40320, 362880, 3628800, 39916800, 479001600];
const fact = (n) => (n < FACT.length ? FACT[n] : n * fact(n - 1));

/**
 * Zernike moment magnitudes (Teague 1980) up to degree 8 — 25 numbers that do
 * not change when the picture is rotated or mirrored — on a disc around the
 * image's centre of brightness, as mahotas.features.zernike_moments.
 */
export function zernikeMoments(gray, w, h, radius = Math.hypot(w, h) / 2, degree = 8) {
  let sum = 0;
  let sy = 0;
  let sx = 0;
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      const v = gray[y * w + x];
      sum += v;
      sy += y * v;
      sx += x * v;
    }
  }
  if (!sum) return null;
  const c0 = sy / sum;
  const c1 = sx / sum;
  const pts = [];
  let total = 0;
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      const v = gray[y * w + x];
      const yn = (y - c0) / radius;
      const xn = (x - c1) / radius;
      const d = Math.max(Math.sqrt(xn * xn + yn * yn), 1e-9);
      if (d <= 1 && v > 0) {
        pts.push([d, xn / d, yn / d, v]);
        total += v;
      }
    }
  }
  const out = [];
  for (let n = 0; n <= degree; n++) {
    for (let l = 0; l <= n; l++) {
      if ((n - l) % 2) continue;
      const g = [];
      for (let m = 0; m <= (n - l) / 2; m++) g.push(((m & 1 ? -1 : 1) * fact(n - m)) / (fact(m) * fact((n - 2 * m + l) / 2) * fact((n - 2 * m - l) / 2)));
      let vr = 0;
      let vi = 0;
      for (const [d, cr, ci, p] of pts) {
        let rad = 0;
        for (let m = 0; m < g.length; m++) rad += g[m] * d ** (n - 2 * m);
        // (cos θ + i sin θ)^l
        let ar = 1;
        let ai = 0;
        for (let k = 0; k < l; k++) [ar, ai] = [ar * cr - ai * ci, ar * ci + ai * cr];
        // p · conj(R(d) · a)
        vr += (p / total) * rad * ar;
        vi -= (p / total) * rad * ai;
      }
      out.push((Math.hypot(vr, vi) * (n + 1)) / Math.PI);
    }
  }
  return Float64Array.from(out);
}

export function euclidean(a, b) {
  let s = 0;
  for (let i = 0; i < a.length; i++) s += (a[i] - b[i]) ** 2;
  return Math.sqrt(s);
}
