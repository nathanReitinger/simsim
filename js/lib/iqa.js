// Full-reference image quality / similarity metrics. All functions take two
// images of identical size. Grayscale inputs are Float64Array planes in 0..255.

// ---------------------------------------------------------------- filtering

function gaussian1d(sigma, radius) {
  const k = new Float64Array(2 * radius + 1);
  let sum = 0;
  for (let i = -radius; i <= radius; i++) {
    k[i + radius] = Math.exp((-0.5 * i * i) / (sigma * sigma));
    sum += k[i + radius];
  }
  for (let i = 0; i < k.length; i++) k[i] /= sum;
  return k;
}

function reflect(i, n) {
  // scipy.ndimage mode='reflect' (d c b a | a b c d | d c b a)
  while (i < 0 || i >= n) i = i < 0 ? -i - 1 : 2 * n - i - 1;
  return i;
}

/** Separable filter with scipy 'reflect' borders; output has the input size. */
function filterSame(src, w, h, k) {
  const r = (k.length - 1) / 2;
  const tmp = new Float64Array(w * h);
  const out = new Float64Array(w * h);
  for (let y = 0; y < h; y++) {
    const row = y * w;
    for (let x = 0; x < w; x++) {
      let s = 0;
      for (let i = -r; i <= r; i++) s += k[i + r] * src[row + reflect(x + i, w)];
      tmp[row + x] = s;
    }
  }
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      let s = 0;
      for (let i = -r; i <= r; i++) s += k[i + r] * tmp[reflect(y + i, h) * w + x];
      out[y * w + x] = s;
    }
  }
  return out;
}

/** Separable filter keeping only the 'valid' region (no padding). */
function filterValid(src, w, h, k) {
  const n = k.length;
  const ow = w - n + 1;
  const oh = h - n + 1;
  const tmp = new Float64Array(ow * h);
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < ow; x++) {
      let s = 0;
      for (let i = 0; i < n; i++) s += k[i] * src[y * w + x + i];
      tmp[y * ow + x] = s;
    }
  }
  const out = new Float64Array(ow * oh);
  for (let y = 0; y < oh; y++) {
    for (let x = 0; x < ow; x++) {
      let s = 0;
      for (let i = 0; i < n; i++) s += k[i] * tmp[(y + i) * ow + x];
      out[y * ow + x] = s;
    }
  }
  return { data: out, w: ow, h: oh };
}

const mul = (a, b) => a.map((v, i) => v * b[i]);

// ---------------------------------------------------------------- SSIM

const C1 = (0.01 * 255) ** 2;
const C2 = (0.03 * 255) ** 2;

/**
 * SSIM (Wang, Bovik, Sheikh & Simoncelli 2004) with an 11x11 Gaussian window
 * (sigma 1.5). Matches skimage.metrics.structural_similarity(gaussian_weights
 * =True, use_sample_covariance=False, data_range=255).
 */
export function ssim(X, Y, w, h, mask = null) {
  const k = gaussian1d(1.5, 5);
  const ux = filterSame(X, w, h, k);
  const uy = filterSame(Y, w, h, k);
  const uxx = filterSame(mul(X, X), w, h, k);
  const uyy = filterSame(mul(Y, Y), w, h, k);
  const uxy = filterSame(mul(X, Y), w, h, k);
  const map = new Float64Array(w * h);
  for (let i = 0; i < map.length; i++) {
    const vx = uxx[i] - ux[i] * ux[i];
    const vy = uyy[i] - uy[i] * uy[i];
    const vxy = uxy[i] - ux[i] * uy[i];
    map[i] =
      ((2 * ux[i] * uy[i] + C1) * (2 * vxy + C2)) / ((ux[i] * ux[i] + uy[i] * uy[i] + C1) * (vx + vy + C2));
  }
  const pad = 5;
  let sum = 0;
  let n = 0;
  for (let y = pad; y < h - pad; y++) {
    for (let x = pad; x < w - pad; x++) {
      if (mask && !mask[y * w + x]) continue;
      sum += map[y * w + x];
      n++;
    }
  }
  return { value: n ? sum / n : NaN, map, count: n };
}

function ssimValid(X, Y, w, h, k) {
  const ux = filterValid(X, w, h, k);
  const uy = filterValid(Y, w, h, k).data;
  const uxx = filterValid(mul(X, X), w, h, k).data;
  const uyy = filterValid(mul(Y, Y), w, h, k).data;
  const uxy = filterValid(mul(X, Y), w, h, k).data;
  let ss = 0;
  let cs = 0;
  const n = ux.data.length;
  for (let i = 0; i < n; i++) {
    const mx = ux.data[i];
    const my = uy[i];
    const c = (2 * (uxy[i] - mx * my) + C2) / (uxx[i] - mx * mx + uyy[i] - my * my + C2);
    cs += c;
    ss += ((2 * mx * my + C1) / (mx * mx + my * my + C1)) * c;
  }
  return { ssim: ss / n, cs: cs / n };
}

function avgPool2(X, w, h) {
  // torch.nn.functional.avg_pool2d(kernel=2, padding=size%2, count_include_pad=True)
  const pw = w % 2;
  const ph = h % 2;
  const ow = Math.floor((w + 2 * pw - 2) / 2) + 1;
  const oh = Math.floor((h + 2 * ph - 2) / 2) + 1;
  const out = new Float64Array(ow * oh);
  for (let y = 0; y < oh; y++) {
    for (let x = 0; x < ow; x++) {
      let s = 0;
      for (let dy = 0; dy < 2; dy++) {
        for (let dx = 0; dx < 2; dx++) {
          const sy = 2 * y + dy - ph;
          const sx = 2 * x + dx - pw;
          if (sy >= 0 && sy < h && sx >= 0 && sx < w) s += X[sy * w + sx];
        }
      }
      out[y * ow + x] = s / 4;
    }
  }
  return { data: out, w: ow, h: oh };
}

/**
 * Multi-scale SSIM (Wang, Simoncelli & Bovik 2003), following pytorch_msssim.
 * Small images use fewer scales (weights renormalised).
 */
export function msssim(X, Y, w, h) {
  if (Math.min(w, h) < 11) return { value: NaN, levels: 0, perScale: [] };
  const allWeights = [0.0448, 0.2856, 0.3001, 0.2363, 0.1333];
  let levels = 5;
  while (levels > 1 && Math.min(w, h) <= 10 * 2 ** (levels - 1)) levels--;
  const weights = allWeights.slice(0, levels);
  // The published weights sum to 1.0001; use them as-is (like the reference
  // implementations) and only renormalise when scales had to be dropped.
  const wsum = levels === allWeights.length ? 1 : weights.reduce((a, b) => a + b, 0);
  const k = gaussian1d(1.5, 5);
  let x = X;
  let y = Y;
  let cw = w;
  let ch = h;
  let result = 1;
  const perScale = [];
  for (let i = 0; i < levels; i++) {
    const { ssim: s, cs } = ssimValid(x, y, cw, ch, k);
    const v = i < levels - 1 ? Math.max(cs, 0) : Math.max(s, 0);
    perScale.push(v);
    result *= v ** (weights[i] / wsum);
    if (i < levels - 1) {
      const px = avgPool2(x, cw, ch);
      y = avgPool2(y, cw, ch).data;
      x = px.data;
      cw = px.w;
      ch = px.h;
    }
  }
  return { value: result, levels, perScale };
}

// ---------------------------------------------------------------- UQI

function boxSums(src, w, h, b) {
  // sums over every b x b window ('valid')
  const I = new Float64Array((w + 1) * (h + 1));
  for (let y = 0; y < h; y++) {
    let row = 0;
    for (let x = 0; x < w; x++) {
      row += src[y * w + x];
      I[(y + 1) * (w + 1) + x + 1] = I[y * (w + 1) + x + 1] + row;
    }
  }
  const ow = w - b + 1;
  const oh = h - b + 1;
  const out = new Float64Array(ow * oh);
  for (let y = 0; y < oh; y++) {
    for (let x = 0; x < ow; x++) {
      out[y * ow + x] =
        I[(y + b) * (w + 1) + x + b] - I[y * (w + 1) + x + b] - I[(y + b) * (w + 1) + x] + I[y * (w + 1) + x];
    }
  }
  return out;
}

/** Universal Image Quality Index (Wang & Bovik 2002), 8x8 sliding window. */
export function uqi(X, Y, w, h, b = 8) {
  if (Math.min(w, h) < b) return NaN;
  const N = b * b;
  const sx = boxSums(X, w, h, b);
  const sy = boxSums(Y, w, h, b);
  const sxx = boxSums(mul(X, X), w, h, b);
  const syy = boxSums(mul(Y, Y), w, h, b);
  const sxy = boxSums(mul(X, Y), w, h, b);
  let total = 0;
  for (let i = 0; i < sx.length; i++) {
    const prod = sx[i] * sy[i];
    const sqsum = sx[i] * sx[i] + sy[i] * sy[i];
    const num = 4 * (N * sxy[i] - prod) * prod;
    const den1 = N * (sxx[i] + syy[i]) - sqsum;
    const den = den1 * sqsum;
    let q = 1;
    if (den !== 0) q = num / den;
    else if (den1 === 0 && sqsum !== 0) q = (2 * prod) / sqsum;
    total += q;
  }
  return total / sx.length;
}

// ---------------------------------------------------------------- GMSD

/** Gradient Magnitude Similarity Deviation (Xue et al. 2014), MATLAB reference. */
export function gmsd(X, Y, w, h) {
  const down = (A) => {
    const ow = Math.ceil(w / 2);
    const oh = Math.ceil(h / 2);
    const out = new Float64Array(ow * oh);
    for (let y = 0; y < oh; y++) {
      for (let x = 0; x < ow; x++) {
        let s = 0;
        for (let dy = 0; dy < 2; dy++) {
          for (let dx = 0; dx < 2; dx++) {
            const sy = 2 * y + dy;
            const sx = 2 * x + dx;
            if (sy < h && sx < w) s += A[sy * w + sx];
          }
        }
        out[y * ow + x] = s / 4;
      }
    }
    return { a: out, w: ow, h: oh };
  };
  const gradient = ({ a, w: gw, h: gh }) => {
    const at = (x, y) => (x < 0 || y < 0 || x >= gw || y >= gh ? 0 : a[y * gw + x]);
    const g = new Float64Array(gw * gh);
    for (let y = 0; y < gh; y++) {
      for (let x = 0; x < gw; x++) {
        let ix = 0;
        let iy = 0;
        for (let d = -1; d <= 1; d++) {
          ix += at(x - 1, y + d) - at(x + 1, y + d);
          iy += at(x + d, y - 1) - at(x + d, y + 1);
        }
        g[y * gw + x] = Math.hypot(ix / 3, iy / 3);
      }
    }
    return g;
  };
  const dx = down(X);
  const g1 = gradient(dx);
  const g2 = gradient(down(Y));
  const T = 170;
  const n = g1.length;
  const q = new Float64Array(n);
  let mean = 0;
  for (let i = 0; i < n; i++) {
    q[i] = (2 * g1[i] * g2[i] + T) / (g1[i] * g1[i] + g2[i] * g2[i] + T);
    mean += q[i];
  }
  mean /= n;
  let v = 0;
  for (let i = 0; i < n; i++) v += (q[i] - mean) ** 2;
  return { value: Math.sqrt(v / (n - 1)), map: q, w: dx.w, h: dx.h };
}

// ---------------------------------------------------------------- pixel stats

/** MSE / PSNR / MAE over RGB channels. */
export function pixelErrors(rgbA, rgbB) {
  let se = 0;
  let ae = 0;
  for (let i = 0; i < rgbA.length; i++) {
    const d = rgbA[i] - rgbB[i];
    se += d * d;
    ae += Math.abs(d);
  }
  const mse = se / rgbA.length;
  return { mse, psnr: mse === 0 ? Infinity : 10 * Math.log10((255 * 255) / mse), mae: ae / rgbA.length };
}

/** Pearson correlation of two planes (normalised cross-correlation). */
export function ncc(X, Y) {
  const n = X.length;
  let mx = 0;
  let my = 0;
  for (let i = 0; i < n; i++) {
    mx += X[i];
    my += Y[i];
  }
  mx /= n;
  my /= n;
  let sxy = 0;
  let sxx = 0;
  let syy = 0;
  for (let i = 0; i < n; i++) {
    const a = X[i] - mx;
    const b = Y[i] - my;
    sxy += a * b;
    sxx += a * a;
    syy += b * b;
  }
  if (sxx === 0 && syy === 0) return mx === my ? 1 : 0;
  if (sxx === 0 || syy === 0) return 0;
  return sxy / Math.sqrt(sxx * syy);
}

// ---------------------------------------------------------------- color

const SRGB_LINEAR = Float64Array.from({ length: 256 }, (_, i) => {
  const c = i / 255;
  return c <= 0.04045 ? c / 12.92 : ((c + 0.055) / 1.055) ** 2.4;
});

function labF(t) {
  return t > 216 / 24389 ? Math.cbrt(t) : t / (3 * (6 / 29) ** 2) + 4 / 29;
}

/** sRGB (D65) -> CIE L*a*b*, interleaved Float64Array. */
export function rgbToLab(rgb) {
  const n = rgb.length / 3;
  const lab = new Float64Array(n * 3);
  for (let i = 0; i < n; i++) {
    const r = SRGB_LINEAR[rgb[i * 3]];
    const g = SRGB_LINEAR[rgb[i * 3 + 1]];
    const b = SRGB_LINEAR[rgb[i * 3 + 2]];
    const fx = labF((0.4124564 * r + 0.3575761 * g + 0.1804375 * b) / 0.95047);
    const fy = labF(0.2126729 * r + 0.7151522 * g + 0.072175 * b);
    const fz = labF((0.0193339 * r + 0.119192 * g + 0.9503041 * b) / 1.08883);
    lab[i * 3] = 116 * fy - 16;
    lab[i * 3 + 1] = 500 * (fx - fy);
    lab[i * 3 + 2] = 200 * (fy - fz);
  }
  return lab;
}

const DEG = Math.PI / 180;
const P25_7 = 25 ** 7;

/** CIEDE2000 colour difference (Sharma, Wu & Dalal 2005), kL = kC = kH = 1. */
export function ciede2000(L1, a1, b1, L2, a2, b2) {
  const C1 = Math.hypot(a1, b1);
  const C2 = Math.hypot(a2, b2);
  const Cb7 = ((C1 + C2) / 2) ** 7;
  const G = 0.5 * (1 - Math.sqrt(Cb7 / (Cb7 + P25_7)));
  const a1p = (1 + G) * a1;
  const a2p = (1 + G) * a2;
  const C1p = Math.hypot(a1p, b1);
  const C2p = Math.hypot(a2p, b2);
  const hue = (b, a) => {
    if (a === 0 && b === 0) return 0;
    const h = Math.atan2(b, a) / DEG;
    return h < 0 ? h + 360 : h;
  };
  const h1p = hue(b1, a1p);
  const h2p = hue(b2, a2p);
  const dLp = L2 - L1;
  const dCp = C2p - C1p;
  let dhp = 0;
  if (C1p * C2p !== 0) {
    dhp = h2p - h1p;
    if (dhp > 180) dhp -= 360;
    else if (dhp < -180) dhp += 360;
  }
  const dHp = 2 * Math.sqrt(C1p * C2p) * Math.sin((dhp / 2) * DEG);
  const Lbp = (L1 + L2) / 2;
  const Cbp = (C1p + C2p) / 2;
  let hbp = h1p + h2p;
  if (C1p * C2p !== 0) {
    if (Math.abs(h1p - h2p) <= 180) hbp = (h1p + h2p) / 2;
    else if (h1p + h2p < 360) hbp = (h1p + h2p + 360) / 2;
    else hbp = (h1p + h2p - 360) / 2;
  }
  const T =
    1 -
    0.17 * Math.cos((hbp - 30) * DEG) +
    0.24 * Math.cos(2 * hbp * DEG) +
    0.32 * Math.cos((3 * hbp + 6) * DEG) -
    0.2 * Math.cos((4 * hbp - 63) * DEG);
  const dTheta = 30 * Math.exp(-(((hbp - 275) / 25) ** 2));
  const Cbp7 = Cbp ** 7;
  const RC = 2 * Math.sqrt(Cbp7 / (Cbp7 + P25_7));
  const SL = 1 + (0.015 * (Lbp - 50) ** 2) / Math.sqrt(20 + (Lbp - 50) ** 2);
  const SC = 1 + 0.045 * Cbp;
  const SH = 1 + 0.015 * Cbp * T;
  const RT = -Math.sin(2 * dTheta * DEG) * RC;
  const l = dLp / SL;
  const c = dCp / SC;
  const hh = dHp / SH;
  return Math.sqrt(l * l + c * c + hh * hh + RT * c * hh);
}

/** Per-pixel CIEDE2000 between two RGB images. */
export function deltaE2000Map(rgbA, rgbB) {
  const A = rgbToLab(rgbA);
  const B = rgbToLab(rgbB);
  const n = A.length / 3;
  const map = new Float64Array(n);
  let sum = 0;
  for (let i = 0; i < n; i++) {
    const j = i * 3;
    map[i] = ciede2000(A[j], A[j + 1], A[j + 2], B[j], B[j + 1], B[j + 2]);
    sum += map[i];
  }
  return { map, mean: sum / n };
}
