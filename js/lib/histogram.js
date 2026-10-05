// Colour / intensity histograms and the classic ways of comparing them.
// Histograms ignore *where* colours are, so they are fast but easily fooled.

/** OpenCV-style hue/saturation histogram (50 x 60 bins), normalised to sum 1. */
export function hsHistogram(rgb, hBins = 50, sBins = 60) {
  const hist = new Float64Array(hBins * sBins);
  const n = rgb.length / 3;
  for (let i = 0; i < n; i++) {
    const r = rgb[i * 3];
    const g = rgb[i * 3 + 1];
    const b = rgb[i * 3 + 2];
    const v = Math.max(r, g, b);
    const mn = Math.min(r, g, b);
    const d = v - mn;
    const s = v === 0 ? 0 : d / v;
    let h = 0;
    if (d !== 0) {
      if (v === r) h = (60 * (g - b)) / d;
      else if (v === g) h = 120 + (60 * (b - r)) / d;
      else h = 240 + (60 * (r - g)) / d;
      if (h < 0) h += 360;
    }
    const hb = Math.min(hBins - 1, Math.floor((h / 360) * hBins));
    const sb = Math.min(sBins - 1, Math.floor(s * sBins));
    hist[hb * sBins + sb]++;
  }
  return normalize(hist);
}

/** Joint RGB histogram with 8 bins per channel (Swain & Ballard 1991). */
export function rgbHistogram(rgb) {
  const hist = new Float64Array(512);
  for (let i = 0; i < rgb.length; i += 3) hist[((rgb[i] >> 5) << 6) | ((rgb[i + 1] >> 5) << 3) | (rgb[i + 2] >> 5)]++;
  return normalize(hist);
}

/** 256-bin histogram of a 0..255 plane. */
export function grayHistogram(plane) {
  const hist = new Float64Array(256);
  for (const v of plane) hist[Math.min(255, Math.max(0, Math.round(v)))]++;
  return normalize(hist);
}

function normalize(h) {
  let s = 0;
  for (const v of h) s += v;
  if (s > 0) for (let i = 0; i < h.length; i++) h[i] /= s;
  return h;
}

export function correlation(h1, h2) {
  const n = h1.length;
  let m1 = 0;
  let m2 = 0;
  for (let i = 0; i < n; i++) {
    m1 += h1[i];
    m2 += h2[i];
  }
  m1 /= n;
  m2 /= n;
  let num = 0;
  let d1 = 0;
  let d2 = 0;
  for (let i = 0; i < n; i++) {
    const a = h1[i] - m1;
    const b = h2[i] - m2;
    num += a * b;
    d1 += a * a;
    d2 += b * b;
  }
  return d1 && d2 ? num / Math.sqrt(d1 * d2) : 1;
}

/** Symmetric chi-square distance, 0 (same) .. 1 (disjoint). */
export function chiSquare(h1, h2) {
  let s = 0;
  for (let i = 0; i < h1.length; i++) {
    const t = h1[i] + h2[i];
    if (t > 0) s += ((h1[i] - h2[i]) ** 2) / t;
  }
  return s / 2;
}

export function intersection(h1, h2) {
  let s = 0;
  for (let i = 0; i < h1.length; i++) s += Math.min(h1[i], h2[i]);
  return s;
}

/** Bhattacharyya (Hellinger) distance, 0 (same) .. 1 (disjoint). */
export function bhattacharyya(h1, h2) {
  let bc = 0;
  for (let i = 0; i < h1.length; i++) bc += Math.sqrt(h1[i] * h2[i]);
  return Math.sqrt(Math.max(0, 1 - bc));
}

/** 1-D Earth Mover's (Wasserstein-1) distance, normalised to 0..1. */
export function emd1d(h1, h2) {
  let c1 = 0;
  let c2 = 0;
  let d = 0;
  for (let i = 0; i < h1.length; i++) {
    c1 += h1[i];
    c2 += h2[i];
    d += Math.abs(c1 - c2);
  }
  return d / (h1.length - 1);
}
