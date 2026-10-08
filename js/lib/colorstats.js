// Colour descriptors from content-based image retrieval: statistical
// divergences between histograms, colour moments, colour coherence vectors,
// colour correlograms and dominant-colour palettes. All ignore where things
// are, so they survive crops, flips and rotations, and all can be fooled by
// unrelated pictures that happen to share a palette.

import { ciede2000, rgbToLab } from './iqa.js';
import { fitWithin, resizeImg } from './pixels.js';

/** The image shrunk (box filter) so its longest side is at most `max`. */
export function shrink(img, max) {
  const [w, h] = fitWithin(img.w, img.h, max);
  return resizeImg(img, w, h, 'box');
}

// ---------------------------------------------------------------- divergences

/** OpenCV's HISTCMP_KL_DIV: Kullback–Leibler divergence of h1 from h2 (0 = same). */
export function klDivergence(h1, h2) {
  let s = 0;
  for (let i = 0; i < h1.length; i++) {
    const p = h1[i];
    if (Math.abs(p) <= Number.EPSILON) continue;
    let q = h2[i];
    if (Math.abs(q) <= Number.EPSILON) q = 1e-10;
    s += p * Math.log(p / q);
  }
  return s;
}

/** Jensen–Shannon distance in bits, 0 (same) .. 1 (disjoint), as scipy.spatial.distance.jensenshannon(p, q, base=2). */
export function jensenShannon(h1, h2) {
  let s1 = 0;
  let s2 = 0;
  for (let i = 0; i < h1.length; i++) {
    s1 += h1[i];
    s2 += h2[i];
  }
  let js = 0;
  for (let i = 0; i < h1.length; i++) {
    const p = h1[i] / s1;
    const q = h2[i] / s2;
    const m = (p + q) / 2;
    if (p > 0) js += p * Math.log(p / m);
    if (q > 0) js += q * Math.log(q / m);
  }
  return Math.sqrt(Math.max(0, js / Math.LN2 / 2));
}

/** Two-sample Kolmogorov–Smirnov statistic of two 8-bit planes: the largest gap between their cumulative distributions. */
export function ksStatistic(ga, gb) {
  const ha = new Float64Array(256);
  const hb = new Float64Array(256);
  for (const v of ga) ha[v]++;
  for (const v of gb) hb[v]++;
  let ca = 0;
  let cb = 0;
  let d = 0;
  for (let i = 0; i < 256; i++) {
    ca += ha[i] / ga.length;
    cb += hb[i] / gb.length;
    d = Math.max(d, Math.abs(ca - cb));
  }
  return d;
}

// ---------------------------------------------------------------- colour moments

/** HSV in [0, 1] (Smith 1978). */
function hsv(r, g, b) {
  const max = Math.max(r, g, b);
  const min = Math.min(r, g, b);
  const d = max - min;
  let h = 0;
  if (d > 0) {
    if (max === r) h = ((g - b) / d) % 6;
    else if (max === g) h = (b - r) / d + 2;
    else h = (r - g) / d + 4;
    h /= 6;
    if (h < 0) h += 1;
  }
  return [h, max === 0 ? 0 : d / max, max / 255];
}

/**
 * Colour moments (Stricker & Orengo 1995): mean, standard deviation and the
 * cube root of the third central moment of hue, saturation and value.
 */
export function colorMoments(img) {
  const { rgb } = img;
  const n = rgb.length / 3;
  const ch = [new Float64Array(n), new Float64Array(n), new Float64Array(n)];
  for (let i = 0; i < n; i++) {
    const [h, s, v] = hsv(rgb[i * 3], rgb[i * 3 + 1], rgb[i * 3 + 2]);
    ch[0][i] = h;
    ch[1][i] = s;
    ch[2][i] = v;
  }
  const out = [];
  for (const c of ch) {
    let mean = 0;
    for (const v of c) mean += v;
    mean /= n;
    let m2 = 0;
    let m3 = 0;
    for (const v of c) {
      const d = v - mean;
      m2 += d * d;
      m3 += d * d * d;
    }
    out.push(mean, Math.sqrt(m2 / n), Math.cbrt(m3 / n));
  }
  return Float64Array.from(out);
}

/** Stricker & Orengo's distance: sum of absolute differences of the nine moments (equal weights). */
export function colorMomentsDistance(a, b) {
  let s = 0;
  for (let i = 0; i < a.length; i++) s += Math.abs(a[i] - b[i]);
  return s;
}

// ---------------------------------------------------------------- 64-colour images

/** Each pixel's colour reduced to 64 (2 bits per channel). */
function quantize64(rgb) {
  const n = rgb.length / 3;
  const q = new Uint8Array(n);
  for (let i = 0; i < n; i++) q[i] = ((rgb[i * 3] >> 6) << 4) | ((rgb[i * 3 + 1] >> 6) << 2) | (rgb[i * 3 + 2] >> 6);
  return q;
}

/**
 * Colour coherence vector (Pass, Zabih & Miller 1996): after a 3×3 blur and
 * reduction to 64 colours, the share of pixels of each colour that belong to
 * a large connected patch (≥ tau of the image) and the share that do not.
 */
export function colorCoherence(img, tau = 0.01) {
  const { w, h, rgb } = img;
  const n = w * h;
  // 3×3 mean, edges repeated
  const blurred = new Uint8Array(n * 3);
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      for (let c = 0; c < 3; c++) {
        let s = 0;
        for (let dy = -1; dy <= 1; dy++) {
          const yy = Math.min(h - 1, Math.max(0, y + dy));
          for (let dx = -1; dx <= 1; dx++) s += rgb[(yy * w + Math.min(w - 1, Math.max(0, x + dx))) * 3 + c];
        }
        blurred[(y * w + x) * 3 + c] = Math.floor(s / 9);
      }
    }
  }
  const q = quantize64(blurred);
  const seen = new Uint8Array(n);
  const queue = new Int32Array(n);
  const alpha = new Float64Array(64);
  const beta = new Float64Array(64);
  const min = tau * n;
  for (let p = 0; p < n; p++) {
    if (seen[p]) continue;
    const colour = q[p];
    let head = 0;
    let tail = 0;
    queue[tail++] = p;
    seen[p] = 1;
    while (head < tail) {
      const i = queue[head++];
      const x = i % w;
      const y = (i / w) | 0;
      for (let dy = -1; dy <= 1; dy++) {
        const yy = y + dy;
        if (yy < 0 || yy >= h) continue;
        for (let dx = -1; dx <= 1; dx++) {
          const xx = x + dx;
          if (xx < 0 || xx >= w) continue;
          const j = yy * w + xx;
          if (!seen[j] && q[j] === colour) {
            seen[j] = 1;
            queue[tail++] = j;
          }
        }
      }
    }
    if (tail >= min) alpha[colour] += tail / n;
    else beta[colour] += tail / n;
  }
  return { alpha, beta };
}

/** CCV distance: half the summed differences of coherent and incoherent shares, 0 (same) .. 1. */
export function coherenceDistance(a, b) {
  let s = 0;
  for (let i = 0; i < 64; i++) s += Math.abs(a.alpha[i] - b.alpha[i]) + Math.abs(a.beta[i] - b.beta[i]);
  return s / 2;
}

/**
 * Colour auto-correlogram (Huang et al. 1997): for each of 64 colours and
 * each distance d, the chance that a pixel d steps away (chessboard
 * distance) from a pixel of that colour has the same colour.
 */
export function autoCorrelogram(img, distances = [1, 3, 5, 7]) {
  const { w, h } = img;
  const q = quantize64(img.rgb);
  const out = new Float64Array(64 * distances.length);
  distances.forEach((d, k) => {
    const same = new Float64Array(64);
    const total = new Float64Array(64);
    const ring = [];
    for (let dx = -d; dx <= d; dx++) ring.push([dx, -d], [dx, d]);
    for (let dy = -d + 1; dy <= d - 1; dy++) ring.push([-d, dy], [d, dy]);
    for (let y = 0; y < h; y++) {
      for (let x = 0; x < w; x++) {
        const c = q[y * w + x];
        for (const [dx, dy] of ring) {
          const xx = x + dx;
          const yy = y + dy;
          if (xx < 0 || yy < 0 || xx >= w || yy >= h) continue;
          total[c]++;
          if (q[yy * w + xx] === c) same[c]++;
        }
      }
    }
    for (let c = 0; c < 64; c++) out[k * 64 + c] = total[c] ? same[c] / total[c] : 0;
  });
  return out;
}

/** Huang et al.'s relative L1 distance, summed over colours and distances (0 = same). */
export function correlogramDistance(a, b) {
  let s = 0;
  for (let i = 0; i < a.length; i++) s += Math.abs(a[i] - b[i]) / (1 + a[i] + b[i]);
  return s;
}

// ---------------------------------------------------------------- dominant colours

/**
 * Up to k dominant colours in CIELAB by k-means (Lloyd), started from pixels
 * spread evenly through the image's brightness order so the result does not
 * depend on chance. Returns [{ lab, weight }].
 */
export function dominantColors(img, k = 8, iterations = 30) {
  const lab = rgbToLab(img.rgb);
  const n = lab.length / 3;
  const order = Array.from({ length: n }, (_, i) => i).sort((p, q) => lab[p * 3] - lab[q * 3] || p - q);
  let centres = Array.from({ length: Math.min(k, n) }, (_, i) => {
    const p = order[Math.floor(((i + 0.5) * n) / k)];
    return [lab[p * 3], lab[p * 3 + 1], lab[p * 3 + 2]];
  });
  const assign = new Int32Array(n).fill(-1);
  for (let it = 0; it < iterations; it++) {
    let changed = false;
    for (let i = 0; i < n; i++) {
      let best = 0;
      let bd = Infinity;
      for (let c = 0; c < centres.length; c++) {
        const d = (lab[i * 3] - centres[c][0]) ** 2 + (lab[i * 3 + 1] - centres[c][1]) ** 2 + (lab[i * 3 + 2] - centres[c][2]) ** 2;
        if (d < bd) {
          bd = d;
          best = c;
        }
      }
      if (assign[i] !== best) {
        assign[i] = best;
        changed = true;
      }
    }
    if (!changed) break;
    const sums = centres.map(() => [0, 0, 0, 0]);
    for (let i = 0; i < n; i++) {
      const s = sums[assign[i]];
      s[0] += lab[i * 3];
      s[1] += lab[i * 3 + 1];
      s[2] += lab[i * 3 + 2];
      s[3]++;
    }
    centres = centres.map((c, j) => (sums[j][3] ? [sums[j][0] / sums[j][3], sums[j][1] / sums[j][3], sums[j][2] / sums[j][3]] : c));
  }
  const counts = new Float64Array(centres.length);
  for (let i = 0; i < n; i++) counts[assign[i]]++;
  return centres.map((lab3, j) => ({ lab: lab3, weight: counts[j] / n })).filter((c) => c.weight > 0);
}

/**
 * Exact transportation problem (MODI method on a spanning-tree basis):
 * the cheapest way to move supplies a onto demands b at unit costs C.
 */
export function transport(a, b, C) {
  const m = a.length;
  const n = b.length;
  const x = Array.from({ length: m }, () => new Float64Array(n));
  const basic = Array.from({ length: m }, () => new Uint8Array(n));
  const ra = Array.from(a);
  const rb = Array.from(b);
  // north-west corner start with exactly m + n − 1 basic cells
  let i = 0;
  let j = 0;
  for (;;) {
    const q = Math.min(ra[i], rb[j]);
    x[i][j] = q;
    basic[i][j] = 1;
    ra[i] -= q;
    rb[j] -= q;
    if (i === m - 1 && j === n - 1) break;
    if (j === n - 1 || (i < m - 1 && ra[i] <= rb[j])) i++;
    else j++;
  }
  const eps = 1e-12;
  for (let iter = 0; iter < 1000; iter++) {
    // potentials u_i + v_j = C_ij on basic cells
    const u = new Float64Array(m).fill(NaN);
    const v = new Float64Array(n).fill(NaN);
    u[0] = 0;
    for (let changed = true; changed; ) {
      changed = false;
      for (let r = 0; r < m; r++) {
        for (let c = 0; c < n; c++) {
          if (!basic[r][c]) continue;
          if (!Number.isNaN(u[r]) && Number.isNaN(v[c])) {
            v[c] = C[r][c] - u[r];
            changed = true;
          } else if (Number.isNaN(u[r]) && !Number.isNaN(v[c])) {
            u[r] = C[r][c] - v[c];
            changed = true;
          }
        }
      }
    }
    let best = -eps;
    let ei = -1;
    let ej = -1;
    for (let r = 0; r < m; r++) {
      for (let c = 0; c < n; c++) {
        if (basic[r][c]) continue;
        const rc = C[r][c] - u[r] - v[c];
        if (rc < best) {
          best = rc;
          ei = r;
          ej = c;
        }
      }
    }
    if (ei < 0) break;
    // path in the basis tree from column ej to row ei (nodes: rows 0..m-1, columns m..m+n-1)
    const prev = new Int32Array(m + n).fill(-2);
    prev[m + ej] = -1;
    const queue = [m + ej];
    while (queue.length && prev[ei] === -2) {
      const node = queue.shift();
      if (node >= m) {
        const c = node - m;
        for (let r = 0; r < m; r++) {
          if (basic[r][c] && prev[r] === -2) {
            prev[r] = node;
            queue.push(r);
          }
        }
      } else {
        for (let c = 0; c < n; c++) {
          if (basic[node][c] && prev[m + c] === -2) {
            prev[m + c] = node;
            queue.push(m + c);
          }
        }
      }
    }
    // cells along the path alternate −, +, −, … starting next to column ej
    const cells = [];
    for (let node = ei; prev[node] !== -1; node = prev[node]) {
      const other = prev[node];
      cells.push(node < m ? [node, other - m] : [other, node - m]);
    }
    cells.reverse();
    let theta = Infinity;
    let leave = null;
    cells.forEach(([r, c], k) => {
      if (k % 2 === 0 && x[r][c] < theta) {
        theta = x[r][c];
        leave = [r, c];
      }
    });
    x[ei][ej] = theta;
    basic[ei][ej] = 1;
    cells.forEach(([r, c], k) => {
      x[r][c] += k % 2 === 0 ? -theta : theta;
    });
    basic[leave[0]][leave[1]] = 0;
    x[leave[0]][leave[1]] = 0;
  }
  let cost = 0;
  for (let r = 0; r < m; r++) for (let c = 0; c < n; c++) cost += x[r][c] * C[r][c];
  return { cost, flow: x };
}

/** Earth Mover's Distance between two palettes, in CIELAB ΔE units (Rubner, Tomasi & Guibas 2000). */
export function paletteEmd(pa, pb) {
  const sa = pa.reduce((s, c) => s + c.weight, 0);
  const sb = pb.reduce((s, c) => s + c.weight, 0);
  const C = pa.map((p) => pb.map((q) => Math.hypot(p.lab[0] - q.lab[0], p.lab[1] - q.lab[1], p.lab[2] - q.lab[2])));
  return transport(
    pa.map((c) => c.weight / sa),
    pb.map((c) => c.weight / sb),
    C,
  ).cost;
}

/** CIEDE2000 between the average CIELAB colours of two images. */
export function meanColorDifference(imgA, imgB) {
  const mean = (img) => {
    const lab = rgbToLab(img.rgb);
    const n = lab.length / 3;
    const m = [0, 0, 0];
    for (let i = 0; i < n; i++) for (let c = 0; c < 3; c++) m[c] += lab[i * 3 + c] / n;
    return m;
  };
  const a = mean(imgA);
  const b = mean(imgB);
  return { value: ciede2000(...a, ...b), a, b };
}
