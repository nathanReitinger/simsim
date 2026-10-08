// Perceptual hashes. aHash/dHash/pHash/wHash follow the Python `imagehash`
// library (same resizing, thresholds and hex encoding), Blockhash follows
// blockhash-js, and PDQ follows Meta's reference implementation.

import { resize, roundHalfEven } from './resample.js';
import { cropImg, grayPIL, lumaFloat, median, resizeImg } from './pixels.js';

// ---------------------------------------------------------------- helpers

/** Bits (0/1 array, MSB first) -> hex string, as imagehash prints them. */
export function bitsToHex(bits) {
  let hex = '';
  for (let i = 0; i < bits.length; i += 4) {
    let n = 0;
    for (let j = 0; j < 4; j++) n = (n << 1) | (bits[i + j] || 0);
    hex += n.toString(16);
  }
  return hex;
}

export function hamming(a, b) {
  let d = 0;
  for (let i = 0; i < a.length; i++) d += a[i] !== b[i] ? 1 : 0;
  return d;
}

const grayCache = new WeakMap();

function grayResized(img, w, h) {
  let gray = grayCache.get(img);
  if (!gray) {
    gray = grayPIL(img);
    grayCache.set(img, gray);
  }
  return resize(gray, img.w, img.h, 1, w, h, 'lanczos');
}

// ---------------------------------------------------------------- imagehash

export function averageHash(img, size = 8) {
  const px = grayResized(img, size, size);
  let mean = 0;
  for (const v of px) mean += v;
  mean /= px.length;
  return Uint8Array.from(px, (v) => (v > mean ? 1 : 0));
}

export function differenceHash(img, size = 8) {
  const px = grayResized(img, size + 1, size);
  const bits = new Uint8Array(size * size);
  for (let y = 0; y < size; y++) {
    for (let x = 0; x < size; x++) {
      const row = y * (size + 1);
      bits[y * size + x] = px[row + x + 1] > px[row + x] ? 1 : 0;
    }
  }
  return bits;
}

const dctCache = new Map();
function dctMatrix(n) {
  // scipy.fftpack.dct type II, norm=None: y[k] = 2 * sum x[i] cos(pi k (2i+1) / 2n)
  if (!dctCache.has(n)) {
    const m = new Float64Array(n * n);
    for (let k = 0; k < n; k++) {
      for (let i = 0; i < n; i++) m[k * n + i] = 2 * Math.cos((Math.PI * k * (2 * i + 1)) / (2 * n));
    }
    dctCache.set(n, m);
  }
  return dctCache.get(n);
}

export function perceptualHash(img, size = 8, highfreq = 4) {
  const n = size * highfreq;
  const px = grayResized(img, n, n);
  const D = dctMatrix(n);
  // column DCT (axis 0) for the first `size` output rows
  const tmp = new Float64Array(size * n);
  for (let k = 0; k < size; k++) {
    for (let x = 0; x < n; x++) {
      let s = 0;
      for (let y = 0; y < n; y++) s += D[k * n + y] * px[y * n + x];
      tmp[k * n + x] = s;
    }
  }
  // row DCT (axis 1), keep the low-frequency size x size block
  const low = new Float64Array(size * size);
  for (let k = 0; k < size; k++) {
    for (let l = 0; l < size; l++) {
      let s = 0;
      for (let x = 0; x < n; x++) s += tmp[k * n + x] * D[l * n + x];
      low[k * size + l] = s;
    }
  }
  const med = median(low);
  return Uint8Array.from(low, (v) => (v > med ? 1 : 0));
}

export function waveletHash(img, size = 8) {
  // imagehash.whash with Haar wavelets and remove_max_haar_ll=True. Removing
  // the top-level LL coefficient subtracts the image mean, and the level-L
  // Haar LL coefficients are scaled block sums, so the hash compares block
  // sums against their median.
  const minSide = Math.min(img.w, img.h);
  const scale = Math.max(2 ** Math.floor(Math.log2(minSide)), size);
  const px = grayResized(img, scale, scale);
  let mean = 0;
  for (const v of px) mean += v / 255;
  mean /= px.length;
  const block = scale / size;
  const sums = new Float64Array(size * size);
  for (let y = 0; y < scale; y++) {
    const by = Math.floor(y / block);
    for (let x = 0; x < scale; x++) {
      sums[by * size + Math.floor(x / block)] += px[y * scale + x] / 255 - mean;
    }
  }
  const med = median(sums);
  return Uint8Array.from(sums, (v) => (v > med ? 1 : 0));
}

/** imagehash.phash_simple: DCT of the rows only, top-left 8×8 (skipping the DC column) against their mean. */
export function perceptualHashSimple(img, size = 8, highfreq = 4) {
  const n = size * highfreq;
  const px = grayResized(img, n, n);
  const D = dctMatrix(n);
  const low = new Float64Array(size * size);
  for (let y = 0; y < size; y++) {
    for (let k = 1; k <= size; k++) {
      let s = 0;
      for (let x = 0; x < n; x++) s += D[k * n + x] * px[y * n + x];
      low[y * size + k - 1] = s;
    }
  }
  let mean = 0;
  for (const v of low) mean += v;
  mean /= low.length;
  return Uint8Array.from(low, (v) => (v > mean ? 1 : 0));
}

/** imagehash.dhash_vertical: is each pixel of a 8×9 thumbnail brighter than the one above it? */
export function differenceHashVertical(img, size = 8) {
  const px = grayResized(img, size, size + 1);
  const bits = new Uint8Array(size * size);
  for (let y = 0; y < size; y++) {
    for (let x = 0; x < size; x++) bits[y * size + x] = px[(y + 1) * size + x] > px[y * size + x] ? 1 : 0;
  }
  return bits;
}

// PyWavelets 'db4' decomposition low-pass filter
const DB4 = [-0.010597401785069032, 0.0328830116668852, 0.030841381835560764, -0.18703481171909309, -0.027983769416859854, 0.6308807679298589, 0.7148465705529157, 0.2303778133088965];

/** One level of pywt's low-pass DWT along rows of a (rows × n) array, 'symmetric' extension. */
function dwtLowRows(src, rows, n) {
  const F = DB4.length;
  const m = Math.floor((n + F - 1) / 2);
  const out = new Float64Array(rows * m);
  const sym = (i) => {
    while (i < 0 || i >= n) i = i < 0 ? -1 - i : 2 * n - 1 - i;
    return i;
  };
  for (let r = 0; r < rows; r++) {
    const o = r * n;
    for (let k = 0; k < m; k++) {
      let s = 0;
      for (let j = 0; j < F; j++) s += DB4[j] * src[o + sym(2 * k + 1 - j)];
      out[r * m + k] = s;
    }
  }
  return { data: out, n: m };
}

function transpose(src, rows, cols) {
  const out = new Float64Array(rows * cols);
  for (let r = 0; r < rows; r++) for (let c = 0; c < cols; c++) out[c * rows + r] = src[r * cols + c];
  return out;
}

/**
 * imagehash.whash with Daubechies-4 wavelets (mode='db4'): the low-frequency
 * band after log2(scale/8) levels of a 2-D db4 transform, against its median.
 * With pywt's symmetric padding the band is 14×14 for any image at least
 * 64 px on its short side; `scale` can be fixed so two hashes have one size.
 */
export function waveletHashDb4(img, size = 8, scale = null) {
  scale ??= Math.max(2 ** Math.floor(Math.log2(Math.min(img.w, img.h))), size);
  const levels = Math.log2(scale) - Math.log2(size);
  const px = grayResized(img, scale, scale);
  // remove_max_haar_ll: subtracting the full-depth Haar LL removes the mean
  let mean = 0;
  for (const v of px) mean += v / 255;
  mean /= px.length;
  let cur = Float64Array.from(px, (v) => v / 255 - mean);
  let rows = scale;
  let cols = scale;
  for (let l = 0; l < levels; l++) {
    // pywt.dwtn transforms axis 0 first, then axis 1
    const t = dwtLowRows(transpose(cur, rows, cols), cols, rows);
    const t2 = transpose(t.data, cols, t.n);
    const u = dwtLowRows(t2, t.n, cols);
    cur = u.data;
    rows = t.n;
    cols = u.n;
  }
  const med = median(cur);
  return { bits: Uint8Array.from(cur, (v) => (v > med ? 1 : 0)), side: rows, scale };
}

/** Pillow's RGB → HSV conversion (Convert.c rgb2hsv_row, float arithmetic). */
function hsvPIL(r, g, b) {
  const f = Math.fround;
  const maxc = Math.max(r, g, b);
  const minc = Math.min(r, g, b);
  if (minc === maxc) return [0, 0, maxc];
  const cr = maxc - minc;
  const s = f(cr / maxc);
  const rc = f((maxc - r) / cr);
  const gc = f((maxc - g) / cr);
  const bc = f((maxc - b) / cr);
  let h;
  if (r === maxc) h = f(bc - gc);
  else if (g === maxc) h = f(f(2 + rc) - bc);
  else h = f(f(4 + gc) - rc);
  h = f((h / 6 + 1) % 1);
  const clip = (v) => Math.max(0, Math.min(255, Math.trunc(v)));
  return [clip(h * 255), clip(s * 255), maxc];
}

/**
 * imagehash.colorhash (binbits = 3): the share of black pixels, of grey ones,
 * and of faint and of bright colours in six hue bands, each coded in 3 bits.
 */
export function colorHash(img, binbits = 3) {
  const { rgb } = img;
  const n = rgb.length / 3;
  const L = grayPIL(img);
  let black = 0;
  let gray = 0;
  let colors = 0;
  const faint = new Float64Array(6);
  const bright = new Float64Array(6);
  // numpy.histogram on linspace(0, 255, 7): half-open bins, the last one closed
  const hueBin = (h) => Math.min(5, Math.floor(h / 42.5));
  for (let i = 0; i < n; i++) {
    if (L[i] < 32) {
      black++;
      continue;
    }
    const [h, s] = hsvPIL(rgb[i * 3], rgb[i * 3 + 1], rgb[i * 3 + 2]);
    if (s < 85) {
      gray++;
      continue;
    }
    colors++;
    if (s < 170) faint[hueBin(h)]++;
    else if (s > 170) bright[hueBin(h)]++;
  }
  const max = 2 ** binbits;
  const c = Math.max(1, colors);
  const values = [Math.min(max - 1, Math.trunc((black / n) * max)), Math.min(max - 1, Math.trunc((gray / n) * max))];
  for (const count of [...faint, ...bright]) values.push(Math.min(max - 1, Math.trunc((count * max * 1) / c)));
  const bits = [];
  for (const v of values) for (let i = 0; i < binbits; i++) bits.push(Math.floor(v / 2 ** (binbits - i - 1)) % 2 ** (binbits - i) > 0 ? 1 : 0);
  return Uint8Array.from(bits);
}

// ---------------------------------------------------------------- crop-resistant hash

/** Pillow's GaussianBlur on an 8-bit plane: three passes of an extended box blur each way. */
function gaussianBlurPIL(src, w, h, radius = 2, passes = 3) {
  const f = Math.fround;
  const sigma2 = f(f(radius * radius) / passes);
  const L = f(Math.sqrt(12 * sigma2 + 1));
  const l = f(Math.floor((L - 1) / 2));
  let a = f(f(2 * l + 1) * f(f(l * f(l + 1)) - f(3 * sigma2)));
  a = f(a / f(6 * f(sigma2 - f(f(l + 1) * f(l + 1)))));
  const boxRadius = f(l + a);
  const blurRows = (data, width, height) => {
    const r = Math.trunc(boxRadius);
    const ww = Math.trunc(f(16777216 / f(boxRadius * 2 + 1)));
    const fw = Math.floor((16777216 - (r * 2 + 1) * ww) / 2);
    const edgeA = Math.min(r + 1, width);
    const edgeB = Math.max(width - r - 1, 0);
    const last = width - 1;
    const line = new Uint8Array(width);
    for (let y = 0; y < height; y++) {
      const o = y * width;
      const at = (x) => data[o + x];
      let acc = at(0) * (r + 1);
      for (let x = 0; x < edgeA - 1; x++) acc += at(x);
      acc += at(last) * (r - edgeA + 1);
      const save = (x, left, right) => {
        line[x] = Math.floor((acc * ww + (at(left) + at(right)) * fw + 8388608) / 16777216);
      };
      if (edgeA <= edgeB) {
        for (let x = 0; x < edgeA; x++) {
          acc += at(x + r) - at(0);
          save(x, 0, x + r + 1);
        }
        for (let x = edgeA; x < edgeB; x++) {
          acc += at(x + r) - at(x - r - 1);
          save(x, x - r - 1, x + r + 1);
        }
        for (let x = edgeB; x <= last; x++) {
          acc += at(last) - at(x - r - 1);
          save(x, x - r - 1, last);
        }
      } else {
        for (let x = 0; x < edgeB; x++) {
          acc += at(x + r) - at(0);
          save(x, 0, x + r + 1);
        }
        for (let x = edgeB; x < edgeA; x++) {
          acc += at(last) - at(0);
          save(x, 0, last);
        }
        for (let x = edgeA; x <= last; x++) {
          acc += at(last) - at(x - r - 1);
          save(x, x - r - 1, last);
        }
      }
      data.set(line, o);
    }
  };
  let img = Uint8Array.from(src);
  for (let p = 0; p < passes; p++) blurRows(img, w, h);
  const t = new Uint8Array(w * h);
  for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) t[x * h + y] = img[y * w + x];
  for (let p = 0; p < passes; p++) blurRows(t, h, w);
  img = new Uint8Array(w * h);
  for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) img[y * w + x] = t[x * h + y];
  return img;
}

/** Pillow's MedianFilter(3): the median of each 3×3 neighbourhood, edges repeated. */
function medianFilter3(src, w, h) {
  const out = new Uint8Array(w * h);
  const win = new Uint8Array(9);
  const cl = (v, n) => (v < 0 ? 0 : v >= n ? n - 1 : v);
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      let k = 0;
      for (let dy = -1; dy <= 1; dy++) for (let dx = -1; dx <= 1; dx++) win[k++] = src[cl(y + dy, h) * w + cl(x + dx, w)];
      win.sort();
      out[y * w + x] = win[4];
    }
  }
  return out;
}

/**
 * imagehash.crop_resistant_hash (Steinebach, Liu & Yannikos 2014, "Efficient
 * Cropping-Resistant Robust Image Hashing"): split a blurred 300×300 thumbnail
 * into bright "hills" and dark "valleys", and dHash the part of the image under
 * each segment larger than 500 pixels. Returns the segment boxes (in image
 * pixels) and their 64-bit dHashes.
 */
export function cropResistantHash(img, { threshold = 128, minSegment = 500, size = 300 } = {}) {
  const small = grayResized(img, size, size);
  const px = medianFilter3(gaussianBlurPIL(small, size, size), size, size);
  const N = size * size;
  const unassigned = new Uint8Array(N).fill(1);
  const hill = Uint8Array.from(px, (v) => (v > threshold ? 1 : 0));
  // imagehash counts the 4·size pixels just outside the border as already segmented
  let segmentedCount = 4 * size;
  const segments = [];
  const queue = new Int32Array(N);
  const component = (start, want) => {
    // 4-connected component of unassigned pixels of the wanted kind
    let head = 0;
    let tail = 0;
    queue[tail++] = start;
    unassigned[start] = 0;
    let minR = size;
    let maxR = -1;
    let minC = size;
    let maxC = -1;
    while (head < tail) {
      const p = queue[head++];
      const r = (p / size) | 0;
      const c = p % size;
      if (r < minR) minR = r;
      if (r > maxR) maxR = r;
      if (c < minC) minC = c;
      if (c > maxC) maxC = c;
      const visit = (q) => {
        if (unassigned[q] && hill[q] === want) {
          unassigned[q] = 0;
          queue[tail++] = q;
        }
      };
      if (r > 0) visit(p - size);
      if (r < size - 1) visit(p + size);
      if (c > 0) visit(p - 1);
      if (c < size - 1) visit(p + 1);
    }
    // a single-pixel region never enters imagehash's "segmented" set
    if (tail > 1) segmentedCount += tail;
    if (tail > minSegment) segments.push([minR, minC, maxR, maxC]);
  };
  for (let p = 0; p < N; p++) if (hill[p] && unassigned[p]) component(p, 1);
  for (let p = 0; p < N && segmentedCount < N; p++) if (!hill[p] && unassigned[p]) component(p, 0);
  if (!segments.length) segments.push([0, 0, size - 1, size - 1]);
  const sx = img.w / size;
  const sy = img.h / size;
  return segments.map(([r0, c0, r1, c1]) => {
    // PIL crops at Python-rounded (half-to-even) coordinates
    const x0 = roundHalfEven(c0 * sx);
    const y0 = roundHalfEven(r0 * sy);
    const x1 = roundHalfEven((c1 + 1) * sx);
    const y1 = roundHalfEven((r1 + 1) * sy);
    const box = [x0, y0, Math.max(x0 + 1, x1), Math.max(y0 + 1, y1)];
    return { box, bits: differenceHash(cropImg(img, box[0], box[1], box[2] - box[0], box[3] - box[1])) };
  });
}

/**
 * imagehash.ImageMultiHash.hash_diff: how many of A's segments have a segment
 * of B within `cutoff` bits (default 25% of 64), and the sum of those distances.
 */
export function cropResistantCompare(A, B, cutoff = 16) {
  let matches = 0;
  let sum = 0;
  for (const a of A) {
    let best = Infinity;
    for (const b of B) best = Math.min(best, hamming(a.bits, b.bits));
    if (best <= cutoff) {
      matches++;
      sum += best;
    }
  }
  return { matches, sum, total: A.length };
}

// ---------------------------------------------------------------- blockhash

function blockhashBits(blocks, pixelsPerBlock) {
  const half = (pixelsPerBlock * 256 * 3) / 2;
  const band = blocks.length / 4;
  const bits = new Uint8Array(blocks.length);
  for (let i = 0; i < 4; i++) {
    const m = median(blocks.slice(i * band, (i + 1) * band));
    for (let j = i * band; j < (i + 1) * band; j++) {
      const v = blocks[j];
      bits[j] = v > m || (Math.abs(v - m) < 1 && m > half) ? 1 : 0;
    }
  }
  return bits;
}

/** Blockhash (Yang, Gu & Niu 2006), `bmvbhash` from blockhash-js. */
export function blockHash(img, bitsPerSide = 16) {
  const { w, h, rgb } = img;
  const value = (x, y) => {
    const i = (y * w + x) * 3;
    return rgb[i] + rgb[i + 1] + rgb[i + 2];
  };
  const evenX = w % bitsPerSide === 0;
  const evenY = h % bitsPerSide === 0;
  const blocks = new Float64Array(bitsPerSide * bitsPerSide);

  if (evenX && evenY) {
    const bw = Math.floor(w / bitsPerSide);
    const bh = Math.floor(h / bitsPerSide);
    for (let by = 0; by < bitsPerSide; by++) {
      for (let bx = 0; bx < bitsPerSide; bx++) {
        let total = 0;
        for (let iy = 0; iy < bh; iy++) {
          for (let ix = 0; ix < bw; ix++) total += value(bx * bw + ix, by * bh + iy);
        }
        blocks[by * bitsPerSide + bx] = total;
      }
    }
    return blockhashBits(Array.from(blocks), bw * bh);
  }

  const blockW = w / bitsPerSide;
  const blockH = h / bitsPerSide;
  for (let y = 0; y < h; y++) {
    let top, bottom, wTop, wBottom;
    if (evenY) {
      top = bottom = Math.floor(y / blockH);
      wTop = 1;
      wBottom = 0;
    } else {
      const yMod = (y + 1) % blockH;
      const yFrac = yMod - Math.floor(yMod);
      const yInt = yMod - yFrac;
      wTop = 1 - yFrac;
      wBottom = yFrac;
      if (yInt > 0 || y + 1 === h) {
        top = bottom = Math.floor(y / blockH);
      } else {
        top = Math.floor(y / blockH);
        bottom = Math.ceil(y / blockH);
      }
    }
    for (let x = 0; x < w; x++) {
      const v = value(x, y);
      let left, right, wLeft, wRight;
      if (evenX) {
        left = right = Math.floor(x / blockW);
        wLeft = 1;
        wRight = 0;
      } else {
        const xMod = (x + 1) % blockW;
        const xFrac = xMod - Math.floor(xMod);
        const xInt = xMod - xFrac;
        wLeft = 1 - xFrac;
        wRight = xFrac;
        if (xInt > 0 || x + 1 === w) {
          left = right = Math.floor(x / blockW);
        } else {
          left = Math.floor(x / blockW);
          right = Math.ceil(x / blockW);
        }
      }
      blocks[top * bitsPerSide + left] += v * wTop * wLeft;
      blocks[top * bitsPerSide + right] += v * wTop * wRight;
      blocks[bottom * bitsPerSide + left] += v * wBottom * wLeft;
      blocks[bottom * bitsPerSide + right] += v * wBottom * wRight;
    }
  }
  return blockhashBits(Array.from(blocks), blockW * blockH);
}

// ---------------------------------------------------------------- PDQ

const PDQ_DCT = (() => {
  const d = new Float64Array(16 * 64);
  const scale = Math.sqrt(2 / 64);
  for (let i = 0; i < 16; i++) {
    for (let j = 0; j < 64; j++) d[i * 64 + j] = scale * Math.cos((Math.PI / 2 / 64) * (i + 1) * (2 * j + 1));
  }
  return d;
})();

function box1D(input, inOff, output, outOff, len, stride, win) {
  const half = Math.trunc((win + 2) / 2);
  const p1 = half - 1;
  const p2 = win - half + 1;
  const p3 = len - win;
  const p4 = half - 1;
  let li = 0;
  let ri = 0;
  let oi = 0;
  let sum = 0;
  let cur = 0;
  for (let i = 0; i < p1; i++) {
    sum += input[inOff + ri];
    cur++;
    ri += stride;
  }
  for (let i = 0; i < p2; i++) {
    sum += input[inOff + ri];
    cur++;
    output[outOff + oi] = sum / cur;
    ri += stride;
    oi += stride;
  }
  for (let i = 0; i < p3; i++) {
    sum += input[inOff + ri];
    sum -= input[inOff + li];
    output[outOff + oi] = sum / cur;
    li += stride;
    ri += stride;
    oi += stride;
  }
  for (let i = 0; i < p4; i++) {
    sum -= input[inOff + li];
    cur--;
    output[outOff + oi] = sum / cur;
    li += stride;
    oi += stride;
  }
}

function pdqBits(dct16) {
  const med = Float64Array.from(dct16).sort()[127]; // Torben median of 256 values
  return Uint8Array.from(dct16, (v) => (v > med ? 1 : 0));
}

/** The eight dihedral variants of the 16x16 DCT (rotations and flips). */
function pdqDihedral(A) {
  const out = {};
  const mk = (f) => {
    const B = new Float64Array(256);
    for (let i = 0; i < 16; i++) for (let j = 0; j < 16; j++) f(B, i, j, A[i * 16 + j]);
    return B;
  };
  out.original = A;
  out.rotate90 = mk((B, i, j, v) => (B[j * 16 + i] = j & 1 ? v : -v));
  out.rotate180 = mk((B, i, j, v) => (B[i * 16 + j] = (i + j) & 1 ? -v : v));
  out.rotate270 = mk((B, i, j, v) => (B[j * 16 + i] = i & 1 ? v : -v));
  out.flipX = mk((B, i, j, v) => (B[i * 16 + j] = i & 1 ? v : -v));
  out.flipY = mk((B, i, j, v) => (B[i * 16 + j] = j & 1 ? v : -v));
  out.flipPlus1 = mk((B, i, j, v) => (B[j * 16 + i] = v));
  out.flipMinus1 = mk((B, i, j, v) => (B[j * 16 + i] = (i + j) & 1 ? -v : v));
  return out;
}

/**
 * PDQ (Meta, 2019). Returns { bits, hex, quality, dihedral } where `bits[i*16+j]`
 * is the bit for DCT coefficient (i, j) and `dihedral` maps each of the eight
 * rotation/flip variants to its bit array.
 */
export function pdqHash(img) {
  let src = img;
  // Like the reference C++ tool, shrink large inputs to 512x512 first.
  if (img.w > 512 || img.h > 512) src = resizeImg(img, 512, 512, 'bilinear');
  const rows = src.h;
  const cols = src.w;
  const buf1 = lumaFloat(src);
  const buf2 = new Float64Array(rows * cols);
  const winRows = Math.trunc((cols + 127) / 128);
  const winCols = Math.trunc((rows + 127) / 128);
  for (let pass = 0; pass < 2; pass++) {
    for (let i = 0; i < rows; i++) box1D(buf1, i * cols, buf2, i * cols, cols, 1, winRows);
    for (let j = 0; j < cols; j++) box1D(buf2, j, buf1, j, rows, cols, winCols);
  }
  const b64 = new Float64Array(64 * 64);
  for (let i = 0; i < 64; i++) {
    const ini = Math.trunc(((i + 0.5) * rows) / 64);
    for (let j = 0; j < 64; j++) {
      const inj = Math.trunc(((j + 0.5) * cols) / 64);
      b64[i * 64 + j] = buf1[ini * cols + inj];
    }
  }
  // Image-domain quality metric (0-100); below 50 the hash is less reliable.
  let gradient = 0;
  for (let i = 0; i < 63; i++) {
    for (let j = 0; j < 64; j++) gradient += Math.abs(Math.trunc(((b64[i * 64 + j] - b64[(i + 1) * 64 + j]) * 100) / 255));
  }
  for (let i = 0; i < 64; i++) {
    for (let j = 0; j < 63; j++) gradient += Math.abs(Math.trunc(((b64[i * 64 + j] - b64[i * 64 + j + 1]) * 100) / 255));
  }
  const quality = Math.min(100, Math.trunc(gradient / 90));
  // 16x16 low-frequency DCT (skipping the DC row/column)
  const T = new Float64Array(16 * 64);
  for (let i = 0; i < 16; i++) {
    for (let j = 0; j < 64; j++) {
      let s = 0;
      for (let k = 0; k < 64; k++) s += PDQ_DCT[i * 64 + k] * b64[k * 64 + j];
      T[i * 64 + j] = s;
    }
  }
  const dct = new Float64Array(256);
  for (let i = 0; i < 16; i++) {
    for (let j = 0; j < 16; j++) {
      let s = 0;
      for (let k = 0; k < 64; k++) s += T[i * 64 + k] * PDQ_DCT[j * 64 + k];
      dct[i * 16 + j] = s;
    }
  }
  const variants = pdqDihedral(dct);
  const dihedral = {};
  for (const [name, m] of Object.entries(variants)) dihedral[name] = pdqBits(m);
  const bits = dihedral.original;
  return { bits, hex: pdqHex(bits), quality, dihedral };
}

/** PDQ hex: 16 words of 16 bits, most significant word (row 15) first. */
export function pdqHex(bits) {
  let hex = '';
  for (let i = 15; i >= 0; i--) {
    let word = 0;
    for (let j = 0; j < 16; j++) if (bits[i * 16 + j]) word |= 1 << j;
    hex += word.toString(16).padStart(4, '0');
  }
  return hex;
}
