// Perceptual hashes. aHash/dHash/pHash/wHash follow the Python `imagehash`
// library (same resizing, thresholds and hex encoding), Blockhash follows
// blockhash-js, and PDQ follows Meta's reference implementation.

import { resize } from './resample.js';
import { grayPIL, lumaFloat, median, resizeImg } from './pixels.js';

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
