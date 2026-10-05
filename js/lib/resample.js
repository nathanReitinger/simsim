// Pillow-compatible image resampling (port of Pillow's Resample.c for 8-bit
// images). Matching Pillow matters because the reference implementations of
// imagehash, torchvision and Hugging Face preprocessing all resize with it.

const PRECISION_BITS = 22; // 32 - 8 - 2, as in Pillow
const ONE = 1 << PRECISION_BITS;
const HALF = 1 << (PRECISION_BITS - 1);

function sinc(x) {
  if (x === 0) return 1;
  x *= Math.PI;
  return Math.sin(x) / x;
}

export const FILTERS = {
  box: { support: 0.5, fn: (x) => (x > -0.5 && x <= 0.5 ? 1 : 0) },
  bilinear: {
    support: 1,
    fn: (x) => {
      x = Math.abs(x);
      return x < 1 ? 1 - x : 0;
    },
  },
  bicubic: {
    support: 2,
    fn: (x) => {
      const a = -0.5;
      x = Math.abs(x);
      if (x < 1) return ((a + 2) * x - (a + 3)) * x * x + 1;
      if (x < 2) return (((x - 5) * x + 8) * x - 4) * a;
      return 0;
    },
  },
  lanczos: { support: 3, fn: (x) => (x >= -3 && x < 3 ? sinc(x) * sinc(x / 3) : 0) },
};

function precompute(inSize, outSize, filter) {
  const scale = inSize / outSize;
  const filterscale = Math.max(scale, 1);
  const support = filter.support * filterscale;
  const ksize = Math.ceil(support) * 2 + 1;
  const bounds = new Int32Array(outSize * 2);
  const kk = new Int32Array(outSize * ksize);
  const tmp = new Float64Array(ksize);
  const ss = 1 / filterscale;
  for (let xx = 0; xx < outSize; xx++) {
    const center = (xx + 0.5) * scale;
    let xmin = Math.trunc(center - support + 0.5);
    if (xmin < 0) xmin = 0;
    let xmax = Math.trunc(center + support + 0.5);
    if (xmax > inSize) xmax = inSize;
    xmax -= xmin;
    let ww = 0;
    for (let x = 0; x < xmax; x++) {
      const w = filter.fn((x + xmin - center + 0.5) * ss);
      tmp[x] = w;
      ww += w;
    }
    for (let x = 0; x < xmax; x++) {
      const v = ww !== 0 ? tmp[x] / ww : tmp[x];
      kk[xx * ksize + x] = v < 0 ? Math.trunc(-0.5 + v * ONE) : Math.trunc(0.5 + v * ONE);
    }
    bounds[xx * 2] = xmin;
    bounds[xx * 2 + 1] = xmax;
  }
  return { ksize, bounds, kk };
}

function clip8(v) {
  v = Math.floor(v / ONE);
  return v < 0 ? 0 : v > 255 ? 255 : v;
}

/**
 * Resize an interleaved 8-bit image (1, 3 or 4 channels) the way Pillow's
 * Image.resize does: horizontal pass, round to uint8, vertical pass.
 */
export function resize(src, w, h, channels, outW, outH, filterName = 'bilinear') {
  const filter = FILTERS[filterName];
  if (!filter) throw new Error(`unknown filter ${filterName}`);
  let cur = src;
  let curW = w;
  if (outW !== w) {
    const { ksize, bounds, kk } = precompute(w, outW, filter);
    const out = new Uint8Array(outW * h * channels);
    const acc = new Float64Array(channels);
    for (let y = 0; y < h; y++) {
      const rowIn = y * w * channels;
      const rowOut = y * outW * channels;
      for (let xx = 0; xx < outW; xx++) {
        const xmin = bounds[xx * 2];
        const xmax = bounds[xx * 2 + 1];
        const kOff = xx * ksize;
        acc.fill(HALF);
        for (let x = 0; x < xmax; x++) {
          const k = kk[kOff + x];
          const p = rowIn + (x + xmin) * channels;
          for (let c = 0; c < channels; c++) acc[c] += cur[p + c] * k;
        }
        for (let c = 0; c < channels; c++) out[rowOut + xx * channels + c] = clip8(acc[c]);
      }
    }
    cur = out;
    curW = outW;
  }
  if (outH !== h) {
    const { ksize, bounds, kk } = precompute(h, outH, filter);
    const rowLen = curW * channels;
    const out = new Uint8Array(rowLen * outH);
    const acc = new Float64Array(rowLen);
    for (let yy = 0; yy < outH; yy++) {
      const ymin = bounds[yy * 2];
      const ymax = bounds[yy * 2 + 1];
      const kOff = yy * ksize;
      acc.fill(HALF);
      for (let y = 0; y < ymax; y++) {
        const k = kk[kOff + y];
        const rowIn = (y + ymin) * rowLen;
        for (let i = 0; i < rowLen; i++) acc[i] += cur[rowIn + i] * k;
      }
      const rowOut = yy * rowLen;
      for (let i = 0; i < rowLen; i++) out[rowOut + i] = clip8(acc[i]);
    }
    cur = out;
  }
  return cur === src ? src.slice() : cur;
}

/** Python-style round-half-to-even, used by torchvision's center_crop. */
export function roundHalfEven(x) {
  const r = Math.round(x);
  return Math.abs(x % 1) === 0.5 && r % 2 !== 0 ? r - 1 : r;
}
