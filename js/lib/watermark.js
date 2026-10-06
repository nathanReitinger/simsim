// Stable Diffusion's invisible watermark: the "dwtDct" method of the
// invisible-watermark library (ShieldMnt), which the Stable Diffusion 1.x/2.x
// reference scripts and the SDXL pipeline in diffusers apply to every image.
//
// The encoder converts BGR to YUV, takes a one-level Haar transform of the U
// channel, and in each 4×4 block of the approximation band quantizes the
// largest coefficient (other than the first) so that its value modulo 36
// lands near 9 (bit 0) or 27 (bit 1). Bits repeat across blocks in row-major
// order. Decoding reads every block and votes per bit position.

const bytesToBits = (text) => Array.from(new TextEncoder().encode(text)).flatMap((b) => Array.from({ length: 8 }, (_, k) => (b >> (7 - k)) & 1));

export const SD_WATERMARKS = [
  { name: 'Stable Diffusion 1.x reference scripts (“StableDiffusionV1”)', bits: bytesToBits('StableDiffusionV1') },
  { name: 'Stable Diffusion 2.x reference scripts (“SDV2”)', bits: bytesToBits('SDV2') },
  { name: 'SDXL (Stability AI / diffusers)', bits: Array.from('101100111110110010010000011110111011000110011110', Number) },
];

const SCALE = 36;

/** One bit per 4×4 block of the Haar approximation of the U channel, in block order. */
export function blockBits(rgba, w, h) {
  const rows = Math.floor(h / 4) * 4;
  const cols = Math.floor(w / 4) * 4;
  if (rows < 64 || cols < 64) return null;
  // U of OpenCV's BGR->YUV, rounded to 8 bits as the encoder sees it
  const U = new Uint8ClampedArray(rows * cols);
  for (let y = 0; y < rows; y++) {
    for (let x = 0; x < cols; x++) {
      const o = (y * w + x) * 4;
      U[y * cols + x] = Math.round(-0.14713 * rgba[o] - 0.28886 * rgba[o + 1] + 0.436 * rgba[o + 2] + 128);
    }
  }
  // one-level Haar approximation: (a + b + c + d) / 2
  const ch = rows / 2;
  const cw = cols / 2;
  const cA = new Float32Array(ch * cw);
  for (let y = 0; y < ch; y++) {
    for (let x = 0; x < cw; x++) {
      const i = 2 * y * cols + 2 * x;
      cA[y * cw + x] = (U[i] + U[i + 1] + U[i + cols] + U[i + cols + 1]) / 2;
    }
  }
  const by = Math.floor(ch / 4);
  const bx = Math.floor(cw / 4);
  const bits = new Uint8Array(by * bx);
  let n = 0;
  for (let i = 0; i < by; i++) {
    for (let j = 0; j < bx; j++) {
      let best = -1;
      let val = 0;
      for (let k = 1; k < 16; k++) {
        const v = Math.abs(cA[(i * 4 + (k >> 2)) * cw + j * 4 + (k & 3)]);
        if (v > best) {
          best = v;
          val = v;
        }
      }
      bits[n++] = val % SCALE > 0.5 * SCALE ? 1 : 0;
    }
  }
  return bits;
}

/** Which known Stable Diffusion watermark (if any) the image carries. */
export function detectSDWatermark(rgba, w, h) {
  const seq = blockBits(rgba, w, h);
  if (!seq) return { checked: false, reason: 'image too small' };
  const results = SD_WATERMARKS.map((m) => {
    const L = m.bits.length;
    const ones = new Float64Array(L);
    const count = new Float64Array(L);
    for (let i = 0; i < seq.length; i++) {
      ones[i % L] += seq[i];
      count[i % L]++;
    }
    let agree = 0;
    for (let k = 0; k < L; k++) if ((ones[k] / Math.max(1, count[k]) > 127 / 255 ? 1 : 0) === m.bits[k]) agree++;
    return { name: m.name, bits: L, accuracy: agree / L };
  });
  results.sort((p, q) => q.accuracy - p.accuracy);
  return { checked: true, best: results[0], all: results };
}
