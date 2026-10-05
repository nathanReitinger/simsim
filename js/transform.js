// Transform lab: produce an edited copy of an image (crop, rotate, recolour,
// blur, noise, captions, re-compression) to see which tests survive which edits.

export const DEFAULTS = {
  crop: 100, // % of width/height kept
  cropX: 50, // crop position, % of the free space
  cropY: 50,
  scale: 100, // % of the cropped size
  rotate: 0, // degrees
  flip: false, // mirror horizontally
  brightness: 0, // -100..100
  contrast: 0, // -100..100
  saturation: 100, // %
  hue: 0, // degrees
  grayscale: false,
  blur: 0, // px
  noise: 0, // std-dev in grey levels
  text: '',
  format: 'jpeg', // jpeg | png
  quality: 90, // JPEG quality 1..100
};

export const PRESETS = [
  { id: 'recompress', label: 'Heavy JPEG compression', params: { quality: 12 } },
  { id: 'resize', label: 'Downscaled to 40%', params: { scale: 40 } },
  { id: 'crop', label: 'Screenshot crop (lossless)', params: { crop: 55, cropX: 35, cropY: 40, format: 'png' } },
  { id: 'filter', label: 'Instagram-style filter', params: { crop: 92, contrast: 18, saturation: 140, hue: -12, brightness: 6, quality: 85 } },
  { id: 'meme', label: 'Meme caption', params: { text: 'ORIGINAL CONTENT, DO NOT STEAL', quality: 80 } },
  { id: 'mirror', label: 'Mirrored', params: { flip: true } },
  { id: 'rotate', label: 'Rotated 12° and cropped', params: { rotate: 12, crop: 80 } },
  { id: 'gray', label: 'Black & white + blur', params: { grayscale: true, blur: 2 } },
];

function mulberry32(seed) {
  return () => {
    seed |= 0;
    seed = (seed + 0x6d2b79f5) | 0;
    let t = Math.imul(seed ^ (seed >>> 15), 1 | seed);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

/** Colour adjustments using the same matrices as CSS filter functions. */
function adjustColors(d, p) {
  const b = p.brightness * 2.55;
  const c = (100 + p.contrast) / 100;
  const s = p.saturation / 100;
  const th = (p.hue * Math.PI) / 180;
  const cos = Math.cos(th);
  const sin = Math.sin(th);
  const hue = [
    0.213 + cos * 0.787 - sin * 0.213, 0.715 - cos * 0.715 - sin * 0.715, 0.072 - cos * 0.072 + sin * 0.928,
    0.213 - cos * 0.213 + sin * 0.143, 0.715 + cos * 0.285 + sin * 0.14, 0.072 - cos * 0.072 - sin * 0.283,
    0.213 - cos * 0.213 - sin * 0.787, 0.715 - cos * 0.715 + sin * 0.715, 0.072 + cos * 0.928 + sin * 0.072,
  ];
  const sat = [
    0.213 + 0.787 * s, 0.715 - 0.715 * s, 0.072 - 0.072 * s,
    0.213 - 0.213 * s, 0.715 + 0.285 * s, 0.072 - 0.072 * s,
    0.213 - 0.213 * s, 0.715 - 0.715 * s, 0.072 + 0.928 * s,
  ];
  for (let i = 0; i < d.length; i += 4) {
    let r = d[i];
    let g = d[i + 1];
    let bl = d[i + 2];
    if (p.hue) [r, g, bl] = [hue[0] * r + hue[1] * g + hue[2] * bl, hue[3] * r + hue[4] * g + hue[5] * bl, hue[6] * r + hue[7] * g + hue[8] * bl];
    if (p.saturation !== 100) [r, g, bl] = [sat[0] * r + sat[1] * g + sat[2] * bl, sat[3] * r + sat[4] * g + sat[5] * bl, sat[6] * r + sat[7] * g + sat[8] * bl];
    if (p.grayscale) r = g = bl = 0.2126 * r + 0.7152 * g + 0.0722 * bl;
    r = (r - 128) * c + 128 + b;
    g = (g - 128) * c + 128 + b;
    bl = (bl - 128) * c + 128 + b;
    d[i] = r;
    d[i + 1] = g;
    d[i + 2] = bl;
  }
}

/** Approximate Gaussian blur with three box-blur passes. */
function blur(d, w, h, radius) {
  const r = Math.max(1, Math.round(radius));
  const tmp = new Float32Array(w * h * 4);
  const src = Float32Array.from(d);
  const pass = (from, to, horizontal) => {
    const len = horizontal ? w : h;
    const lines = horizontal ? h : w;
    for (let l = 0; l < lines; l++) {
      for (let ch = 0; ch < 3; ch++) {
        let acc = 0;
        const at = (k) => {
          const kk = Math.min(len - 1, Math.max(0, k));
          return horizontal ? (l * w + kk) * 4 + ch : (kk * w + l) * 4 + ch;
        };
        for (let k = -r; k <= r; k++) acc += from[at(k)];
        for (let k = 0; k < len; k++) {
          to[at(k)] = acc / (2 * r + 1);
          acc += from[at(k + r + 1)] - from[at(k - r)];
        }
      }
    }
  };
  for (let i = 0; i < 3; i++) {
    pass(src, tmp, true);
    pass(tmp, src, false);
  }
  for (let i = 0; i < d.length; i += 4) {
    d[i] = src[i];
    d[i + 1] = src[i + 1];
    d[i + 2] = src[i + 2];
  }
}

function addNoise(d, sigma) {
  const rand = mulberry32(1234);
  for (let i = 0; i < d.length; i += 4) {
    for (let c = 0; c < 3; c++) {
      const u = Math.max(1e-12, rand());
      const v = rand();
      d[i + c] += sigma * Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
    }
  }
}

function makeCanvas(w, h) {
  if (typeof OffscreenCanvas !== 'undefined') return new OffscreenCanvas(w, h);
  const c = document.createElement('canvas');
  c.width = w;
  c.height = h;
  return c;
}

/** Apply the edits to an ImageBitmap; resolves with an encoded Blob. */
export async function applyTransform(bitmap, params) {
  const p = { ...DEFAULTS, ...params };
  const sw = bitmap.width;
  const sh = bitmap.height;
  const cw = Math.max(1, Math.round((sw * p.crop) / 100));
  const ch = Math.max(1, Math.round((sh * p.crop) / 100));
  const cx = Math.round(((sw - cw) * p.cropX) / 100);
  const cy = Math.round(((sh - ch) * p.cropY) / 100);
  const ow = Math.max(1, Math.round((cw * p.scale) / 100));
  const oh = Math.max(1, Math.round((ch * p.scale) / 100));

  const canvas = makeCanvas(ow, oh);
  const g = canvas.getContext('2d', { willReadFrequently: true });
  g.fillStyle = '#000';
  g.fillRect(0, 0, ow, oh);
  g.imageSmoothingEnabled = true;
  g.imageSmoothingQuality = 'high';
  g.save();
  g.translate(ow / 2, oh / 2);
  if (p.rotate) g.rotate((p.rotate * Math.PI) / 180);
  if (p.flip) g.scale(-1, 1);
  g.drawImage(bitmap, cx, cy, cw, ch, -ow / 2, -oh / 2, ow, oh);
  g.restore();

  const pixelOps = p.brightness || p.contrast || p.saturation !== 100 || p.hue || p.grayscale || p.blur || p.noise;
  if (pixelOps) {
    const id = g.getImageData(0, 0, ow, oh);
    const d = new Float32Array(id.data);
    adjustColors(d, p);
    if (p.blur > 0) blur(d, ow, oh, p.blur);
    if (p.noise > 0) addNoise(d, p.noise);
    for (let i = 0; i < d.length; i++) id.data[i] = d[i];
    g.putImageData(id, 0, 0);
  }

  if (p.text) {
    const size = Math.max(12, Math.round(ow / 16));
    g.font = `900 ${size}px Impact, "Arial Black", "Helvetica Neue", sans-serif`;
    g.textAlign = 'center';
    g.textBaseline = 'bottom';
    g.lineJoin = 'round';
    g.lineWidth = Math.max(2, size / 7);
    g.strokeStyle = '#000';
    g.fillStyle = '#fff';
    const lines = p.text.toUpperCase().split('\n');
    lines.forEach((line, i) => {
      const y = oh - size * 0.5 - (lines.length - 1 - i) * size * 1.1;
      g.strokeText(line, ow / 2, y, ow * 0.94);
      g.fillText(line, ow / 2, y, ow * 0.94);
    });
  }

  const type = p.format === 'png' ? 'image/png' : 'image/jpeg';
  const quality = Math.min(1, Math.max(0.01, p.quality / 100));
  if (canvas.convertToBlob) return canvas.convertToBlob({ type, quality });
  return new Promise((resolve) => canvas.toBlob(resolve, type, quality));
}

/** Short human description of the edits, used as a caption. */
export function describe(params) {
  const p = { ...DEFAULTS, ...params };
  const out = [];
  if (p.crop < 100) out.push(`cropped to ${p.crop}%`);
  if (p.scale !== 100) out.push(`scaled to ${p.scale}%`);
  if (p.rotate) out.push(`rotated ${p.rotate}°`);
  if (p.flip) out.push('mirrored');
  if (p.brightness) out.push(`brightness ${p.brightness > 0 ? '+' : ''}${p.brightness}`);
  if (p.contrast) out.push(`contrast ${p.contrast > 0 ? '+' : ''}${p.contrast}`);
  if (p.saturation !== 100) out.push(`saturation ${p.saturation}%`);
  if (p.hue) out.push(`hue ${p.hue > 0 ? '+' : ''}${p.hue}°`);
  if (p.grayscale) out.push('greyscale');
  if (p.blur) out.push(`blur ${p.blur}px`);
  if (p.noise) out.push(`noise σ${p.noise}`);
  if (p.text) out.push('caption added');
  out.push(p.format === 'png' ? 'saved as PNG' : `JPEG q${p.quality}`);
  return out.join(', ');
}
