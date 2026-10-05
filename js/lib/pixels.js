// Small helpers for working with raw pixel buffers.
// An "image" here is { w, h, rgb } with rgb an interleaved Uint8Array.

import { resize } from './resample.js';

/** RGBA -> RGB, compositing any transparency over white. */
export function rgbaToRgb(rgba, w, h) {
  const rgb = new Uint8Array(w * h * 3);
  for (let i = 0, j = 0; i < rgba.length; i += 4, j += 3) {
    const a = rgba[i + 3];
    if (a === 255) {
      rgb[j] = rgba[i];
      rgb[j + 1] = rgba[i + 1];
      rgb[j + 2] = rgba[i + 2];
    } else {
      const f = a / 255;
      const bg = 255 * (1 - f);
      rgb[j] = Math.round(rgba[i] * f + bg);
      rgb[j + 1] = Math.round(rgba[i + 1] * f + bg);
      rgb[j + 2] = Math.round(rgba[i + 2] * f + bg);
    }
  }
  return { w, h, rgb };
}

/** Pillow's RGB -> "L" conversion (ITU-R 601-2 luma, integer arithmetic). */
export function grayPIL(img) {
  const { w, h, rgb } = img;
  const out = new Uint8Array(w * h);
  for (let i = 0, j = 0; j < out.length; i += 3, j++) {
    out[j] = (rgb[i] * 19595 + rgb[i + 1] * 38470 + rgb[i + 2] * 7471 + 0x8000) >>> 16;
  }
  return out;
}

/** Floating point luma (0..255) with BT.601 weights. */
export function lumaFloat(img) {
  const { w, h, rgb } = img;
  const out = new Float64Array(w * h);
  for (let i = 0, j = 0; j < out.length; i += 3, j++) {
    out[j] = 0.299 * rgb[i] + 0.587 * rgb[i + 1] + 0.114 * rgb[i + 2];
  }
  return out;
}

/** Dimensions that fit inside maxSide (never upscales). */
export function fitWithin(w, h, maxSide) {
  const s = Math.min(1, maxSide / Math.max(w, h));
  return [Math.max(1, Math.round(w * s)), Math.max(1, Math.round(h * s))];
}

export function resizeImg(img, outW, outH, filter = 'bilinear') {
  if (outW === img.w && outH === img.h) return img;
  return { w: outW, h: outH, rgb: resize(img.rgb, img.w, img.h, 3, outW, outH, filter) };
}

/** torchvision-style Resize(size): shorter edge -> size, longer edge truncated. */
export function shortEdgeSize(w, h, size) {
  if (w <= h) return [size, Math.trunc((size * h) / w)];
  return [Math.trunc((size * w) / h), size];
}

/** Crop an interleaved RGB image. */
export function cropImg(img, left, top, cw, ch) {
  const out = new Uint8Array(cw * ch * 3);
  for (let y = 0; y < ch; y++) {
    const src = ((top + y) * img.w + left) * 3;
    out.set(img.rgb.subarray(src, src + cw * 3), y * cw * 3);
  }
  return { w: cw, h: ch, rgb: out };
}

/** Planar float tensor [3,H,W] normalised with per-channel mean/std. */
export function toCHW(img, mean, std, scale = 1 / 255) {
  const { w, h, rgb } = img;
  const n = w * h;
  const out = new Float32Array(3 * n);
  for (let i = 0; i < n; i++) {
    for (let c = 0; c < 3; c++) {
      out[c * n + i] = (rgb[i * 3 + c] * scale - mean[c]) / std[c];
    }
  }
  return out;
}

export function median(values) {
  const a = Float64Array.from(values).sort();
  const m = a.length >> 1;
  return a.length % 2 ? a[m] : (a[m - 1] + a[m]) / 2;
}
