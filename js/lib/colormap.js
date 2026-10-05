// Inferno colour map (matplotlib) for heat maps: dark = similar, bright = different.

const STOPS = [
  [0, 0, 4],
  [22, 11, 57],
  [66, 10, 104],
  [106, 23, 110],
  [147, 38, 103],
  [188, 55, 84],
  [221, 81, 58],
  [243, 120, 25],
  [252, 165, 10],
  [246, 215, 70],
  [252, 255, 164],
];

const LUT = new Uint8Array(256 * 3);
for (let i = 0; i < 256; i++) {
  const t = (i / 255) * (STOPS.length - 1);
  const k = Math.min(STOPS.length - 2, Math.floor(t));
  const f = t - k;
  for (let c = 0; c < 3; c++) LUT[i * 3 + c] = Math.round(STOPS[k][c] * (1 - f) + STOPS[k + 1][c] * f);
}

/** Colour of a value in 0..1 as [r, g, b]. */
export function lutColor(v) {
  const j = Math.round(Math.max(0, Math.min(1, v)) * 255) * 3;
  return [LUT[j], LUT[j + 1], LUT[j + 2]];
}

/** Values in 0..1 (clamped) -> RGBA buffer. Pixels where mask is 0 are dimmed. */
export function heatmap(values, w, h, mask = null) {
  const out = new Uint8ClampedArray(w * h * 4);
  for (let i = 0; i < w * h; i++) {
    const v = Math.max(0, Math.min(1, values[i]));
    const j = Math.round(v * 255) * 3;
    out[i * 4] = LUT[j];
    out[i * 4 + 1] = LUT[j + 1];
    out[i * 4 + 2] = LUT[j + 2];
    out[i * 4 + 3] = mask && !mask[i] ? 40 : 255;
  }
  return { w, h, data: out };
}

export function rgbToRgba(img) {
  const out = new Uint8ClampedArray(img.w * img.h * 4);
  for (let i = 0, j = 0; i < img.rgb.length; i += 3, j += 4) {
    out[j] = img.rgb[i];
    out[j + 1] = img.rgb[i + 1];
    out[j + 2] = img.rgb[i + 2];
    out[j + 3] = 255;
  }
  return { w: img.w, h: img.h, data: out };
}

export function grayToRgba(gray, w, h, mask = null) {
  const out = new Uint8ClampedArray(w * h * 4);
  for (let i = 0; i < w * h; i++) {
    out[i * 4] = out[i * 4 + 1] = out[i * 4 + 2] = gray[i];
    out[i * 4 + 3] = mask && !mask[i] ? 0 : 255;
  }
  return { w, h, data: out };
}
