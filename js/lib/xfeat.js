// XFeat (Potje et al., "XFeat: Accelerated Features for Lightweight Image
// Matching", CVPR 2024): the network runs in ONNX; this file reproduces the
// reference detectAndCompute post-processing and mutual-nearest-neighbour
// matching (modules/xfeat.py in verlab/accelerated_features).

import { resizeImg } from './pixels.js';

/** Image resized so both sides are multiples of 32, as a [1, 3, H, W] tensor (0–255). */
export function xfeatInput(img, maxSide = 640) {
  const s = Math.min(1, maxSide / Math.max(img.w, img.h));
  const W = Math.max(32, Math.floor((img.w * s) / 32) * 32);
  const H = Math.max(32, Math.floor((img.h * s) / 32) * 32);
  const r = resizeImg(img, W, H, 'bilinear');
  const plane = W * H;
  const data = new Float32Array(3 * plane);
  for (let i = 0, j = 0; i < plane; i++, j += 3) {
    data[i] = r.rgb[j];
    data[plane + i] = r.rgb[j + 1];
    data[2 * plane + i] = r.rgb[j + 2];
  }
  return { data, dims: [1, 3, H, W], W, H, sx: img.w / W, sy: img.h / H };
}

// grid_sample(align_corners=False) with XFeat's normalisation 2x/(size-1) - 1
const toGrid = (x, full, size) => (x * size) / (full - 1) - 0.5;

function bilinear(map, gw, gh, ch, c, fx, fy) {
  const x0 = Math.floor(fx);
  const y0 = Math.floor(fy);
  let v = 0;
  for (const [xx, yy, w] of [
    [x0, y0, (1 - (fx - x0)) * (1 - (fy - y0))],
    [x0 + 1, y0, (fx - x0) * (1 - (fy - y0))],
    [x0, y0 + 1, (1 - (fx - x0)) * (fy - y0)],
    [x0 + 1, y0 + 1, (fx - x0) * (fy - y0)],
  ]) {
    if (xx >= 0 && yy >= 0 && xx < gw && yy < gh) v += w * map[(c * gh + yy) * gw + xx];
  }
  return v;
}

// cubic convolution weights (A = -0.75, as in PyTorch)
function cubicWeights(t) {
  const A = -0.75;
  const w = (d) => {
    d = Math.abs(d);
    if (d <= 1) return ((A + 2) * d - (A + 3)) * d * d + 1;
    if (d < 2) return ((A * d - 5 * A) * d + 8 * A) * d - 4 * A;
    return 0;
  };
  return [w(1 + t), w(t), w(1 - t), w(2 - t)];
}

/**
 * Keypoints and 64-d descriptors from the network outputs, in the original
 * image's pixel coordinates.
 */
export function xfeatDetect(out, input, { topK = 2048, threshold = 0.05 } = {}) {
  const { W, H } = input;
  const [, , gh, gw] = out.keypoints.dims;
  const K = out.keypoints.data;
  const R = out.heatmap.data;
  const M = out.feats.data;
  const cells = gw * gh;
  // softmax over the 65 logits of each cell; the 64 positions form an H×W heat map
  const heat = new Float32Array(W * H);
  for (let cy = 0; cy < gh; cy++) {
    for (let cx = 0; cx < gw; cx++) {
      const cell = cy * gw + cx;
      let mx = -Infinity;
      for (let k = 0; k < 65; k++) mx = Math.max(mx, K[k * cells + cell]);
      let sum = 0;
      for (let k = 0; k < 65; k++) sum += Math.exp(K[k * cells + cell] - mx);
      for (let k = 0; k < 64; k++) heat[(cy * 8 + (k >> 3)) * W + cx * 8 + (k & 7)] = Math.exp(K[k * cells + cell] - mx) / sum;
    }
  }
  // 5×5 non-maximum suppression above the threshold
  const cand = [];
  for (let y = 0; y < H; y++) {
    for (let x = 0; x < W; x++) {
      const v = heat[y * W + x];
      if (!(v > threshold)) continue;
      let isMax = true;
      for (let dy = -2; dy <= 2 && isMax; dy++) {
        const yy = y + dy;
        if (yy < 0 || yy >= H) continue;
        for (let dx = -2; dx <= 2; dx++) {
          const xx = x + dx;
          if (xx >= 0 && xx < W && heat[yy * W + xx] > v) {
            isMax = false;
            break;
          }
        }
      }
      if (isMax) cand.push([x, y]);
    }
  }
  // score = keypoint probability (nearest) × reliability (bilinear), as in the reference
  const scored = cand.map(([x, y]) => {
    const nx = Math.round(toGrid(x, W, W));
    const ny = Math.round(toGrid(y, H, H));
    const kp = nx >= 0 && ny >= 0 && nx < W && ny < H ? heat[ny * W + nx] : 0;
    const rel = bilinear(R, gw, gh, 1, 0, toGrid(x, W, gw), toGrid(y, H, gh));
    return { x, y, s: kp * rel };
  });
  scored.sort((p, q) => q.s - p.s);
  const kept = scored.filter((p) => p.s > 0).slice(0, topK);
  const n = kept.length;
  // descriptors: L2-normalise the feature map, sample bicubically, normalise again
  const norm = new Float32Array(cells);
  for (let i = 0; i < cells; i++) {
    let s = 0;
    for (let c = 0; c < 64; c++) s += M[c * cells + i] ** 2;
    norm[i] = Math.max(1e-12, Math.sqrt(s));
  }
  const at = (c, xx, yy) => (xx < 0 || yy < 0 || xx >= gw || yy >= gh ? 0 : M[c * cells + yy * gw + xx] / norm[yy * gw + xx]);
  const desc = new Float32Array(n * 64);
  const kpts = new Float32Array(n * 2);
  kept.forEach((p, k) => {
    const fx = toGrid(p.x, W, gw);
    const fy = toGrid(p.y, H, gh);
    const x0 = Math.floor(fx);
    const y0 = Math.floor(fy);
    const wx = cubicWeights(fx - x0);
    const wy = cubicWeights(fy - y0);
    let s2 = 0;
    for (let c = 0; c < 64; c++) {
      let v = 0;
      for (let j = 0; j < 4; j++) {
        let row = 0;
        for (let i = 0; i < 4; i++) row += wx[i] * at(c, x0 - 1 + i, y0 - 1 + j);
        v += wy[j] * row;
      }
      desc[k * 64 + c] = v;
      s2 += v * v;
    }
    const inv = 1 / Math.max(1e-12, Math.sqrt(s2));
    for (let c = 0; c < 64; c++) desc[k * 64 + c] *= inv;
    kpts[k * 2] = p.x * input.sx;
    kpts[k * 2 + 1] = p.y * input.sy;
  });
  return { n, kpts, desc, scores: Float32Array.from(kept, (p) => p.s) };
}

/** Mutual nearest neighbours with cosine above minCos: [[iA, iB], ...]. */
export function xfeatMatch(A, B, minCos = 0.82) {
  const bestA = new Int32Array(A.n).fill(-1);
  const simA = new Float32Array(A.n).fill(-2);
  const bestB = new Int32Array(B.n).fill(-1);
  const simB = new Float32Array(B.n).fill(-2);
  for (let i = 0; i < A.n; i++) {
    const oa = i * 64;
    for (let j = 0; j < B.n; j++) {
      const ob = j * 64;
      let s = 0;
      for (let c = 0; c < 64; c++) s += A.desc[oa + c] * B.desc[ob + c];
      if (s > simA[i]) {
        simA[i] = s;
        bestA[i] = j;
      }
      if (s > simB[j]) {
        simB[j] = s;
        bestB[j] = i;
      }
    }
  }
  const out = [];
  for (let i = 0; i < A.n; i++) if (bestA[i] >= 0 && bestB[bestA[i]] === i && simA[i] > minCos) out.push([i, bestA[i]]);
  return out;
}
