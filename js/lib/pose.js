// Body pose: ViTPose-B (Xu et al., NeurIPS 2022) finds the 17 COCO keypoints
// of each person the object detector found; poses are then compared after
// removing position, size and rotation (Procrustes), also trying the mirror
// image, with an OKS-style score per joint.

export const JOINTS = ['nose', 'left eye', 'right eye', 'left ear', 'right ear', 'left shoulder', 'right shoulder', 'left elbow', 'right elbow', 'left wrist', 'right wrist', 'left hip', 'right hip', 'left knee', 'right knee', 'left ankle', 'right ankle'];
export const LIMBS = [
  [5, 7], [7, 9], [6, 8], [8, 10], [5, 6], [5, 11], [6, 12], [11, 12], [11, 13], [13, 15], [12, 14], [14, 16], [0, 1], [0, 2], [1, 3], [2, 4],
];
// left/right partner of each joint, for comparing against a mirror image
const FLIP = [0, 2, 1, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11, 14, 13, 16, 15];
// COCO keypoint sigmas (how precisely people agree on each joint)
const SIGMA = [0.026, 0.025, 0.025, 0.035, 0.035, 0.079, 0.079, 0.072, 0.072, 0.062, 0.062, 0.107, 0.107, 0.087, 0.087, 0.089, 0.089];

const W = 192;
const H = 256;
const MEAN = [0.485, 0.456, 0.406];
const STD = [0.229, 0.224, 0.225];

/** A person box expanded to the model's 3:4 aspect with 25% margin, and the crop as a tensor. */
export function poseInput(img, box) {
  const cx = (box[0] + box[2]) / 2;
  const cy = (box[1] + box[3]) / 2;
  let bw = box[2] - box[0];
  let bh = box[3] - box[1];
  if (bw > (W / H) * bh) bh = bw * (H / W);
  else bw = bh * (W / H);
  bw *= 1.25;
  bh *= 1.25;
  const x0 = cx - bw / 2;
  const y0 = cy - bh / 2;
  const sx = bw / W;
  const sy = bh / H;
  const plane = W * H;
  const data = new Float32Array(3 * plane);
  for (let y = 0; y < H; y++) {
    for (let x = 0; x < W; x++) {
      // bilinear sample at the centre of output pixel (x, y); black outside the image
      const fx = x0 + (x + 0.5) * sx - 0.5;
      const fy = y0 + (y + 0.5) * sy - 0.5;
      const ix = Math.floor(fx);
      const iy = Math.floor(fy);
      const tx = fx - ix;
      const ty = fy - iy;
      for (let c = 0; c < 3; c++) {
        let v = 0;
        for (const [dx, dy, w] of [
          [0, 0, (1 - tx) * (1 - ty)],
          [1, 0, tx * (1 - ty)],
          [0, 1, (1 - tx) * ty],
          [1, 1, tx * ty],
        ]) {
          const xx = ix + dx;
          const yy = iy + dy;
          if (xx >= 0 && yy >= 0 && xx < img.w && yy < img.h) v += w * img.rgb[(yy * img.w + xx) * 3 + c];
        }
        data[c * plane + y * W + x] = (v / 255 - MEAN[c]) / STD[c];
      }
    }
  }
  return { data, dims: [1, 3, H, W], x0, y0, sx, sy };
}

/** Keypoints [x, y, confidence] in image pixels from the 17 heatmaps (argmax + quarter-pixel refinement). */
export function decodeHeatmaps(heat, dims, input) {
  const [, J, hh, hw] = dims;
  const stride = W / hw;
  const out = [];
  for (let j = 0; j < J; j++) {
    const o = j * hh * hw;
    let best = 0;
    for (let i = 1; i < hh * hw; i++) if (heat[o + i] > heat[o + best]) best = i;
    let x = best % hw;
    let y = Math.floor(best / hw);
    const at = (xx, yy) => (xx >= 0 && yy >= 0 && xx < hw && yy < hh ? heat[o + yy * hw + xx] : heat[o + best]);
    const rx = x + 0.25 * Math.sign(at(x + 1, y) - at(x - 1, y));
    const ry = y + 0.25 * Math.sign(at(x, y + 1) - at(x, y - 1));
    x = (rx + 0.5) * stride;
    y = (ry + 0.5) * stride;
    out.push([input.x0 + x * input.sx, input.y0 + y * input.sy, heat[o + best]]);
  }
  return out;
}

/** Procrustes fit of B's joints onto A's (scale, rotation, translation) over the given joints. */
function fit(pa, pb, idx) {
  const n = idx.length;
  let ax = 0;
  let ay = 0;
  let bx = 0;
  let by = 0;
  for (const i of idx) {
    ax += pa[i][0];
    ay += pa[i][1];
    bx += pb[i][0];
    by += pb[i][1];
  }
  ax /= n;
  ay /= n;
  bx /= n;
  by /= n;
  let sxx = 0;
  let sxy = 0;
  let nb = 0;
  let na = 0;
  for (const i of idx) {
    const [x1, y1] = [pb[i][0] - bx, pb[i][1] - by];
    const [x2, y2] = [pa[i][0] - ax, pa[i][1] - ay];
    sxx += x1 * x2 + y1 * y2;
    sxy += x1 * y2 - y1 * x2;
    nb += x1 * x1 + y1 * y1;
    na += x2 * x2 + y2 * y2;
  }
  const r = Math.hypot(sxx, sxy);
  const s = r / Math.max(1e-9, nb);
  const cos = sxx / Math.max(1e-9, r);
  const sin = sxy / Math.max(1e-9, r);
  const map = ([x, y]) => [ax + s * (cos * (x - bx) - sin * (y - by)), ay + s * (sin * (x - bx) + cos * (y - by))];
  return { map, scaleA: Math.sqrt(na / n), angle: (Math.atan2(sin, cos) * 180) / Math.PI };
}

/**
 * Similarity of two poses (0–1): B is fitted onto A, then each joint scores
 * exp(-d² / 2(2σ·k)²) relative to the size of A's pose. Also tries B's
 * mirror image (left and right swapped) and keeps the better fit.
 */
export function compareSkeletons(pa, pb, { minConf = 0.3, tolerance = 1.5 } = {}) {
  const tryOne = (pbUse) => {
    const idx = [];
    for (let i = 0; i < 17; i++) if (pa[i][2] >= minConf && pbUse[i][2] >= minConf) idx.push(i);
    if (idx.length < 5) return null;
    const f = fit(pa, pbUse, idx);
    const scale = Math.max(1e-6, f.scaleA * tolerance);
    const perJoint = new Array(17).fill(null);
    let sum = 0;
    for (const i of idx) {
      const [x, y] = f.map(pbUse[i]);
      const d = Math.hypot(x - pa[i][0], y - pa[i][1]) / scale;
      const e = Math.exp(-(d * d) / (2 * (2 * SIGMA[i]) ** 2));
      perJoint[i] = { e, at: [x, y] };
      sum += e;
    }
    return { similarity: sum / idx.length, joints: idx.length, perJoint, angle: f.angle, map: f.map };
  };
  const direct = tryOne(pb);
  const mirrored = tryOne(pb.map((_, i) => [-pb[FLIP[i]][0], pb[FLIP[i]][1], pb[FLIP[i]][2]]));
  if (!direct && !mirrored) return null;
  if (mirrored && (!direct || mirrored.similarity > direct.similarity + 0.02)) return { ...mirrored, mirrored: true };
  return { ...direct, mirrored: false };
}
