// Neural similarity models. Preprocessing mirrors each model's reference
// Python pipeline (torchvision / Hugging Face, which resize with Pillow), and
// inference runs through onnxruntime-web.

import { cropImg, resizeImg, shortEdgeSize, toCHW } from './pixels.js';
import { roundHalfEven } from './resample.js';

const IMAGENET = { mean: [0.485, 0.456, 0.406], std: [0.229, 0.224, 0.225] };
const CLIP_NORM = { mean: [0.48145466, 0.4578275, 0.40821073], std: [0.26862954, 0.26130258, 0.27577711] };
const HALF = { mean: [0.5, 0.5, 0.5], std: [0.5, 0.5, 0.5] };

export const MODELS = {
  sscd: {
    label: 'SSCD ResNet-50 (sscd_disc_mixup)',
    // also outputs per-location contributions (see tools/export_sscd.py --explain)
    files: ['models/sscd_disc_mixup_explain.onnx'],
    bytes: 29045579,
    input: 'input',
    output: 'embedding',
  },
  sscdLarge: {
    label: 'SSCD ResNeXt-101 (sscd_disc_large)',
    files: ['models/sscd_disc_large.onnx'],
    bytes: 48906033,
    input: 'input',
    output: 'embedding',
  },
  clip: {
    label: 'CLIP ViT-B/32 image encoder',
    files: ['models/clip_vit_b32_vision.onnx.part1', 'models/clip_vit_b32_vision.onnx.part2'],
    bytes: 96052464,
    input: 'pixel_values',
    output: 'image_embeds',
  },
  dino: {
    label: 'DINOv2 ViT-S/14',
    files: ['models/dinov2_small.onnx'],
    bytes: 30307426,
    input: 'pixel_values',
    output: 'last_hidden_state',
  },
  lpips: {
    label: 'LPIPS (AlexNet, v0.1)',
    files: ['models/lpips_alex.onnx'],
    bytes: 2501705,
  },
  dfine: {
    label: 'D-FINE-M object detector (Objects365)',
    files: ['models/dfine_m_obj365.onnx'],
    bytes: 20900541,
  },
};

/** torchvision CenterCrop (offsets use Python's round-half-even). */
function centerCropTorch(img, size) {
  const top = roundHalfEven((img.h - size) / 2);
  const left = roundHalfEven((img.w - size) / 2);
  return cropImg(img, left, top, size, size);
}

/** Hugging Face image processors: shortest edge resize (bicubic) + center crop. */
function hfPreprocess(img, shortEdge, crop, norm) {
  const [w, h] = shortEdgeSize(img.w, img.h, shortEdge);
  const r = resizeImg(img, w, h, 'bicubic');
  const c = cropImg(r, Math.floor((w - crop) / 2), Math.floor((h - crop) / 2), crop, crop);
  return { data: toCHW(c, norm.mean, norm.std), dims: [1, 3, crop, crop] };
}

export const PREPROCESS = {
  // SSCD README "small_288": Resize(288) on the short edge, ImageNet normalisation
  sscd(img) {
    const [w, h] = shortEdgeSize(img.w, img.h, 288);
    const r = resizeImg(img, w, h, 'bilinear');
    return { data: toCHW(r, IMAGENET.mean, IMAGENET.std), dims: [1, 3, h, w] };
  },
  // Somepalli et al. (github.com/somepago/DCR): Resize(256), CenterCrop(224), Normalize(0.5, 0.5)
  sscdLarge(img) {
    const [w, h] = shortEdgeSize(img.w, img.h, 256);
    const c = centerCropTorch(resizeImg(img, w, h, 'bilinear'), 224);
    return { data: toCHW(c, HALF.mean, HALF.std), dims: [1, 3, 224, 224] };
  },
  clip: (img) => hfPreprocess(img, 224, 224, CLIP_NORM),
  dino: (img) => hfPreprocess(img, 256, 224, IMAGENET),
};

export function createSession(ort, bytes) {
  return ort.InferenceSession.create(bytes, { executionProviders: ['wasm'], graphOptimizationLevel: 'all' });
}

function l2normalize(v) {
  let n = 0;
  for (const x of v) n += x * x;
  n = Math.sqrt(n) || 1;
  return Float32Array.from(v, (x) => x / n);
}

/** L2-normalised embedding of one image. */
export async function embed(ort, session, key, img) {
  const m = MODELS[key];
  const { data, dims } = PREPROCESS[key](img);
  const out = await session.run({ [m.input]: new ort.Tensor('float32', data, dims) });
  let v = out[m.output].data;
  if (key === 'dino') v = v.subarray(0, 384); // CLS token of last_hidden_state
  return l2normalize(v);
}

export function cosine(a, b) {
  let s = 0;
  for (let i = 0; i < a.length; i++) s += a[i] * b[i];
  return s;
}

/**
 * SSCD descriptor plus the per-location terms that add up to it. For another
 * image's descriptor eB, sum over cells of (eB . contrib[:, cell]) / znorm is
 * the cosine similarity (up to a bias term near zero).
 */
export async function sscdWithContrib(ort, session, img) {
  const { data, dims } = PREPROCESS.sscd(img);
  const out = await session.run({ input: new ort.Tensor('float32', data, dims) });
  const [, , fh, fw] = out.contrib.dims;
  return { e: out.embedding.data.slice(), znorm: out.znorm.data[0], contrib: out.contrib.data, fh, fw };
}

/** Map of each cell's contribution to cos(e_self, eOther). */
export function sscdEvidence(self, eOther) {
  const n = self.fh * self.fw;
  const map = new Float32Array(n);
  for (let o = 0; o < eOther.length; o++) {
    const w = eOther[o] / self.znorm;
    const row = o * n;
    for (let i = 0; i < n; i++) map[i] += w * self.contrib[row + i];
  }
  return map;
}

/**
 * Dense DINOv2 patch features for the whole (uncropped) image, resized so
 * both sides are multiples of the 14-pixel patch size. Returns L2-normalised
 * 384-d features on a gw × gh grid.
 */
export async function dinoPatches(ort, session, img, maxSide = 448) {
  const s = maxSide / Math.max(img.w, img.h);
  const W = Math.max(14, Math.round((img.w * s) / 14) * 14);
  const H = Math.max(14, Math.round((img.h * s) / 14) * 14);
  const r = resizeImg(img, W, H, 'bicubic');
  const out = await session.run({ pixel_values: new ort.Tensor('float32', toCHW(r, IMAGENET.mean, IMAGENET.std), [1, 3, H, W]) });
  const gw = W / 14;
  const gh = H / 14;
  const d = 384;
  const all = out.last_hidden_state.data;
  const feats = new Float32Array(gw * gh * d);
  for (let p = 0; p < gw * gh; p++) {
    let n = 0;
    for (let k = 0; k < d; k++) n += all[(p + 1) * d + k] ** 2;
    n = Math.sqrt(n) || 1;
    for (let k = 0; k < d; k++) feats[p * d + k] = all[(p + 1) * d + k] / n;
  }
  return { feats, gw, gh, d };
}

/**
 * Patch correspondences between two images: best match (cosine) for every
 * patch in each direction, and mutual nearest neighbours.
 */
export function patchCorrespondence(A, B) {
  const { d } = A;
  const nA = A.gw * A.gh;
  const nB = B.gw * B.gh;
  const bestA = new Float32Array(nA).fill(-1);
  const argA = new Int32Array(nA);
  const bestB = new Float32Array(nB).fill(-1);
  const argB = new Int32Array(nB);
  for (let i = 0; i < nA; i++) {
    const ai = i * d;
    for (let j = 0; j < nB; j++) {
      const bj = j * d;
      let s = 0;
      for (let k = 0; k < d; k++) s += A.feats[ai + k] * B.feats[bj + k];
      if (s > bestA[i]) {
        bestA[i] = s;
        argA[i] = j;
      }
      if (s > bestB[j]) {
        bestB[j] = s;
        argB[j] = i;
      }
    }
  }
  const mutual = [];
  for (let i = 0; i < nA; i++) if (argB[argA[i]] === i) mutual.push([i, argA[i], bestA[i]]);
  return { bestA, bestB, mutual };
}

/** LPIPS distance between two same-sized images (inputs scaled to [-1, 1]). */
export async function lpipsDistance(ort, session, A, B) {
  const t = (img) => new ort.Tensor('float32', toCHW(img, HALF.mean, HALF.std), [1, 3, img.h, img.w]);
  const out = await session.run({ a: t(A), b: t(B) });
  return out.distance.data[0];
}
