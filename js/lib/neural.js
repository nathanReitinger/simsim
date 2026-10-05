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
    files: ['models/sscd_disc_mixup.onnx'],
    bytes: 29046743,
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

/** LPIPS distance between two same-sized images (inputs scaled to [-1, 1]). */
export async function lpipsDistance(ort, session, A, B) {
  const t = (img) => new ort.Tensor('float32', toCHW(img, HALF.mean, HALF.std), [1, 3, img.h, img.w]);
  const out = await session.run({ a: t(A), b: t(B) });
  return out.distance.data[0];
}
