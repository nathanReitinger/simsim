// Analysis worker: decodes both images, runs every test in turn and streams
// each result back to the page as soon as it is ready.

import { ENGINE_BY_ID, verdictFor } from './engines.js';
import { md5, sha } from './lib/digest.js';
import { cropImg, fitWithin, grayPIL, lumaFloat, resizeImg, rgbaToRgb } from './lib/pixels.js';
import * as H from './lib/hashes.js';
import * as Q from './lib/iqa.js';
import * as G from './lib/histogram.js';
import * as F from './lib/features.js';
import * as N from './lib/neural.js';
import * as O from './lib/objects.js';
import * as Ann from './lib/annotate.js';
import * as P from './lib/provenance.js';
import { detectSDWatermark } from './lib/watermark.js';
import * as X from './lib/xfeat.js';
import { grayToRgba, heatmap, rgbToRgba } from './lib/colormap.js';
import exifr from '../vendor/exifr/exifr.full.esm.js';

const ROOT = new URL('../', import.meta.url);
const WORK_MAX = 4096; // longest side kept for hashing and embeddings
const PAIR_MAX = 512; // aligned pair for pixel / structural metrics
const FEATURE_MAX = 800; // keypoint detection
const LPIPS_MAX = 256;
const CROP_MAX = 640; // template search

const ORDER = [
  'sha256', 'sha1', 'md5', 'payload', 'pixels',
  'pdq', 'pdqDihedral', 'phash', 'dhash', 'ahash', 'whash', 'blockhash',
  'ssim', 'msssim', 'psnr', 'ncc', 'uqi', 'gmsd', 'deltaE', 'changed', 'carlini',
  'histCorrel', 'histChi', 'histIntersect', 'histBhatt', 'lumaEmd',
  'exif', 'cmi', 'lineage', 'jpegq', 'thumbnail', 'sdmark',
  'crop', 'orb', 'akaze', 'brisk', 'xfeat', 'alignedSsim',
  'sscd', 'sscdAligned', 'sscdLarge', 'dino', 'clip', 'dreamsim', 'lpips',
  'objects',
];

const post = (msg, transfer = []) => self.postMessage(msg, transfer);
const tick = () => new Promise((r) => setTimeout(r, 0));

// ---------------------------------------------------------------- runtimes

let ortPromise = null;
let cvPromise = null;

function loadOrt() {
  ortPromise ??= import('../vendor/ort/ort.wasm.min.js').then((ort) => {
    // explicit file URLs (renamed from .mjs so any static host serves them as JavaScript)
    ort.env.wasm.wasmPaths = {
      mjs: new URL('vendor/ort/ort-wasm-simd-threaded.js', ROOT).href,
      wasm: new URL('vendor/ort/ort-wasm-simd-threaded.wasm', ROOT).href,
    };
    ort.env.wasm.numThreads = self.crossOriginIsolated ? Math.min(4, navigator.hardwareConcurrency || 1) : 1;
    ort.env.logLevel = 'error';
    return ort;
  });
  return ortPromise;
}

function loadOpenCV() {
  cvPromise ??= (async () => {
    const res = await fetch(new URL('vendor/opencv/opencv.js', ROOT));
    if (!res.ok) throw new Error(`could not load OpenCV (HTTP ${res.status})`);
    (0, eval)(await res.text()); // UMD bundle; defines self.cv
    const mod = self.cv;
    if (mod && mod.Mat && !mod.then) return mod;
    return new Promise((resolve) => {
      if (typeof mod.then === 'function') {
        mod.then((m) => {
          delete m.then; // the Emscripten module is a thenable; unwrap it
          resolve(m);
        });
      } else {
        mod.onRuntimeInitialized = () => resolve(mod);
      }
    });
  })();
  cvPromise.catch(() => (cvPromise = null));
  return cvPromise;
}

// ---------------------------------------------------------------- models

const sessions = new Map();
const waiters = new Map();

function requestModel(key) {
  return new Promise((resolve, reject) => {
    waiters.set(key, { resolve, reject });
    post({ type: 'need-model', key });
  });
}

function session(key) {
  if (!sessions.has(key)) {
    const p = (async () => {
      const [ort, bytes] = await Promise.all([loadOrt(), requestModel(key)]);
      post({ type: 'model-status', key, status: 'initializing' });
      const s = await N.createSession(ort, bytes);
      post({ type: 'model-status', key, status: 'ready' });
      return s;
    })();
    p.catch(() => sessions.delete(key));
    sessions.set(key, p);
  }
  return sessions.get(key);
}

// ---------------------------------------------------------------- decoding

async function prepare(input) {
  const { bitmap } = input;
  const w = bitmap.width;
  const h = bitmap.height;
  const canvas = new OffscreenCanvas(w, h);
  const g = canvas.getContext('2d', { willReadFrequently: true });
  g.drawImage(bitmap, 0, 0);
  const rgba = g.getImageData(0, 0, w, h).data;
  bitmap.close();
  let hasAlpha = false;
  for (let i = 3; i < rgba.length; i += 4) {
    if (rgba[i] !== 255) {
      hasAlpha = true;
      break;
    }
  }
  let work = rgbaToRgb(rgba, w, h);
  if (Math.max(w, h) > WORK_MAX) {
    const [ww, hh] = fitWithin(w, h, WORK_MAX);
    work = resizeImg(work, ww, hh, 'box');
  }
  const bytes = new Uint8Array(input.bytes);
  let exif = null;
  try {
    exif = await exifr.parse(bytes, {
      tiff: true,
      ifd0: true,
      exif: true,
      gps: true,
      xmp: true,
      icc: false,
      iptc: false,
      jfif: false,
      ihdr: false,
      interop: false,
      makerNote: false,
      userComment: false,
      mergeOutput: true,
    });
  } catch {
    exif = null;
  }
  // provenance: container structure, namespaced metadata, embedded thumbnail
  let prov = null;
  try {
    const info = await P.parseContainer(bytes);
    const meta = await exifr
      .parse(bytes, { tiff: true, ifd0: true, exif: true, xmp: true, iptc: true, icc: false, gps: false, interop: false, makerNote: false, userComment: false, mergeOutput: false })
      .catch(() => null);
    const thumb = await exifr.thumbnail(bytes).catch(() => null);
    prov = { info, meta: meta || null, thumb: thumb && thumb.length ? new Uint8Array(thumb) : null };
  } catch (err) {
    console.warn('provenance parsing failed', err);
  }
  return { name: input.name, type: input.type, size: bytes.length, bytes, w, h, rgba, hasAlpha, work, exif: exif || null, prov };
}

// ---------------------------------------------------------------- context

class Ctx {
  constructor(a, b, disabled = []) {
    this.disabled = disabled;
    this.a = a;
    this.b = b;
    this.memo = new Map();
    this.features = {};
    this.visuals = {};
    this.info = { a: {}, b: {} };
  }

  get(key, fn) {
    if (!this.memo.has(key)) this.memo.set(key, fn());
    return this.memo.get(key);
  }

  /** A fitted inside 512 px, B stretched to the same size. */
  get pair() {
    return this.get('pair', () => {
      const [w, h] = atLeast(fitWithin(this.a.work.w, this.a.work.h, PAIR_MAX), 32);
      const A = resizeImg(this.a.work, w, h, 'bilinear');
      const B = resizeImg(this.b.work, w, h, 'bilinear');
      return { w, h, A, B, YA: lumaFloat(A), YB: lumaFloat(B), GA: grayPIL(A), GB: grayPIL(B) };
    });
  }

  get deltaE() {
    return this.get('deltaE', () => Q.deltaE2000Map(this.pair.A.rgb, this.pair.B.rgb));
  }

  hist(kind) {
    return this.get(`hist-${kind}`, () => {
      const p = this.pair;
      if (kind === 'hs') return [G.hsHistogram(p.A.rgb), G.hsHistogram(p.B.rgb)];
      if (kind === 'rgb') return [G.rgbHistogram(p.A.rgb), G.rgbHistogram(p.B.rgb)];
      return [G.grayHistogram(p.YA), G.grayHistogram(p.YB)];
    });
  }

  hash(side, kind) {
    return this.get(`hash-${side}-${kind}`, () => {
      let img = this[side].work;
      if (kind === 'blockhash' && (img.w < 16 || img.h < 16)) img = resizeImg(img, ...atLeast([img.w, img.h], 16), 'bilinear');
      const fn = { ahash: H.averageHash, dhash: H.differenceHash, phash: H.perceptualHash, whash: H.waveletHash, blockhash: H.blockHash, pdq: H.pdqHash }[kind];
      const out = fn(img);
      this.info[side][kind] = kind === 'pdq' ? `${out.hex} (quality ${out.quality})` : H.bitsToHex(out);
      return out;
    });
  }

  featureImg(side) {
    return this.get(`feat-${side}`, () => {
      const img = this[side].work;
      const [w, h] = fitWithin(img.w, img.h, FEATURE_MAX);
      return { w, h, gray: grayPIL(resizeImg(img, w, h, 'bilinear')) };
    });
  }

  aspectNote() {
    const ra = this.a.w / this.a.h;
    const rb = this.b.w / this.b.h;
    if (Math.abs(Math.log(ra / rb)) < 0.02) return undefined;
    return 'Aspect ratios differ, so B was stretched to A’s shape before comparing.';
  }
}

// ---------------------------------------------------------------- runners

const exact = (x, y) => ({
  verdict: x === y ? 'identical' : 'none',
  display: x === y ? 'identical' : 'different',
  detail: { A: x, B: y },
});

const hamming64 = (kind) => (c) => {
  const a = c.hash('a', kind);
  const b = c.hash('b', kind);
  return { value: H.hamming(a, b), detail: { A: H.bitsToHex(a), B: H.bitsToHex(b) } };
};

/** XFeat keypoints and descriptors of one image (optionally mirrored), in feature-image coordinates. */
function xfeatFeatures(c, side, mirror = false) {
  return c.get(`xfeat-${side}-${mirror}`, async () => {
    const [ort, s] = await Promise.all([loadOrt(), session('xfeat')]);
    let img = c[side].work;
    if (mirror) {
      const rgb = new Uint8Array(img.rgb.length);
      for (let y = 0; y < img.h; y++) {
        for (let x = 0; x < img.w; x++) {
          const o = (y * img.w + x) * 3;
          const m = (y * img.w + img.w - 1 - x) * 3;
          rgb[o] = img.rgb[m];
          rgb[o + 1] = img.rgb[m + 1];
          rgb[o + 2] = img.rgb[m + 2];
        }
      }
      img = { w: img.w, h: img.h, rgb };
    }
    const inp = X.xfeatInput(img, 640);
    const out = await s.run({ image: new ort.Tensor('float32', inp.data, inp.dims) });
    const d = X.xfeatDetect(out, inp, { topK: 2048 });
    const f = c.featureImg(side);
    const kx = f.w / img.w;
    const ky = f.h / img.h;
    for (let i = 0; i < d.n; i++) {
      d.kpts[2 * i] *= kx;
      d.kpts[2 * i + 1] *= ky;
    }
    return d;
  });
}

/** Match XFeat features of A and (possibly mirrored) B and fit a homography B -> A. */
async function xfeatMatch(c, mirror = false) {
  const cv = await loadOpenCV();
  const [A, B] = await Promise.all([xfeatFeatures(c, 'a'), xfeatFeatures(c, 'b', mirror)]);
  const src = [];
  const dst = [];
  for (const [i, j] of X.xfeatMatch(A, B)) {
    src.push(B.kpts[2 * j], B.kpts[2 * j + 1]);
    dst.push(A.kpts[2 * i], A.kpts[2 * i + 1]);
  }
  const fb = c.featureImg('b');
  return { ...F.homographyFromMatches(cv, src, dst, fb.w, fb.h), keypointsA: A.n, keypointsB: B.n };
}

function featureVerdict(c, r) {
  const ratio = r.good ? r.inliers / r.good : 0;
  let verdict = 'none';
  if (r.sane && r.inliers >= 20 && ratio >= 0.5) verdict = 'match';
  else if (r.sane && r.inliers >= 10 && ratio >= 0.4) verdict = 'partial';
  return {
    value: r.inliers,
    verdict,
    detail: {
      'keypoints A / B': `${r.keypointsA} / ${r.keypointsB}`,
      'ratio-test matches': r.good,
      'RANSAC inliers': `${r.inliers} (${Math.round(ratio * 100)}%)`,
      transform: r.sane ? describeTransform(c, r.H) : r.H ? 'implausible (rejected)' : 'none found',
    },
  };
}

async function featureRun(c, kind) {
  const cv = await loadOpenCV();
  const r = F.matchFeatures(cv, c.featureImg('a'), c.featureImg('b'), kind);
  c.features[kind] = r;
  const ratio = r.good ? r.inliers / r.good : 0;
  let verdict = 'none';
  if (r.sane && r.inliers >= 20 && ratio >= 0.5) verdict = 'match';
  else if (r.sane && r.inliers >= 10 && ratio >= 0.4) verdict = 'partial';
  const detail = {
    'keypoints A / B': `${r.keypointsA} / ${r.keypointsB}`,
    'ratio-test matches': r.good,
    'RANSAC inliers': `${r.inliers} (${Math.round(ratio * 100)}%)`,
    transform: r.sane ? describeTransform(c, r.H) : r.H ? 'implausible (rejected)' : 'none found',
  };
  return { value: r.inliers, verdict, detail };
}

function describeTransform(c, Hm) {
  const fa = c.featureImg('a');
  const fb = c.featureImg('b');
  const scaleFeat = Math.sqrt(Math.abs(Hm[0] * Hm[4] - Hm[1] * Hm[3]));
  const scale = (scaleFeat * (c.a.w / fa.w)) / (c.b.w / fb.w);
  const rot = (Math.atan2(Hm[3], Hm[0]) * 180) / Math.PI;
  return `1 px of B ≈ ${scale.toFixed(2)} px of A, rotated ${rot.toFixed(1)}°`;
}

async function embedCompare(c, key) {
  const [ort, s] = await Promise.all([loadOrt(), session(key)]);
  const ea = await N.embed(ort, s, key, c.a.work);
  const eb = await N.embed(ort, s, key, c.b.work);
  return { value: N.cosine(ea, eb) };
}

function sameRegion(big, small, x0, y0) {
  if (x0 < 0 || y0 < 0 || x0 + small.w > big.w || y0 + small.h > big.h) return false;
  const rowLen = small.w * 4;
  for (let y = 0; y < small.h; y++) {
    const bo = ((y0 + y) * big.w + x0) * 4;
    const so = y * rowLen;
    for (let i = 0; i < rowLen; i++) if (big.rgba[bo + i] !== small.rgba[so + i]) return false;
  }
  return true;
}

// ---------------------------------------------------------------- objects

const mat3 = (A, B) => [
  A[0] * B[0] + A[1] * B[3] + A[2] * B[6], A[0] * B[1] + A[1] * B[4] + A[2] * B[7], A[0] * B[2] + A[1] * B[5] + A[2] * B[8],
  A[3] * B[0] + A[4] * B[3] + A[5] * B[6], A[3] * B[1] + A[4] * B[4] + A[5] * B[7], A[3] * B[2] + A[4] * B[5] + A[5] * B[8],
  A[6] * B[0] + A[7] * B[3] + A[8] * B[6], A[6] * B[1] + A[7] * B[4] + A[8] * B[7], A[6] * B[2] + A[7] * B[5] + A[8] * B[8],
];
const scale3 = (k) => [k, 0, 0, 0, k, 0, 0, 0, 1];
const union = (a, b) => [Math.min(a[0], b[0]), Math.min(a[1], b[1]), Math.max(a[2], b[2]), Math.max(a[3], b[3])];
const scaleBox = (b, k) => b.map((v) => v * k);
const MAX_OBJECT_PAIRS = 16;

/** Best keypoint homography (B work pixels -> A work pixels), if trustworthy. */
function workHomography(c) {
  const best = Object.values(c.features)
    .filter((r) => r.sane && r.inliers >= 20)
    .sort((x, y) => y.inliers - x.inliers)[0];
  const sa = c.featureImg('a').w / c.a.work.w;
  const sb = c.featureImg('b').w / c.b.work.w;
  if (best) return mat3(mat3(scale3(1 / sa), best.H), scale3(sb));
  if (!c.mirror) return null;
  // B matches only as a mirror image: flip B's feature coordinates first
  const fw = c.featureImg('b').w;
  const flip = [-1, 0, fw - 1, 0, 1, 0, 0, 0, 1];
  return mat3(mat3(scale3(1 / sa), mat3(c.mirror.H, flip)), scale3(sb));
}

/** When no transform maps B onto A, try B mirrored left to right. */
async function mirrorCheck(c) {
  c.mirror = null;
  if (Object.values(c.features).some((r) => r.sane && r.inliers >= 20)) return;
  const fb = c.featureImg('b');
  const gray = new Uint8Array(fb.gray.length);
  for (let y = 0; y < fb.h; y++) {
    const row = y * fb.w;
    for (let x = 0; x < fb.w; x++) gray[row + x] = fb.gray[row + fb.w - 1 - x];
  }
  const cv = await loadOpenCV();
  let best = null;
  for (const kind of ['orb', 'akaze']) {
    const r = F.matchFeatures(cv, c.featureImg('a'), { w: fb.w, h: fb.h, gray }, kind);
    if (r.sane && r.inliers >= 20 && (!best || r.inliers > best.inliers)) best = { H: r.H, inliers: r.inliers, kind };
  }
  if (!c.disabled?.includes('xfeat')) {
    const r = await xfeatMatch(c, true).catch(() => null);
    if (r && r.sane && r.inliers >= 20 && (!best || r.inliers > best.inliers)) best = { H: r.H, inliers: r.inliers, kind: 'xfeat' };
  }
  c.mirror = best;
}

function objectVerdict(aligned, m) {
  const sscdOk = m.sscd === undefined ? null : m.sscd;
  if (aligned) {
    if (m.changed <= 0.1 && !(m.ssim < 0.8)) return 'match';
    if ((sscdOk ?? 1) >= 0.5 || m.changed <= 0.4) return 'partial';
    return 'none';
  }
  if (sscdOk !== null) return sscdOk >= 0.75 ? 'match' : sscdOk >= 0.5 ? 'partial' : 'none';
  return m.combined >= 0.75 ? 'match' : m.combined >= 0.5 ? 'partial' : 'none';
}

async function compareObjects(c) {
  const [ort, det] = await Promise.all([loadOrt(), session('dfine')]);
  const A = c.a.work;
  const B = c.b.work;
  const detect = async (img) => {
    const out = await det.run({ pixel_values: new ort.Tensor('float32', O.detectorInput(img), [1, 3, O.DETECT_SIZE, O.DETECT_SIZE]) });
    return O.postprocess(out.logits.data, out.pred_boxes.data, img.w, img.h);
  };
  const detsA = await detect(A);
  const detsB = await detect(B);
  c.objectDets = { a: detsA, b: detsB };
  const toOrigA = c.a.w / A.w;
  const toOrigB = c.b.w / B.w;
  const visual = {
    a: { dets: detsA.map((d) => ({ label: d.label, score: d.score, box: scaleBox(d.box, toOrigA) })) },
    b: { dets: detsB.map((d) => ({ label: d.label, score: d.score, box: scaleBox(d.box, toOrigB) })) },
    pairs: [],
    onlyA: [],
    onlyB: [],
  };
  c.visuals.objects = visual;
  if (!detsA.length && !detsB.length) {
    visual.mode = 'none';
    return { verdict: 'na', display: 'no objects found', note: 'The detector found none of its 365 object categories in either image.' };
  }

  const Hw = workHomography(c);
  const aligned = !!Hw;
  visual.mode = aligned ? 'aligned' : 'appearance';
  let pairs;
  let boxesBinA = null;
  let Ao = null;
  let Bo = null;
  let maskO = null;
  let kA = 1;
  const cropsA = detsA.map((d) => O.cropBox(A, d.box));
  const cropsB = detsB.map((d) => O.cropBox(B, d.box));
  if (aligned) {
    boxesBinA = detsB.map((d) => O.mapBox(Hw, d.box));
    pairs = O.matchByPosition(detsA, boxesBinA, detsB);
    const [ow, oh] = fitWithin(A.w, A.h, 1024);
    kA = ow / A.w;
    Ao = resizeImg(A, ow, oh, 'bilinear');
    const warped = F.warpRgbInto(await loadOpenCV(), B, mat3(scale3(kA), Hw), ow, oh);
    Bo = warped.img;
    maskO = warped.mask;
  } else {
    const sims = new Map();
    pairs = O.matchByAppearance(detsA, detsB, (i, j) => {
      const r = O.cropSimilarity(cropsA[i], cropsB[j]);
      sims.set(`${i},${j}`, r);
      return r.combined;
    });
    for (const p of pairs) p.sim = sims.get(`${p.i},${p.j}`);
  }

  // Boxes of every detection in the comparison frame, used to keep changes to
  // a small object from counting against the larger object it sits on.
  const inner = aligned
    ? [
        ...detsA.map((d, i) => ({ id: `a${i}`, box: scaleBox(d.box, kA) })),
        ...boxesBinA.map((b, j) => ({ id: `b${j}`, box: scaleBox(b, kA) })),
      ]
    : [];
  const area = (b) => Math.max(0, b[2] - b[0]) * Math.max(0, b[3] - b[1]);
  const nestedIn = (region, skip) =>
    inner
      .filter((o) => !skip.includes(o.id) && area(o.box) < 0.7 * area(region))
      .filter((o) => {
        const w = Math.min(o.box[2], region[2]) - Math.max(o.box[0], region[0]);
        const h = Math.min(o.box[3], region[3]) - Math.max(o.box[1], region[1]);
        return w > 0 && h > 0 && (w * h) / area(o.box) > 0.5;
      })
      .map((o) => o.box);

  const useSscd = !c.disabled.includes('sscd');
  const sscd = useSscd ? await session('sscd') : null;
  const results = [];
  for (const p of pairs.slice(0, MAX_OBJECT_PAIRS)) {
    const m = {};
    const thumbs = {};
    if (aligned) {
      const region = scaleBox(union(detsA[p.i].box, boxesBinA[p.j]), kA);
      const r = O.compareRegion(Ao, Bo, maskO, region, nestedIn(region, [`a${p.i}`, `b${p.j}`]));
      if (r) {
        m.ssim = r.ssim;
        m.deltaE = r.deltaE;
        m.changed = r.changed;
        thumbs.a = rgbToRgba(O.thumb(r.crops.a));
        thumbs.b = rgbToRgba(O.thumb(r.crops.b));
        const heat = heatmap(r.deltaEMap.map((v) => v / 25), r.w, r.h, r.mask);
        const t = O.thumb({ w: r.w, h: r.h, rgb: rgbaToRgb(heat.data, r.w, r.h).rgb });
        thumbs.diff = rgbToRgba(t);
      }
    } else {
      m.colour = p.sim.hist;
      m.ssim = p.sim.ssim;
      m.combined = p.sim.combined;
    }
    if (!thumbs.a) {
      thumbs.a = rgbToRgba(O.thumb(cropsA[p.i]));
      thumbs.b = rgbToRgba(O.thumb(cropsB[p.j]));
    }
    if (sscd) m.sscd = N.cosine(await N.embed(ort, sscd, 'sscd', cropsA[p.i]), await N.embed(ort, sscd, 'sscd', cropsB[p.j]));
    if (aligned && m.changed === undefined) m.changed = 0;
    results.push({ i: p.i, j: p.j, label: detsA[p.i].label === detsB[p.j].label ? detsA[p.i].label : `${detsA[p.i].label} / ${detsB[p.j].label}`, metrics: m, verdict: objectVerdict(aligned, m), thumbs });
  }
  const rank = { none: 0, partial: 1, match: 2 };
  results.sort((x, y) => rank[x.verdict] - rank[y.verdict] || (y.metrics.changed ?? 0) - (x.metrics.changed ?? 0));
  results.forEach((r, k) => (r.n = k + 1));
  visual.pairs = results;

  // Objects found in only one image. When aligned, check whether that spot changed.
  const pairedA = new Set(pairs.map((p) => p.i));
  const pairedB = new Set(pairs.map((p) => p.j));
  const regionChange = (box, skip) => {
    if (!aligned) return null;
    const region = scaleBox(box, kA);
    return O.compareRegion(Ao, Bo, maskO, region, nestedIn(region, skip))?.changed ?? null;
  };
  visual.onlyA = detsA
    .map((d, i) => ({ i, d }))
    .filter(({ i }) => !pairedA.has(i))
    .map(({ i, d }) => ({ i, changed: regionChange(d.box, [`a${i}`]), thumb: rgbToRgba(O.thumb(cropsA[i])) }));
  visual.onlyB = detsB
    .map((d, j) => ({ j, d }))
    .filter(({ j }) => !pairedB.has(j))
    .map(({ j, d }) => ({ j, changed: regionChange(boxesBinA ? boxesBinA[j] : d.box, [`b${j}`]), thumb: rgbToRgba(O.thumb(cropsB[j])) }));

  const unchanged = results.filter((r) => r.verdict === 'match').length;
  const changed = results.length - unchanged;
  const total = Math.max(detsA.length, detsB.length);
  let verdict = 'none';
  if (unchanged / total >= 0.8) verdict = 'match';
  else if (pairs.length / total >= 0.5) verdict = 'partial';
  const unpaired = visual.onlyA.length + visual.onlyB.length;
  return {
    value: pairs.length,
    verdict,
    display: `${pairs.length} paired · ${changed} changed${unpaired ? ` · ${unpaired} unpaired` : ''}`,
    detail: {
      'objects in A / B': `${detsA.length} / ${detsB.length}`,
      'paired by': aligned ? 'position (images aligned with keypoints)' : 'kind and appearance (images could not be aligned)',
      'unchanged / changed': `${unchanged} / ${changed}`,
      'only in A / only in B': `${visual.onlyA.length} / ${visual.onlyB.length}`,
    },
    note: pairs.length > MAX_OBJECT_PAIRS ? `Only the first ${MAX_OBJECT_PAIRS} pairs were compared in detail.` : undefined,
  };
}

/** Strongest mutual matches, thinned so neighbouring patches don't all draw lines. */
function spreadLines(mutual, gw, minSim = 0.5, max = 90) {
  const picked = [];
  const taken = new Set();
  for (const [i, j, sim] of [...mutual].filter((m) => m[2] >= minSim).sort((x, y) => y[2] - x[2])) {
    const x = i % gw;
    const y = Math.floor(i / gw);
    let near = false;
    for (let dy = -1; dy <= 1 && !near; dy++) for (let dx = -1; dx <= 1; dx++) if (taken.has(`${x + dx},${y + dy}`)) near = true;
    if (near) continue;
    taken.add(`${x},${y}`);
    picked.push([i, j, sim]);
    if (picked.length >= max) break;
  }
  return picked;
}

const RUN = {
  sha256: async (c) => exact(await sha('SHA-256', c.a.bytes), await sha('SHA-256', c.b.bytes)),
  sha1: async (c) => exact(await sha('SHA-1', c.a.bytes), await sha('SHA-1', c.b.bytes)),
  md5: (c) => exact(md5(c.a.bytes), md5(c.b.bytes)),

  async payload(c) {
    // the encoded image data with every metadata segment set aside
    const pa = c.a.prov?.info;
    const pb = c.b.prov?.info;
    if (!pa?.payload || !pb?.payload) return { verdict: 'na', display: 'n/a', note: 'Only JPEG, PNG and WebP files can be split into image data and metadata.' };
    if (pa.format !== pb.format) return { verdict: 'none', display: 'different formats', detail: { A: pa.format.toUpperCase(), B: pb.format.toUpperCase() } };
    const [ha, hb] = await Promise.all([P.payloadHash(pa), P.payloadHash(pb)]);
    const same = ha && hb && ha.hex === hb.hex;
    const metaA = c.a.size - ha.bytes;
    const metaB = c.b.size - hb.bytes;
    const fileSame = c.a.size === c.b.size && c.a.bytes.every((v, i) => v === c.b.bytes[i]);
    return {
      verdict: same ? 'identical' : 'none',
      display: same ? (fileSame ? 'identical' : 'identical — only metadata differs') : 'different',
      detail: { 'image data A / B': `${ha.bytes.toLocaleString()} / ${hb.bytes.toLocaleString()} bytes`, 'metadata and headers A / B': `${metaA.toLocaleString()} / ${metaB.toLocaleString()} bytes`, 'segments A': pa.segments.join(' '), 'segments B': pb.segments.join(' ') },
      note: same && !fileSame ? 'The compressed picture is byte-for-byte the same; only metadata (credits, camera data, edit history) was added, changed or removed. B was not re-saved.' : undefined,
    };
  },

  pixels(c) {
    const { a, b } = c;
    if (a.w !== b.w || a.h !== b.h) {
      return { verdict: 'none', display: 'different size', detail: { A: `${a.w}×${a.h}`, B: `${b.w}×${b.h}` } };
    }
    let diff = 0;
    const n = a.w * a.h;
    for (let i = 0; i < a.rgba.length; i += 4) {
      if (a.rgba[i] !== b.rgba[i] || a.rgba[i + 1] !== b.rgba[i + 1] || a.rgba[i + 2] !== b.rgba[i + 2] || a.rgba[i + 3] !== b.rgba[i + 3]) diff++;
    }
    if (!diff) return { verdict: 'identical', display: 'all pixels equal', value: 1, detail: { pixels: n.toLocaleString() } };
    const share = diff / n;
    return {
      verdict: 'none',
      value: 1 - share,
      display: `${share < 0.001 ? '<0.1' : (share * 100).toFixed(1)}% differ`,
      detail: { 'differing pixels': `${diff.toLocaleString()} of ${n.toLocaleString()}` },
    };
  },

  pdq(c) {
    const a = c.hash('a', 'pdq');
    const b = c.hash('b', 'pdq');
    const low = Math.min(a.quality, b.quality);
    return {
      value: H.hamming(a.bits, b.bits),
      detail: { A: a.hex, B: b.hex, 'quality A / B': `${a.quality} / ${b.quality}` },
      note: low < 50 ? 'One image has little texture (PDQ quality < 50), so this hash is less reliable.' : undefined,
    };
  },
  pdqDihedral(c) {
    const a = c.hash('a', 'pdq');
    const b = c.hash('b', 'pdq');
    let best = Infinity;
    let which = 'original';
    for (const [name, bits] of Object.entries(b.dihedral)) {
      const d = H.hamming(a.bits, bits);
      if (d < best) {
        best = d;
        which = name;
      }
    }
    const names = { original: 'as is', rotate90: 'B rotated 90°', rotate180: 'B rotated 180°', rotate270: 'B rotated 270°', flipX: 'B flipped vertically', flipY: 'B mirrored', flipPlus1: 'B transposed', flipMinus1: 'B anti-transposed' };
    return { value: best, detail: { 'best orientation': names[which] } };
  },
  phash: hamming64('phash'),
  dhash: hamming64('dhash'),
  ahash: hamming64('ahash'),
  whash: hamming64('whash'),
  blockhash: (c) => {
    const a = c.hash('a', 'blockhash');
    const b = c.hash('b', 'blockhash');
    return { value: H.hamming(a, b), detail: { A: H.bitsToHex(a), B: H.bitsToHex(b) } };
  },

  ssim(c) {
    const p = c.pair;
    const r = Q.ssim(p.YA, p.YB, p.w, p.h);
    c.visuals.ssim = heatmap(r.map.map((v) => 1 - v), p.w, p.h);
    return { value: r.value, note: c.aspectNote() };
  },
  msssim(c) {
    const p = c.pair;
    const r = Q.msssim(p.YA, p.YB, p.w, p.h);
    return { value: r.value, detail: { scales: r.levels, 'per scale': r.perScale.map((v) => v.toFixed(3)).join(' · ') }, note: c.aspectNote() };
  },
  psnr(c) {
    const e = Q.pixelErrors(c.pair.A.rgb, c.pair.B.rgb);
    return { value: e.psnr, detail: { MSE: e.mse.toFixed(1), 'mean abs. error': `${e.mae.toFixed(2)} / 255` }, note: c.aspectNote() };
  },
  ncc: (c) => ({ value: Q.ncc(c.pair.YA, c.pair.YB), note: c.aspectNote() }),
  uqi(c) {
    // integer grey levels keep the flat-window special cases exact
    const p = c.pair;
    return { value: Q.uqi(Float64Array.from(p.GA), Float64Array.from(p.GB), p.w, p.h), note: c.aspectNote() };
  },
  gmsd: (c) => ({ value: Q.gmsd(c.pair.YA, c.pair.YB, c.pair.w, c.pair.h).value, note: c.aspectNote() }),
  deltaE(c) {
    const d = c.deltaE;
    const p = c.pair;
    c.visuals.deltaE = heatmap(d.map.map((v) => v / 25), p.w, p.h);
    return { value: d.mean, note: c.aspectNote() };
  },
  changed(c) {
    const m = c.deltaE.map;
    let n = 0;
    for (const v of m) if (v > 5) n++;
    return { value: n / m.length, note: c.aspectNote() };
  },

  cmi(c) {
    // copyright management information: who the files credit, and whether B kept it
    const fa = P.cmiFields(c.a.prov?.meta);
    const fb = P.cmiFields(c.b.prov?.meta);
    const ka = Object.keys(fa);
    const kb = Object.keys(fb);
    const detail = {};
    for (const k of new Set([...ka, ...kb])) detail[k] = `A: ${fa[k] || '—'} · B: ${fb[k] || '—'}`;
    if (!ka.length && !kb.length) return { verdict: 'na', display: 'none in either file', note: 'Neither file carries creator, copyright or credit fields.' };
    if (!ka.length) return { verdict: 'none', display: 'only B credits anyone', detail };
    const norm = (x) => String(x || '').toLowerCase().replace(/[^a-z0-9]+/g, ' ').trim();
    const kept = ka.filter((k) => fb[k] && (norm(fb[k]).includes(norm(fa[k])) || norm(fa[k]).includes(norm(fb[k]))));
    const removed = ka.filter((k) => !fb[k]);
    if (kept.length === ka.length) return { verdict: 'match', display: 'same credits', detail };
    if (!kb.length) {
      return {
        verdict: 'none',
        display: 'removed in B',
        detail,
        note: `A credits ${fa.creator || fa.credit || 'its author'}${fa.rights ? ` (“${fa.rights}”)` : ''}; B carries no credit or copyright fields at all. Removing copyright management information can be a separate violation (17 U.S.C. § 1202), though metadata is also routinely stripped by websites and apps.`,
      };
    }
    return { verdict: kept.length ? 'partial' : 'none', display: kept.length ? `${removed.length ? 'some removed' : 'some changed'}` : 'different credits', detail };
  },

  lineage(c) {
    // XMP edit history: does either file say it was made from the other?
    const la = P.lineage(c.a.prov?.meta);
    const lb = P.lineage(c.b.prov?.meta);
    const idsA = [la.documentID, la.instanceID, la.originalID].filter(Boolean);
    const idsB = [lb.documentID, lb.instanceID, lb.originalID].filter(Boolean);
    const genA = P.generatorText(c.a.prov?.info);
    const genB = P.generatorText(c.b.prov?.info);
    const detail = {};
    const show = (l, gen, side) => {
      const parts = [l.tool && `made with ${l.tool}`, l.documentID && `ID ${l.documentID}`, l.derived.length && `derived from ${l.derived.join(', ')}`, l.sourceType && `source type: ${l.sourceType}`, gen && `${gen.tool} settings: ${gen.text.slice(0, 160)}`].filter(Boolean);
      detail[side] = parts.join(' · ') || 'no edit history';
      if (l.history.length) detail[`history ${side}`] = l.history.slice(-4).join(' → ');
    };
    show(la, genA, 'A');
    show(lb, genB, 'B');
    const refs = (l) => [...l.derived, ...l.ancestors, ...l.ingredients];
    const bFromA = refs(lb).some((id) => idsA.includes(id));
    const aFromB = refs(la).some((id) => idsB.includes(id));
    const sameOrigin = la.originalID && la.originalID === lb.originalID;
    if (bFromA) return { verdict: 'match', display: 'B says it came from A', detail, note: 'B’s own edit history (XMP DerivedFrom / DocumentAncestors) names A’s document ID. Like all metadata it can be edited, but it is rarely faked.' };
    if (aFromB) return { verdict: 'match', display: 'A says it came from B', detail };
    if (sameOrigin) return { verdict: 'match', display: 'same original document', detail, note: 'Both files carry the same OriginalDocumentID: they are versions of one original.' };
    if (!idsA.length && !idsB.length && !genA && !genB) return { verdict: 'na', display: 'no edit history', detail, note: 'Neither file carries XMP document IDs or generator settings.' };
    return { verdict: 'none', display: 'no link', detail };
  },

  jpegq(c) {
    // which encoder settings last saved each file
    const a = c.a.prov?.info;
    const b = c.b.prov?.info;
    if (a?.format !== 'jpeg' || b?.format !== 'jpeg') return { verdict: 'na', display: 'n/a', note: 'Needs two JPEG files.' };
    const describe = (x) => `quality ≈ ${x.quality}${x.standardTables ? ' (standard libjpeg tables)' : ' (custom tables)'} · ${x.subsampling} · ${x.progressive ? 'progressive' : 'baseline'}`;
    const same = P.sameTables(a, b) && a.subsampling === b.subsampling;
    return {
      verdict: same ? 'match' : 'none',
      display: same ? `same tables (q≈${a.quality})` : `q≈${a.quality} vs q≈${b.quality}`,
      detail: { A: describe(a), B: describe(b) },
      note: same
        ? 'Identical quantization tables: both were saved by the same kind of encoder at the same setting.'
        : b.quality < a.quality
          ? 'B was saved at a lower quality — consistent with (but not proof of) B being a re-save of A.'
          : undefined,
    };
  },

  sdmark(c) {
    // Stable Diffusion's invisible watermark, read from the full-resolution pixels
    const found = {};
    const detail = {};
    for (const side of ['a', 'b']) {
      const r = detectSDWatermark(c[side].rgba, c[side].w, c[side].h);
      if (!r.checked) {
        detail[side.toUpperCase()] = r.reason;
        continue;
      }
      detail[side.toUpperCase()] = `${r.best.accuracy >= 0.9 ? 'found: ' : 'not found (best: '}${r.best.name}${r.best.accuracy >= 0.9 ? '' : ')'} — ${Math.round(r.best.accuracy * 100)}% of bits`;
      if (r.best.accuracy >= 0.9) found[side] = r.best.name;
    }
    const sides = Object.keys(found);
    if (!sides.length) {
      return { verdict: 'na', display: 'none found', detail, note: 'Absence proves nothing: most generators do not add this mark, and resizing, cropping or JPEG re-saving erases it.' };
    }
    const short = (n) => n.replace(/ \(.*\)$/, '').replace(' reference scripts', '');
    return {
      verdict: 'info',
      display: sides.length === 2 ? `both: ${short(found.a)}` : `${sides[0].toUpperCase()}: ${short(found[sides[0]])}`,
      detail,
      note: `${sides.map((s) => s.toUpperCase()).join(' and ')} ${sides.length === 2 ? 'carry' : 'carries'} Stable Diffusion’s invisible watermark: ${sides.length === 2 ? 'both images' : 'that image'} came out of a Stable Diffusion pipeline and ${sides.length === 2 ? 'have' : 'has'} not been resized or re-saved since.`,
    };
  },

  async thumbnail(c) {
    // embedded EXIF previews: do they show their own picture, or the other one?
    const ta = c.a.prov?.thumb;
    const tb = c.b.prov?.thumb;
    if (!ta && !tb) return { verdict: 'na', display: 'no thumbnails', note: 'Neither file embeds an EXIF preview image.' };
    const decode = async (bytes) => {
      const bm = await createImageBitmap(new Blob([bytes], { type: 'image/jpeg' }));
      const cv = new OffscreenCanvas(bm.width, bm.height);
      const g = cv.getContext('2d');
      g.drawImage(bm, 0, 0);
      const d = g.getImageData(0, 0, bm.width, bm.height).data;
      bm.close();
      return rgbaToRgb(d, cv.width, cv.height);
    };
    const dist = (x, y) => H.hamming(x, y);
    const hA = c.hash('a', 'dhash');
    const hB = c.hash('b', 'dhash');
    const detail = {};
    let cross = false;
    let stale = false;
    for (const [side, t, own, other] of [
      ['A', ta, hA, hB],
      ['B', tb, hB, hA],
    ]) {
      if (!t) continue;
      const th = H.differenceHash(await decode(t));
      const dOwn = dist(th, own);
      const dOther = dist(th, other);
      detail[`${side}’s preview`] = `${dOwn <= 10 ? 'matches' : 'does not match'} its own picture (${dOwn} bits) · ${dOther <= 10 ? 'matches' : 'differs from'} the other picture (${dOther} bits)`;
      if (dOwn > 10) stale = true;
      if (dOther <= 10 && dOther < dOwn) cross = true;
    }
    return {
      verdict: cross ? 'match' : 'none',
      display: cross ? 'preview shows the other image' : stale ? 'stale preview' : 'previews match their own images',
      detail,
      note: cross
        ? 'One file’s embedded preview shows the other picture, not itself: it was probably edited from that picture without regenerating the preview.'
        : stale
          ? 'An embedded preview does not match its own picture: the image was edited after the preview was written (a crop, for example).'
          : undefined,
    };
  },

  carlini(c) {
    // Carlini et al. (2023): both images at 512×512, split into 16 tiles of
    // 128×128; the distance is the largest per-tile RMS difference (pixels in
    // [0, 1]). At or below 0.15 they counted a generation as "extracted".
    const S = 512;
    const A = resizeImg(c.a.work, S, S, 'bilinear');
    const B = resizeImg(c.b.work, S, S, 'bilinear');
    const tiles = [];
    for (let ty = 0; ty < 4; ty++) {
      for (let tx = 0; tx < 4; tx++) {
        let sum = 0;
        for (let y = ty * 128; y < ty * 128 + 128; y++) {
          for (let x = tx * 128; x < tx * 128 + 128; x++) {
            const o = (y * S + x) * 3;
            for (let k = 0; k < 3; k++) {
              const d = (A.rgb[o + k] - B.rgb[o + k]) / 255;
              sum += d * d;
            }
          }
        }
        tiles.push(Math.sqrt(sum / (128 * 128 * 3)));
      }
    }
    const worst = tiles.indexOf(Math.max(...tiles));
    const rows = ['top', 'upper middle', 'lower middle', 'bottom'];
    const cols = ['left', 'centre-left', 'centre-right', 'right'];
    return {
      value: tiles[worst],
      detail: {
        'worst tile': `${rows[Math.floor(worst / 4)]} ${cols[worst % 4]} (${tiles[worst].toFixed(3)})`,
        'median tile': [...tiles].sort((x, y) => x - y)[7].toFixed(3),
      },
      note: c.aspectNote(),
    };
  },

  histCorrel: (c) => ({ value: G.correlation(...c.hist('hs')) }),
  histChi: (c) => ({ value: G.chiSquare(...c.hist('hs')) }),
  histIntersect: (c) => ({ value: G.intersection(...c.hist('rgb')) }),
  histBhatt: (c) => ({ value: G.bhattacharyya(...c.hist('hs')) }),
  lumaEmd: (c) => ({ value: G.emd1d(...c.hist('luma')) }),

  exif(c) {
    const ea = c.a.exif;
    const eb = c.b.exif;
    const summary = (e) => {
      if (!e) return 'none';
      const parts = [cameraName(e), e.DateTimeOriginal && fmtDate(e.DateTimeOriginal)].filter(Boolean);
      return parts.join(' · ') || 'present (no camera fields)';
    };
    const detail = { A: summary(ea), B: summary(eb) };
    const cam = (e) => [e.Make, e.Model, e.BodySerialNumber || e.SerialNumber].filter(Boolean).join('|');
    if (!ea || !eb || (!cam(ea) && !ea.DateTimeOriginal) || (!cam(eb) && !eb.DateTimeOriginal)) {
      return { verdict: 'na', display: 'no EXIF', detail, note: 'Needs capture metadata in both files; most web images have it stripped.' };
    }
    const same = (k) => ea[k] !== undefined && eb[k] !== undefined && String(ea[k]) === String(eb[k]);
    if (same('ImageUniqueID')) return { verdict: 'match', display: 'same image ID', detail };
    if (same('OriginalDocumentID') || same('DocumentID')) return { verdict: 'match', display: 'same source document', detail };
    const sameTime = ea.DateTimeOriginal && eb.DateTimeOriginal && fmtDate(ea.DateTimeOriginal) === fmtDate(eb.DateTimeOriginal) && (ea.SubSecTimeOriginal ?? '') === (eb.SubSecTimeOriginal ?? '');
    const sameCam = cam(ea) && cam(ea) === cam(eb);
    if (sameTime && sameCam) return { verdict: 'match', display: 'same camera & time', detail };
    if (sameCam) return { verdict: 'partial', display: 'same camera', detail };
    if (sameTime) return { verdict: 'partial', display: 'same capture time', detail };
    return { verdict: 'none', display: 'different', detail };
  },

  async crop(c) {
    const { a, b } = c;
    const aBig = a.w >= b.w && a.h >= b.h;
    const bBig = b.w >= a.w && b.h >= a.h;
    if ((a.w === b.w && a.h === b.h) || (!aBig && !bBig)) {
      return { verdict: 'na', display: 'n/a', note: 'Only applies when one image is smaller than the other in both dimensions.' };
    }
    const big = aBig ? a : b;
    const small = aBig ? b : a;
    const s = Math.min(1, CROP_MAX / Math.max(big.w, big.h));
    const scaled = (X) => {
      const w = Math.max(1, Math.round(X.w * s));
      const h = Math.max(1, Math.round(X.h * s));
      return { w, h, gray: grayPIL(resizeImg(X.work, w, h, 'box')) };
    };
    const S = scaled(small);
    if (S.w < 8 || S.h < 8) return { verdict: 'na', display: 'n/a', note: 'The smaller image is too small to search for.' };
    const cv = await loadOpenCV();
    const r = F.templateSearch(cv, scaled(big), S);
    let x = Math.round(r.x / s);
    let y = Math.round(r.y / s);
    let verbatim = false;
    if (r.score >= 0.97) {
      search: for (let dy = -2; dy <= 2; dy++) {
        for (let dx = -2; dx <= 2; dx++) {
          if (sameRegion(big, small, x + dx, y + dy)) {
            verbatim = true;
            x += dx;
            y += dy;
            break search;
          }
        }
      }
    }
    const where = `${aBig ? 'B' : 'A'} inside ${aBig ? 'A' : 'B'} at (${x}, ${y})`;
    return {
      value: r.score,
      verdict: verbatim ? 'identical' : undefined,
      display: verbatim ? 'verbatim crop' : undefined,
      detail: { 'best placement': where, area: `${((small.w * small.h) / (big.w * big.h) * 100).toFixed(1)}% of the larger image`, 'pixel-exact': verbatim ? 'yes' : 'no' },
    };
  },
  orb: (c) => featureRun(c, 'orb'),
  akaze: (c) => featureRun(c, 'akaze'),
  brisk: (c) => featureRun(c, 'brisk'),
  async alignedSsim(c) {
    const best = Object.entries(c.features)
      .filter(([, r]) => r.sane && r.inliers >= 10)
      .sort((x, y) => y[1].inliers - x[1].inliers)[0];
    if (!best) return { verdict: 'na', display: 'n/a', note: 'No consistent keypoint transform was found, so the images could not be aligned.' };
    const [kind, r] = best;
    const cv = await loadOpenCV();
    const p = c.pair;
    const fa = c.featureImg('a');
    const k = p.w / fa.w;
    const Hs = r.H.map((v, i) => (i < 6 ? v * k : v));
    const warped = F.warpInto(cv, c.featureImg('b'), Hs, p.w, p.h, 5);
    let covered = 0;
    for (const m of warped.mask) if (m) covered++;
    const coverage = covered / warped.mask.length;
    if (coverage < 0.05) return { verdict: 'na', display: 'n/a', note: 'The aligned images barely overlap.' };
    const XA = Float64Array.from(p.GA);
    const XB = Float64Array.from(warped.gray);
    const res = Q.ssim(XA, XB, p.w, p.h, warped.mask);
    const diff = XA.map((v, i) => Math.abs(v - XB[i]) / 96);
    c.visuals.aligned = {
      warped: grayToRgba(warped.gray, p.w, p.h, warped.mask),
      diff: heatmap(diff, p.w, p.h, warped.mask),
      via: kind.toUpperCase(),
    };
    return { value: res.value, detail: { 'aligned with': kind.toUpperCase(), 'overlap with A': `${Math.round(coverage * 100)}%` } };
  },

  objects: (c) => compareObjects(c),
  async sscd(c) {
    // SSCD with per-region evidence: the cells of each map add up to the score.
    const [ort, s] = await Promise.all([loadOrt(), session('sscd')]);
    const A = await N.sscdWithContrib(ort, s, c.a.work);
    const B = await N.sscdWithContrib(ort, s, c.b.work);
    const value = N.cosine(A.e, B.e);
    const ga = { gw: A.fw, gh: A.fh };
    const gb = { gw: B.fw, gh: B.fh };
    const evA = N.sscdEvidence(A, B.e);
    const pairs = N.sscdPairs(A, B);
    c.visuals.sscd = {
      score: value,
      a: { ...ga, values: evA },
      b: { ...gb, values: N.sscdEvidence(B, A.e) },
      pairs,
      links: Ann.copyLinks(pairs, evA, ga, gb, value),
    };
    return { value };
  },
  async xfeat(c) {
    const r = await xfeatMatch(c);
    c.features.xfeat = r;
    const out = featureVerdict(c, r);
    out.detail['mutual matches'] = out.detail['ratio-test matches'];
    delete out.detail['ratio-test matches'];
    return out;
  },

  async sscdAligned(c) {
    // SSCD on the shared region only, after undoing B's crop, rotation or
    // perspective with the keypoint transform: a copy pasted into a larger
    // picture, or rotated, is compared like for like.
    const Hw = workHomography(c);
    if (!Hw) return { verdict: 'na', display: 'n/a', note: 'No consistent keypoint transform was found, so there was nothing to undo.' };
    const A = c.a.work;
    const [w, h] = fitWithin(A.w, A.h, 640);
    const k = w / A.w;
    const cv = await loadOpenCV();
    const warped = F.warpRgbInto(cv, c.b.work, mat3(scale3(k), Hw), w, h);
    const Ad = resizeImg(A, w, h, 'bilinear');
    let x0 = w;
    let y0 = h;
    let x1 = -1;
    let y1 = -1;
    let covered = 0;
    for (let y = 0; y < h; y++) {
      for (let x = 0; x < w; x++) {
        if (!warped.mask[y * w + x]) continue;
        covered++;
        if (x < x0) x0 = x;
        if (x > x1) x1 = x;
        if (y < y0) y0 = y;
        if (y > y1) y1 = y;
      }
    }
    const coverage = covered / (w * h);
    if (coverage < 0.05) return { verdict: 'na', display: 'n/a', note: 'The aligned images barely overlap.' };
    // the same region of both, grey wherever B does not reach
    const bw = x1 - x0 + 1;
    const bh = y1 - y0 + 1;
    const ca = cropImg(Ad, x0, y0, bw, bh);
    const cb = cropImg(warped.img, x0, y0, bw, bh);
    for (let y = 0; y < bh; y++) {
      for (let x = 0; x < bw; x++) {
        if (warped.mask[(y + y0) * w + x + x0]) continue;
        const o = (y * bw + x) * 3;
        ca.rgb[o] = ca.rgb[o + 1] = ca.rgb[o + 2] = 128;
        cb.rgb[o] = cb.rgb[o + 1] = cb.rgb[o + 2] = 128;
      }
    }
    const [ort, s] = await Promise.all([loadOrt(), session('sscd')]);
    const value = N.cosine(await N.embed(ort, s, 'sscd', ca), await N.embed(ort, s, 'sscd', cb));
    const global = c.results?.sscd?.value;
    return {
      value,
      detail: {
        'compared region': `${Math.round(coverage * 100)}% of A (where B lands after alignment)`,
        'SSCD on the whole images': typeof global === 'number' ? global.toFixed(3) : '—',
      },
    };
  },
  sscdLarge: (c) => embedCompare(c, 'sscdLarge'),
  async dino(c) {
    const r = await embedCompare(c, 'dino');
    // Dense patch correspondences (Amir et al. 2021): where the images match.
    const [ort, s] = await Promise.all([loadOrt(), session('dino')]);
    const pa = await N.dinoPatches(ort, s, c.a.work);
    const pb = await N.dinoPatches(ort, s, c.b.work);
    const corr = N.patchCorrespondence(pa, pb);
    c.dinoCorr = { mutual: corr.mutual, ga: { gw: pa.gw, gh: pa.gh }, gb: { gw: pb.gw, gh: pb.gh } };
    const share = corr.mutual.filter(([, , sim]) => sim >= 0.5).length / (pa.gw * pa.gh);
    c.visuals.dino = {
      a: { gw: pa.gw, gh: pa.gh, values: corr.bestA },
      b: { gw: pb.gw, gh: pb.gh, values: corr.bestB },
      lines: spreadLines(corr.mutual, pa.gw),
      mutualShare: share,
      // unit-length patch features, for pointing at a spot and finding its match
      featsA: pa.feats,
      featsB: pb.feats,
      dims: pa.d,
    };
    r.detail = { 'patches of A with a mutual match in B (cos ≥ 0.5)': `${Math.round(share * 100)}%` };
    return r;
  },
  clip: (c) => embedCompare(c, 'clip'),
  async dreamsim(c) {
    const r = await embedCompare(c, 'dreamsim');
    return { value: 1 - r.value };
  },
  async lpips(c) {
    const [ort, s] = await Promise.all([loadOrt(), session('lpips')]);
    const [w, h] = atLeast(fitWithin(c.a.work.w, c.a.work.h, LPIPS_MAX), 64);
    const v = await N.lpipsDistance(ort, s, resizeImg(c.a.work, w, h), resizeImg(c.b.work, w, h));
    return { value: v, note: c.aspectNote() };
  },
};

/** EXIF times have no time zone; exifr reads them as local time, so print local fields. */
function fmtDate(d) {
  if (!(d instanceof Date) || Number.isNaN(d.getTime())) return String(d);
  const p = (n) => String(n).padStart(2, '0');
  return `${d.getFullYear()}-${p(d.getMonth() + 1)}-${p(d.getDate())} ${p(d.getHours())}:${p(d.getMinutes())}:${p(d.getSeconds())}`;
}

function cameraName(e) {
  const make = (e.Make || '').trim();
  const model = (e.Model || '').trim();
  return model.toLowerCase().startsWith(make.toLowerCase()) ? model : [make, model].filter(Boolean).join(' ');
}

/** Scale [w, h] up (keeping aspect) so the short side is at least `min`. */
function atLeast([w, h], min) {
  const s = Math.max(1, min / Math.min(w, h));
  return [Math.round(w * s), Math.round(h * s)];
}

function finalize(engine, r) {
  const out = { ...r };
  if (out.verdict === undefined) out.verdict = verdictFor(engine, r.value);
  if (out.display === undefined) out.display = Number.isNaN(r.value) ? 'n/a' : engine.format ? engine.format(r.value) : String(r.value);
  if (out.note === undefined) delete out.note;
  return out;
}

// ---------------------------------------------------------------- annotations

/**
 * Human-style annotations for the result view: numbered differences when B
 * can be aligned onto A, matching regions for the arrows, and SSCD's evidence
 * peaks. Boxes are returned in each image's original pixel coordinates.
 */
async function buildAnnotations(c) {
  const A = c.a.work;
  const B = c.b.work;
  const kA = c.a.w / A.w;
  const kB = c.b.w / B.w;
  const dets = c.objectDets || { a: [], b: [] };
  const out = { aligned: false, global: false, differences: [], regions: [], peaks: null };
  const toOrig = (box, k) => box.map((v) => v * k);
  const fracBox = (f, img) => [f[0] * img.w, f[1] * img.h, f[2] * img.w, f[3] * img.h];
  const pad = (box, img, p = 0.12) => {
    const pw = (box[2] - box[0]) * p + 4;
    const ph = (box[3] - box[1]) * p + 4;
    return [Math.max(0, box[0] - pw), Math.max(0, box[1] - ph), Math.min(img.w, box[2] + pw), Math.min(img.h, box[3] + ph)];
  };
  const thumbOf = (img, box) => rgbToRgba(O.thumb(O.cropBox(img, pad(box, img), 0), 180));

  // 1. Differences, only when B can be laid exactly over A.
  let Hw = workHomography(c);
  const R = c.results || {};
  const sameShape = Math.abs(Math.log(A.w / A.h / (B.w / B.h))) < 0.03;
  const pixelClose = (R.ssim?.value ?? 0) >= 0.5 || (R.pdq?.value ?? 256) <= 63;
  if (!Hw && sameShape && pixelClose) Hw = [A.w / B.w, 0, 0, 0, A.h / B.h, 0, 0, 0, 1];
  const keypointH = workHomography(c);
  out.mirrored = !!(keypointH && c.mirror && !Object.values(c.features).some((r) => r.sane && r.inliers >= 20));
  if (keypointH) {
    // which part of each image the other one shows (B's frame drawn on A, and A's on B)
    const apply = (M, x, y) => {
      const z = M[6] * x + M[7] * y + M[8];
      return [(M[0] * x + M[1] * y + M[2]) / z, (M[3] * x + M[4] * y + M[5]) / z];
    };
    const corners = (w, h) => [
      [0, 0],
      [w, 0],
      [w, h],
      [0, h],
    ];
    const inv = Ann.invert3(keypointH);
    const quadArea = (q) => Math.abs(q.reduce((t, [x, y], i) => t + x * q[(i + 1) % 4][1] - q[(i + 1) % 4][0] * y, 0)) / 2;
    const clipShare = (q, w, h) => {
      // share of a w×h image covered by quad q, estimated on a 40×40 grid
      let inside = 0;
      for (let i = 0; i < 40; i++) {
        for (let j = 0; j < 40; j++) {
          const x = ((j + 0.5) / 40) * w;
          const y = ((i + 0.5) / 40) * h;
          let sign = 0;
          let ok = true;
          for (let k = 0; k < 4 && ok; k++) {
            const [x1, y1] = q[k];
            const [x2, y2] = q[(k + 1) % 4];
            const cr = Math.sign((x2 - x1) * (y - y1) - (y2 - y1) * (x - x1));
            if (cr && sign && cr !== sign) ok = false;
            if (cr) sign = cr;
          }
          if (ok) inside++;
        }
      }
      return inside / 1600;
    };
    const qA = corners(B.w, B.h).map(([x, y]) => apply(keypointH, x, y));
    const qB = inv ? corners(A.w, A.h).map(([x, y]) => apply(inv, x, y)) : null;
    out.overlap = {
      inA: qA.map(([x, y]) => [x * kA, y * kA]),
      inB: qB ? qB.map(([x, y]) => [x * kB, y * kB]) : null,
      coverA: clipShare(qA, A.w, A.h),
      coverB: qB ? clipShare(qB, B.w, B.h) : 1,
      // B's pixels per pixel of A, in the original files
      zoom: 1 / ((Math.sqrt(quadArea(qA) / (B.w * B.h)) * kA) / kB),
      rotation: (Math.atan2(qA[1][1] - qA[0][1], qA[1][0] - qA[0][0]) * 180) / Math.PI,
    };
  }
  if (Hw) {
    const cv = await loadOpenCV();
    const [dw, dh] = fitWithin(A.w, A.h, 900);
    const kd = dw / A.w;
    const Ad = resizeImg(A, dw, dh, 'bilinear');
    const al = Ann.alignDense(cv, Ad, B, mat3(scale3(kd), Hw), dw, dh);
    const res = Ann.findDifferences(cv, Ad, al.img, al.mask);
    out.aligned = true;
    out.global = res.global;
    out.alignment = { ecc: al.ecc.used, maxShift: al.maxShift };
    res.regions.forEach((r, k) => {
      const boxAw = r.box.map((v) => v / kd);
      const boxBw = Ann.boxToB(al.toB, r.box);
      out.differences.push({
        n: k + 1,
        a: toOrig(boxAw, kA),
        b: toOrig(boxBw, kB),
        label: Ann.labelFor(boxAw, dets.a, A.w * A.h) || Ann.labelFor(boxBw, dets.b, B.w * B.h),
        kind: Ann.describeChange(Ad, al.img, r.box),
        thumbs: { a: thumbOf(Ad, r.box), b: thumbOf(al.img, r.box) },
      });
    });
  }

  // 2. Matching regions from the DINOv2 patch matches, named by detected objects.
  if (c.dinoCorr) {
    const { mutual, ga, gb } = c.dinoCorr;
    Ann.objectRegions(mutual, ga, gb, dets.a, dets.b, A, B).forEach((m, k) => {
      out.regions.push({
        n: k + 1,
        a: toOrig(m.a, kA),
        b: toOrig(m.b, kB),
        sim: m.sim,
        share: m.share,
        labelA: m.labelA,
        labelB: m.labelB,
        thumbs: { a: thumbOf(A, m.a), b: thumbOf(B, m.b) },
      });
    });
  }

  // 3. Which part of A pairs with which part of B in SSCD's copy score.
  if (c.visuals.sscd?.links) {
    out.copyLinks = c.visuals.sscd.links.links.map((k) => {
      const boxAw = fracBox(k.boxA, A);
      const boxBw = fracBox(k.boxB, B);
      return {
        weight: k.weight,
        labelA: Ann.labelFor(boxAw, dets.a, A.w * A.h),
        labelB: Ann.labelFor(boxBw, dets.b, B.w * B.h),
        thumbs: { a: thumbOf(A, boxAw), b: thumbOf(B, boxBw) },
      };
    });
  }

  // 4. Where SSCD's copy evidence peaks.
  if (c.visuals.sscd) {
    const peaks = (map, img, k, ds) =>
      Ann.evidencePeaks(map).map((p) => {
        const r = p.r * img.w;
        const box = [p.x * img.w - r, p.y * img.h - r, p.x * img.w + r, p.y * img.h + r];
        return { x: p.x * img.w * k, y: p.y * img.h * k, r: r * k, v: p.v, label: Ann.labelFor(box, ds, img.w * img.h) };
      });
    out.peaks = { score: c.visuals.sscd.score, a: peaks(c.visuals.sscd.a, A, kA, dets.a), b: peaks(c.visuals.sscd.b, B, kB, dets.b) };
  }
  return out;
}

// ---------------------------------------------------------------- scanning

let currentScan = 0;

function summarizeExif(e) {
  if (!e) return null;
  const pick = ['Make', 'Model', 'LensModel', 'Software', 'DateTimeOriginal', 'ImageUniqueID', 'Artist', 'Copyright', 'DocumentID', 'OriginalDocumentID'];
  const out = {};
  for (const k of pick) if (e[k] !== undefined && e[k] !== '') out[k] = k.startsWith('DateTime') ? fmtDate(e[k]) : String(e[k]);
  if (typeof e.latitude === 'number' && typeof e.longitude === 'number') out.GPS = `${e.latitude.toFixed(5)}, ${e.longitude.toFixed(5)}`;
  return Object.keys(out).length ? out : null;
}

async function scan({ scanId, a, b, disabled }) {
  currentScan = scanId;
  const t0 = performance.now();
  post({ type: 'status', scanId, text: 'Decoding images…' });
  const A = await prepare(a);
  const B = await prepare(b);
  const c = new Ctx(A, B, disabled);
  for (const [side, img] of [['a', A], ['b', B]]) {
    Object.assign(c.info[side], {
      dimensions: `${img.w} × ${img.h}`,
      megapixels: ((img.w * img.h) / 1e6).toFixed(2),
      transparency: img.hasAlpha ? 'yes (composited on white)' : 'no',
      exif: summarizeExif(img.exif),
    });
  }
  post({ type: 'info', scanId, info: c.info });

  c.results = {};
  for (const id of ORDER) {
    if (scanId !== currentScan) return;
    // once every detector has run: retry against a mirrored B if nothing lined up
    if (id === 'alignedSsim') await mirrorCheck(c).catch((err) => console.warn('mirror check failed', err));
    const engine = ENGINE_BY_ID[id];
    if (engine.model && disabled.includes(engine.model)) {
      post({ type: 'result', scanId, id, result: { verdict: 'skipped', display: 'turned off' } });
      continue;
    }
    post({ type: 'running', scanId, id });
    const t = performance.now();
    let result;
    try {
      result = finalize(engine, await RUN[id](c));
    } catch (err) {
      console.error(id, err);
      result = { verdict: 'error', display: 'error', note: String((err && err.message) || err) };
    }
    result.ms = Math.round(performance.now() - t);
    c.results[id] = result;
    if (scanId !== currentScan) return;
    post({ type: 'result', scanId, id, result });
    await tick();
  }

  try {
    c.visuals.annotations = await buildAnnotations(c);
  } catch (err) {
    console.warn('annotations failed', err);
  }
  for (const side of ['a', 'b']) {
    c.info[side].md5 = md5(c[side].bytes);
    c.info[side].sha256 = await sha('SHA-256', c[side].bytes);
  }
  const v = c.visuals;
  v.pairA = rgbToRgba(c.pair.A);
  v.pairB = rgbToRgba(c.pair.B);
  const bestKind = Object.entries(c.features).sort((x, y) => y[1].inliers - x[1].inliers)[0];
  if (bestKind && bestKind[1].matches.length) {
    const [kind, r] = bestKind;
    const step = Math.max(1, Math.floor(r.matches.length / 150));
    v.matches = {
      kind: kind.toUpperCase(),
      a: { w: c.featureImg('a').w, h: c.featureImg('a').h },
      b: { w: c.featureImg('b').w, h: c.featureImg('b').h },
      lines: r.matches.filter((_, i) => i % step === 0),
      inliers: r.inliers,
      sane: r.sane,
    };
  }
  // hand typed arrays over without copying (each buffer once; views into a
  // larger buffer are copied)
  const transfer = [];
  const seen = new Set();
  const collect = (o) => {
    if (!o || typeof o !== 'object') return;
    if (ArrayBuffer.isView(o)) {
      if (o.byteOffset === 0 && o.byteLength === o.buffer.byteLength && !seen.has(o.buffer)) {
        seen.add(o.buffer);
        transfer.push(o.buffer);
      }
      return;
    }
    for (const x of Object.values(o)) collect(x);
  };
  collect(v);
  post({ type: 'info', scanId, info: c.info });
  post({ type: 'visuals', scanId, visuals: v }, transfer);
  post({ type: 'done', scanId, ms: Math.round(performance.now() - t0) });
}

self.onmessage = (e) => {
  const msg = e.data;
  if (msg.type === 'scan') {
    scan(msg).catch((err) => {
      console.error(err);
      post({ type: 'fatal', scanId: msg.scanId, message: String((err && err.message) || err) });
    });
  } else if (msg.type === 'model') {
    waiters.get(msg.key)?.resolve(msg.bytes);
    waiters.delete(msg.key);
  } else if (msg.type === 'model-error') {
    waiters.get(msg.key)?.reject(new Error(msg.message));
    waiters.delete(msg.key);
  } else if (msg.type === 'warm') {
    loadOrt().catch(() => {});
    loadOpenCV().catch(() => {});
  } else if (msg.type === 'cancel') {
    currentScan = -1;
  } else if (msg.type === 'rescore') {
    // the cover-up test: SSCD on images with parts painted over
    (async () => {
      const [ort, s] = await Promise.all([loadOrt(), session('sscd')]);
      const img = (x) => rgbaToRgb(x.data, x.w, x.h);
      const value = N.cosine(await N.embed(ort, s, 'sscd', img(msg.a)), await N.embed(ort, s, 'sscd', img(msg.b)));
      post({ type: 'rescored', id: msg.id, value });
    })().catch((err) => post({ type: 'rescored', id: msg.id, error: String((err && err.message) || err) }));
  }
};
