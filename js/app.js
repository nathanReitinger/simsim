import { ENGINES, ENGINE_BY_ID, GROUPS, VERDICT_LABEL } from './engines.js';
import { MODELS } from './lib/neural.js';
import { ModelStore } from './lib/modelstore.js';
import { applyTransform, DEFAULTS, PRESETS, describe } from './transform.js';
import { CASES, CASE_GROUPS } from './cases.js';
import { lutColor } from './lib/colormap.js';
import * as AV from './annotate-view.js';
import * as C2PA from './c2pa.js';

const ROOT = new URL('../', import.meta.url);
const $ = (sel, el = document) => el.querySelector(sel);
const $$ = (sel, el = document) => [...el.querySelectorAll(sel)];

/** Tiny element builder: el('div', { class: 'x', onclick }, child, 'text') */
function el(tag, attrs = {}, ...children) {
  const node = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs)) {
    if (v === undefined || v === null || v === false) continue;
    if (k.startsWith('on')) node.addEventListener(k.slice(2), v);
    else if (k === 'class') node.className = v;
    else if (k === 'html') node.innerHTML = v;
    else if (k === 'dataset') Object.assign(node.dataset, v);
    else node.setAttribute(k, v === true ? '' : v);
  }
  for (const c of children.flat()) {
    if (c === undefined || c === null || c === false) continue;
    node.append(c instanceof Node ? c : document.createTextNode(String(c)));
  }
  return node;
}

const COUNTED = new Set(['identical', 'match', 'partial', 'none']);
const FLAGGED = new Set(['identical', 'match']);
const MODEL_ORDER = ['sscd', 'xfeat', 'lpips', 'dino', 'dfine', 'sscdLarge', 'clip', 'dreamsim', 'pose'];
const MODEL_INFO = {
  sscd: { name: 'SSCD (ResNet-50)', note: 'The main copy detector. Recommended.' },
  sscdLarge: { name: 'SSCD large (ResNeXt-101)', note: 'Replication-study setting from Somepalli et al.' },
  dino: { name: 'DINOv2 small', note: 'General visual similarity.' },
  clip: { name: 'CLIP ViT-B/32', note: 'Semantic similarity; the largest download.' },
  lpips: { name: 'LPIPS (AlexNet)', note: 'Perceptual distance; tiny.' },
  xfeat: { name: 'XFeat keypoints', note: 'Learned keypoints for aligning copies; tiny.' },
  dreamsim: { name: 'DreamSim', note: 'Similarity as people judge it; a large download.' },
  pose: { name: 'ViTPose (body pose)', note: 'Compares the poses of the people in both images; a large download.' },
  dfine: { name: 'D-FINE object detector', note: 'Finds hats, pictures, tables… for the Objects tab.' },
};
const ICONS = {
  identical: '<svg viewBox="0 0 24 24"><path d="M6 9.5h12M6 14.5h12"/></svg>',
  match: '<svg viewBox="0 0 24 24"><path d="M12 6v8m0 4h.01"/></svg>',
  partial: '<svg viewBox="0 0 24 24"><path d="M6 12h12"/></svg>',
  none: '<svg viewBox="0 0 24 24"><path d="M5 12.5l4.5 4.5L19 7.5"/></svg>',
  na: '<svg viewBox="0 0 24 24"><path d="M7 12h10"/></svg>',
  error: '<svg viewBox="0 0 24 24"><path d="M8 8l8 8m0-8-8 8"/></svg>',
  info: '<svg viewBox="0 0 24 24"><path d="M12 11v6m0-10h.01"/></svg>',
};

// ------------------------------------------------------------------ state

const SETTINGS_KEY = 'image-similarity-scanner/settings';
function loadSettings() {
  try {
    return { disabled: [], ...JSON.parse(localStorage.getItem(SETTINGS_KEY) || '{}') };
  } catch {
    return { disabled: [] };
  }
}
function saveSettings() {
  try {
    localStorage.setItem(SETTINGS_KEY, JSON.stringify(state.settings));
  } catch {
    // storage unavailable: settings last for this visit only
  }
}

const state = {
  slots: { a: null, b: null },
  scanId: 0,
  scanning: false,
  scanT0: 0, // when the current scan started (performance.now)
  runningAt: 0, // when the test running now started
  lastResultAt: 0, // when the latest test finished
  results: {},
  info: null,
  visuals: null,
  elapsed: 0,
  settings: loadSettings(),
  modelsInWorker: new Set(),
  modelStatus: {},
  downloads: {},
  view: 'swipe',
  viewChosen: false,
  whereLens: null,
  whereChosen: false,
  probeSource: 'dino',
  evidenceMode: 'arrows',
  openGroups: new Set(['neural', 'objects']),
  objectView: 'markup',
  objectHighlight: null,
  case: null, // copyright case whose works are loaded in A and B
  caseImages: {}, // case id -> image paths, from assets/cases/manifest.json
  scanWaiters: [], // callbacks for the end of the current scan
  batch: null, // the all-cases run in progress
  lastBatch: null, // rows of the last all-cases run
  openEngines: new Set(),
  warmed: false,
};

const isDisabled = (key) => state.settings.disabled.includes(key);

// ------------------------------------------------------------------ working indicators

// In the spirit of Claude Code's spinner: a star that morphs through
// · ✢ ✳ ✶ ✻ ✽ and back, a shimmering phrase that changes every couple of
// seconds, the time so far and, for a scan, how much of it is done. One
// timer drives every indicator on the page and stops when none are left.

const GLYPHS = ['·', '✢', '✳︎', '✶', '✻', '✽', '✻', '✶', '✳︎', '✢'];

// Asides that keep the site's point in view, mixed into every list of phrases.
const ASIDES = [
  'Measuring pixels, not originality',
  'Not deciding fair use',
  'Comparing bytes, not expression',
  'Leaving substantial similarity to the courts',
  'Ideas are free; checking the pixels',
  'Similar is not the same as infringing',
  'Cross-examining pixels',
  'Asking 101 tests, not a judge',
];
/** A list of phrases with one of the asides after every three. */
const withAsides = (list, offset) => list.flatMap((p, i) => (i % 3 === 2 ? [p, ASIDES[(offset + Math.floor(i / 3)) % ASIDES.length]] : [p]));
const PHRASES = {
  visuals: withAsides(
    [
      'Lining up A and B',
      'Matching ORB keypoints',
      'Matching AKAZE keypoints',
      'Fitting a homography',
      'Warping B onto A',
      'Undoing the crop',
      'Checking for a mirror image',
      'Hunting for shared patches',
      'Pairing up DINOv2 patches',
      'Painting the heat map',
      'Measuring colour shifts (ΔE)',
      'Mapping SSIM patch by patch',
      'Circling what changed',
      'Numbering the differences',
      'Drawing the arrows',
      'Tracing the edges',
      'Outlining the region they share',
      'Squinting at the details',
    ],
    0,
  ),
  objects: withAsides(
    [
      'Looking for objects',
      'Drawing boxes',
      'Naming what is in each picture',
      'Pairing objects up',
      'Comparing them one by one',
      'Matching hats with hats',
      'Counting people, pictures and chairs',
      'Finding what only A has',
      'Finding what only B has',
      'Checking poses joint by joint',
    ],
    3,
  ),
  spectrum: withAsides(
    [
      'Tallying the tests',
      'Weighing the scores',
      'Same file? Same pixels?',
      'Re-saved, or edited?',
      'Shared part, or similar subject?',
      'Asking the copy detectors',
      'Finding the right rung',
    ],
    5,
  ),
  generic: ASIDES,
};

// Typical time per test (ms, median over five copyright cases on a laptop),
// so the progress bar moves with time rather than with the count of tests:
// the object detector and the neural models take most of a scan.
const EXPECTED_MS = {
  objects: 5700,
  dino: 3600,
  pose: 1900,
  sscd: 1100,
  sscdLarge: 1000,
  sscdAligned: 1000,
  dreamsim: 750,
  clip: 720,
  gist: 630,
  xfeat: 620,
  kaze: 580,
  fsim: 490,
  gabor: 300,
  brisk: 220,
  akaze: 200,
  whashDb4: 190,
  vsi: 170,
  orb: 150,
  ssim: 130,
  lpips: 130,
  cropResistant: 120,
  vif: 100,
};
const OTHER_MS = 20; // each of the other tests (median 10 ms, mean 21 ms)
const TAIL_MS = 1100; // reading the files, and marking up the views after the last test

/** How much of the current scan is done, 0–1, from the typical time of each test. */
function scanProgress() {
  if (!state.scanning) return 1;
  const now = performance.now();
  let total = TAIL_MS;
  let done = 0;
  let waiting = false;
  for (const e of ENGINES) {
    if (e.model && isDisabled(e.model)) continue;
    const w = EXPECTED_MS[e.id] ?? OTHER_MS;
    total += w;
    const r = state.results[e.id];
    if (r && r.verdict !== 'running') {
      done += w;
    } else {
      waiting = true;
      // the test running now earns credit for its time so far, up to 90 %
      if (r) done += w * Math.min(0.9, (now - state.runningAt) / w);
    }
  }
  if (!waiting) done += TAIL_MS * Math.min(0.9, (now - state.lastResultAt) / TAIL_MS);
  return Math.min(0.99, done / total);
}

const stillMotion = matchMedia('(prefers-reduced-motion: reduce)');
let loaderTimer = null;

/** A slim progress bar with its percentage; fill it with setBar. */
function progressBar(label) {
  return el(
    'div',
    { class: 'loader-bar' },
    el('div', { class: 'loader-track', role: 'progressbar', 'aria-label': label, 'aria-valuemin': '0', 'aria-valuemax': '100' }, el('i')),
    el('span', { class: 'loader-pct', 'aria-hidden': 'true' }),
  );
}

function setBar(bar, fraction) {
  const percent = Math.floor(fraction * 100);
  const track = $('.loader-track', bar);
  track.firstElementChild.style.width = `${percent}%`;
  track.setAttribute('aria-valuenow', String(percent));
  $('.loader-pct', bar).textContent = `${percent}%`;
}

/**
 * A "working…" indicator: the morphing star and a phrase. `kind` picks the
 * phrases it cycles through, or pass `text` for a fixed label (relabel it
 * with setLoaderText). `pct` adds the percentage of the scan done, `bar` a
 * progress bar with it, `panel` centres it with a `note` underneath, and `t0`
 * is when the work began, so a re-rendered indicator keeps counting. `quiet`
 * leaves it out of the accessibility tree where the row already says what is
 * running.
 */
function loader(kind, { text = null, pct = false, bar = false, panel = false, note = null, time = true, t0 = performance.now(), label = 'Working', quiet = false } = {}) {
  const node = el(
    'div',
    {
      class: panel ? 'loader loader-panel' : 'loader',
      role: quiet ? null : 'status',
      'aria-label': quiet ? null : label,
      'aria-hidden': quiet ? 'true' : null,
      dataset: { kind, seed: Math.floor(Math.random() * 997), t0, fixed: text === null ? '' : '1' },
    },
    el(
      'div',
      { class: 'loader-line', 'aria-hidden': 'true' },
      el('span', { class: 'loader-glyph' }),
      el('span', { class: 'loader-verb' }, text ?? ''),
      time ? el('span', { class: 'loader-time' }) : null,
      pct ? el('span', { class: 'loader-pct' }) : null,
    ),
    bar ? progressBar('Scan progress') : null,
    note ? el('p', { class: 'loader-note' }, note) : null,
  );
  tickLoader(node, performance.now(), scanProgress());
  loaderTimer ??= setInterval(tickLoaders, 120);
  return node;
}

function tickLoader(n, now, progress) {
  const t = Math.max(0, now - Number(n.dataset.t0));
  $('.loader-glyph', n).textContent = stillMotion.matches ? '✻' : GLYPHS[Math.floor(t / 120) % GLYPHS.length];
  if (!n.dataset.fixed) {
    const words = PHRASES[n.dataset.kind] || PHRASES.generic;
    const k = Math.floor(t / 2200);
    if (n.dataset.k !== String(k)) {
      n.dataset.k = k;
      const verb = $('.loader-verb', n);
      const phrase = words[(Number(n.dataset.seed) + k) % words.length];
      verb.textContent = phrase.endsWith('?') ? phrase : `${phrase}…`;
      verb.classList.remove('loader-swap');
      void verb.offsetWidth; // restart the fade-in
      verb.classList.add('loader-swap');
    }
  }
  const time = $('.loader-time', n);
  if (time) time.textContent = t >= 1000 ? `${Math.floor(t / 1000)}s` : '';
  const pct = $('.loader-line .loader-pct', n);
  if (pct) pct.textContent = `${Math.floor(progress * 100)}%`;
  const bar = $('.loader-bar', n);
  if (bar) setBar(bar, progress);
}

function tickLoaders() {
  const nodes = $$('.loader');
  if (!nodes.length) {
    clearInterval(loaderTimer);
    loaderTimer = null;
    return;
  }
  const now = performance.now();
  const progress = scanProgress();
  for (const n of nodes) tickLoader(n, now, progress);
  if (state.scanning) $('#progressBar').style.width = `${Math.floor(progress * 100)}%`;
  const b = state.batch;
  if (b) setBar($('#runAllBar'), (b.rows.length + (state.scanning ? progress : 0)) / b.queue.length);
}

/** Relabel fixed-text indicators, e.g. with the test that is running now. */
function setLoaderText(selector, text) {
  for (const n of $$(selector)) {
    const verb = $('.loader-verb', n);
    if (verb.textContent !== text) verb.textContent = text;
  }
}

/** The scan's own status line, in the summary card. */
function endScanStatus() {
  const box = $('#scanStatus');
  box.hidden = true;
  box.replaceChildren();
}

// ------------------------------------------------------------------ worker & models

let worker = null;
function getWorker() {
  if (!worker) {
    worker = new Worker(new URL('./worker.js', import.meta.url), { type: 'module' });
    worker.onmessage = onWorkerMessage;
    worker.onerror = (e) => {
      console.error(e);
      if (state.scanning) fail(`The analysis worker crashed${e.message ? `: ${e.message}` : ''}. Try smaller images or turn off large models in Settings.`);
      worker = null;
      state.modelsInWorker.clear();
    };
  }
  return worker;
}

const store = new ModelStore(MODELS, ROOT, (key, loaded, total, phase) => {
  state.downloads[key] = { loaded, total, phase };
  renderDownloadStatus();
  for (const e of ENGINES) if (e.model === key) renderEngine(e.id);
  if (phase === 'done') refreshDownloadNote();
});

function warm() {
  if (state.warmed) return;
  state.warmed = true;
  getWorker().postMessage({ type: 'warm' });
  prefetchModels();
}

async function prefetchModels() {
  for (const key of MODEL_ORDER) {
    if (isDisabled(key) || state.modelsInWorker.has(key)) continue;
    try {
      await store.prefetch(key);
    } catch (err) {
      console.warn('prefetch failed', key, err);
    }
  }
}

const rescoreWaiters = new Map();
let rescoreSeq = 0;

/** SSCD score of two (painted-over) images, computed in the worker. */
function rescore(a, b) {
  const id = ++rescoreSeq;
  return new Promise((resolve, reject) => {
    rescoreWaiters.set(id, { resolve, reject });
    getWorker().postMessage({ type: 'rescore', id, a, b }, [a.data.buffer, b.data.buffer]);
  });
}

function onWorkerMessage(e) {
  const m = e.data;
  if (m.type === 'rescored') {
    const w = rescoreWaiters.get(m.id);
    rescoreWaiters.delete(m.id);
    if (w) (m.error ? w.reject(new Error(m.error)) : w.resolve(m.value));
    return;
  }
  if (m.type === 'need-model') {
    store
      .get(m.key)
      .then((bytes) => worker.postMessage({ type: 'model', key: m.key, bytes }, [bytes.buffer]))
      .catch((err) => worker.postMessage({ type: 'model-error', key: m.key, message: `download failed (${err.message})` }));
    return;
  }
  if (m.type === 'model-status') {
    state.modelStatus[m.key] = m.status;
    if (m.status === 'ready') state.modelsInWorker.add(m.key);
    for (const eng of ENGINES) if (eng.model === m.key) renderEngine(eng.id);
    return;
  }
  if (m.scanId !== state.scanId) return; // stale message from an earlier scan
  switch (m.type) {
    case 'status':
      $('#progressText').textContent = m.text;
      break;
    case 'info':
      state.info = m.info;
      renderDetails();
      checkCredentials(m.scanId, m.info);
      break;
    case 'running':
      state.runningAt = performance.now();
      state.results[m.id] = { verdict: 'running' };
      renderEngine(m.id);
      updateSpotlight(m.id);
      setLoaderText('.loader[data-kind="scan"]', `${ENGINE_BY_ID[m.id]?.name || m.id}…`);
      break;
    case 'result':
      state.lastResultAt = performance.now();
      state.results[m.id] = m.result;
      renderEngine(m.id);
      updateSpotlight(m.id);
      updateSummary();
      break;
    case 'visuals':
      state.visuals = m.visuals;
      renderSpectrum();
      renderWhereCard();
      renderViewModes();
      renderVisual();
      renderObjects();
      break;
    case 'done':
      state.scanning = false;
      state.elapsed = m.ms;
      endScanStatus();
      updateSummary();
      updateButtons();
      if (!state.visuals) {
        // nothing to show: replace the "working" panels with their final messages
        renderWhereCard();
        renderVisual();
        renderObjects();
      }
      settleScanWaiters();
      break;
    case 'fatal':
      fail(m.message);
      break;
    default:
  }
}

/** Content Credentials, read on the page with the C2PA SDK when either file carries them. */
async function checkCredentials(scanId, info) {
  const show = (r) => {
    if (scanId !== state.scanId) return;
    state.results.c2pa = r;
    renderEngine('c2pa');
    updateSummary();
  };
  if (!info?.a?.c2pa && !info?.b?.c2pa) {
    show({ verdict: 'na', display: 'none in either file', note: 'Neither file carries Content Credentials. Most images do not, and screenshots and most website uploads remove them.' });
    return;
  }
  show({ verdict: 'running' });
  try {
    const read = async (side) => (info[side]?.c2pa ? C2PA.summarize(await C2PA.readCredentials(state.slots[side].file)) : null);
    const [sa, sb] = await Promise.all([read('a'), read('b')]);
    const detail = { A: C2PA.describe(sa), B: C2PA.describe(sb) };
    if (C2PA.linked(sb, sa)) {
      show({ verdict: 'match', display: 'B lists A as an ingredient', detail, note: 'B’s signed Content Credentials record A among the files it was made from — its own account of being derived from A.' });
    } else if (C2PA.linked(sa, sb)) {
      show({ verdict: 'match', display: 'A lists B as an ingredient', detail, note: 'A’s signed Content Credentials record B among the files it was made from.' });
    } else {
      const ai = [sa?.ai && 'A', sb?.ai && 'B'].filter(Boolean);
      const has = [sa && 'A', sb && 'B'].filter(Boolean);
      show({
        verdict: 'info',
        display: ai.length ? `${ai.join(' and ')}: generative AI declared` : `credentials in ${has.join(' and ')}`,
        detail,
        note: `${has.join(' and ')} ${has.length === 2 ? 'carry' : 'carries'} signed Content Credentials, but neither lists the other as an ingredient.${ai.length ? ` ${ai.join(' and ')} ${ai.length === 2 ? 'declare' : 'declares'} that generative AI produced ${ai.length === 2 ? 'them' : 'it'}.` : ''}`,
      });
    }
  } catch (err) {
    show({ verdict: 'error', display: 'could not read', note: `The C2PA reader failed: ${err.message || err}` });
  }
}

function fail(message) {
  state.scanning = false;
  endScanStatus();
  // the scan will not finish what the indicators are waiting for
  for (const n of $$('#results .loader')) {
    n.replaceWith(n.classList.contains('loader-panel') ? el('p', { class: 'muted small' }, 'The scan stopped before this part finished.') : '');
  }
  $('#spectrum').hidden = true;
  settleScanWaiters();
  const s = $('.summary');
  s.dataset.tone = 'error';
  $('#summaryKicker').textContent = 'Scan failed';
  $('#summaryTitle').textContent = 'Something went wrong';
  $('#summarySub').textContent = message;
  updateButtons();
}

// ------------------------------------------------------------------ image slots

const IMAGE_EXT = /\.(png|jpe?g|jfif|gif|webp|avif|bmp|svg|ico|heic|heif|tiff?)$/i;

async function decodeImage(file) {
  try {
    return await createImageBitmap(file, { imageOrientation: 'from-image' });
  } catch {
    // Some formats (e.g. SVG) only decode through an <img> element.
    const url = URL.createObjectURL(file);
    try {
      const img = new Image();
      img.src = url;
      await img.decode();
      const w = img.naturalWidth || 1024;
      const h = img.naturalHeight || 1024;
      const c = el('canvas', { width: w, height: h });
      c.getContext('2d').drawImage(img, 0, 0, w, h);
      return await createImageBitmap(c);
    } finally {
      URL.revokeObjectURL(url);
    }
  }
}

async function setSlot(slot, file, { keepCase = false } = {}) {
  if (!file) return;
  if (!(file.type || '').startsWith('image/') && !IMAGE_EXT.test(file.name || '')) {
    toast(`“${file.name}” does not look like an image.`);
    return;
  }
  let bitmap;
  try {
    bitmap = await decodeImage(file);
  } catch {
    toast(`Could not decode “${file.name}”. Your browser may not support this format — try PNG or JPEG.`);
    return;
  }
  const prev = state.slots[slot];
  if (prev) {
    URL.revokeObjectURL(prev.url);
    prev.bitmap.close?.();
  }
  state.slots[slot] = { file, bitmap, url: URL.createObjectURL(file), w: bitmap.width, h: bitmap.height };
  if (!keepCase) state.case = null;
  renderSlot(slot);
  updateButtons();
  warm();
}

function clearSlot(slot) {
  state.case = null;
  const prev = state.slots[slot];
  if (prev) {
    URL.revokeObjectURL(prev.url);
    prev.bitmap.close?.();
  }
  state.slots[slot] = null;
  renderSlot(slot);
  updateButtons();
}

function renderSlot(slot) {
  const drop = $(`.drop[data-slot="${slot}"]`);
  const s = state.slots[slot];
  $('.drop-empty', drop).hidden = !!s;
  $('.drop-filled', drop).hidden = !s;
  drop.classList.toggle('filled', !!s);
  if (s) {
    $('img', drop).src = s.url;
    $('.drop-name', drop).textContent = s.file.name || 'pasted image';
    $('.drop-dims', drop).textContent = `${s.w}×${s.h} · ${formatBytes(s.file.size)}`;
  }
}

function updateButtons() {
  const both = state.slots.a && state.slots.b;
  $('#scanBtn').disabled = !both;
  $('#labBtn').disabled = !state.slots.a;
  $('#labLink').disabled = !state.slots.a;
  const done = !state.scanning && Object.keys(state.results).length > 0;
  $('#exportBtn').disabled = !done;
  $('#copyBtn').disabled = !done;
}

function setupDrops() {
  for (const drop of $$('.drop')) {
    const slot = drop.dataset.slot;
    const input = $('input', drop);
    drop.addEventListener('click', (e) => {
      if (e.target.closest('.drop-clear, .linklike')) return;
      input.click();
    });
    drop.addEventListener('keydown', (e) => {
      if (e.target !== drop) return;
      if (e.key === 'Enter' || e.key === ' ') {
        e.preventDefault();
        input.click();
      }
    });
    input.addEventListener('change', () => {
      if (input.files[0]) setSlot(slot, input.files[0]);
      input.value = '';
    });
    $('.drop-clear', drop).addEventListener('click', () => clearSlot(slot));
    drop.addEventListener('dragenter', (e) => {
      e.preventDefault();
      drop.classList.add('dragover');
    });
    drop.addEventListener('dragover', (e) => {
      e.preventDefault();
      drop.classList.add('dragover');
    });
    drop.addEventListener('dragleave', (e) => {
      if (!drop.contains(e.relatedTarget)) drop.classList.remove('dragover');
    });
    drop.addEventListener('drop', (e) => {
      e.preventDefault();
      e.stopPropagation();
      drop.classList.remove('dragover');
      const files = [...(e.dataTransfer?.files || [])];
      if (!files.length) {
        toast('That item is not a file. Save the image to your computer first, then drop the file here.');
        return;
      }
      setSlot(slot, files[0]);
      if (files[1]) setSlot(slot === 'a' ? 'b' : 'a', files[1]);
    });
  }
  // Drops elsewhere on the page fill the empty slots.
  document.addEventListener('dragover', (e) => e.preventDefault());
  document.addEventListener('drop', (e) => {
    e.preventDefault();
    const files = [...(e.dataTransfer?.files || [])];
    fillSlots(files);
  });
  document.addEventListener('paste', (e) => {
    const files = [...(e.clipboardData?.files || [])].filter((f) => f.type.startsWith('image/'));
    if (!files.length) return;
    const focused = document.activeElement?.closest?.('.drop');
    if (focused && files.length === 1) setSlot(focused.dataset.slot, files[0]);
    else fillSlots(files);
  });
  $('#swapBtn').addEventListener('click', () => {
    state.case = null; // A is no longer the plaintiff's work
    [state.slots.a, state.slots.b] = [state.slots.b, state.slots.a];
    renderSlot('a');
    renderSlot('b');
    updateButtons();
    if (!$('#results').hidden && state.slots.a && state.slots.b) startScan();
  });
}

function fillSlots(files) {
  if (files.length >= 2) {
    setSlot('a', files[0]);
    setSlot('b', files[1]);
  } else if (files.length === 1) {
    setSlot(state.slots.a ? 'b' : 'a', files[0]);
  }
}

// ------------------------------------------------------------------ scanning

async function startScan() {
  const { a, b } = state.slots;
  if (!a || !b) return;
  warm();
  const scanId = ++state.scanId;
  state.results = {};
  state.info = null;
  state.visuals = null;
  state.scanning = true;
  state.scanT0 = performance.now();
  state.runningAt = state.scanT0;
  state.lastResultAt = state.scanT0;
  state.openEngines.clear();
  $('#results').hidden = false;
  const status = $('#scanStatus');
  status.replaceChildren(loader('scan', { text: 'Reading the files…', pct: true, t0: state.scanT0, label: 'Scanning the images' }));
  status.hidden = false;
  delete $('#spectrum').dataset.pending;
  const s = $('.summary');
  delete s.dataset.tone;
  renderCaseBanner();
  state.objectHighlight = null;
  state.whereChosen = false;
  renderObjects();
  renderWhereCard();
  renderGroups();
  renderSpotlight();
  renderViewModes();
  renderVisual();
  renderDetails();
  updateSummary();
  updateButtons();
  if (!state.batch) requestAnimationFrame(() => $('#results').scrollIntoView({ behavior: 'smooth', block: 'start' }));
  prefetchModels();

  const [bmA, bmB, bytesA, bytesB] = await Promise.all([
    createImageBitmap(a.bitmap),
    createImageBitmap(b.bitmap),
    a.file.arrayBuffer(),
    b.file.arrayBuffer(),
  ]);
  if (scanId !== state.scanId) return;
  getWorker().postMessage(
    {
      type: 'scan',
      scanId,
      a: { name: a.file.name, type: a.file.type, bytes: bytesA, bitmap: bmA },
      b: { name: b.file.name, type: b.file.type, bytes: bytesB, bitmap: bmB },
      disabled: state.settings.disabled,
    },
    [bytesA, bytesB, bmA, bmB],
  );
}

// ------------------------------------------------------------------ detection list

const engineEls = {};

function renderGroups() {
  const root = $('#groups');
  root.innerHTML = '';
  for (const g of GROUPS) {
    const engines = ENGINES.filter((e) => e.group === g.id);
    const list = el('div', { class: 'engines' });
    for (const engine of engines) {
      const node = engineElement(engine);
      engineEls[engine.id] = node;
      list.append(node);
    }
    const details = el(
      'details',
      { class: 'group card', 'data-group': g.id, open: state.openGroups.has(g.id) },
      el(
        'summary',
        { class: 'group-head' },
        el('span', { class: 'group-chevron', 'aria-hidden': 'true' }),
        el('h3', {}, g.title),
        el('span', { class: 'group-dots', 'aria-hidden': 'true' }, engines.map((e) => el('i', { class: 'gdot', 'data-id': e.id, title: e.name }))),
        el('span', { class: 'group-count' }),
      ),
      el('p', { class: 'group-blurb' }, g.blurb),
      list,
    );
    details.addEventListener('toggle', () => {
      if (details.open) state.openGroups.add(g.id);
      else state.openGroups.delete(g.id);
    });
    root.append(details);
    for (const engine of engines) renderEngine(engine.id);
  }
  updateGroupCounts();
}

function setAllGroups(open) {
  for (const d of $$('#groups details.group')) d.open = open;
}

function engineElement(engine) {
  const node = el(
    'div',
    { class: 'engine', 'data-verdict': 'pending', 'data-id': engine.id },
    el(
      'button',
      { type: 'button', class: 'engine-main', 'aria-expanded': 'false', onclick: () => toggleEngine(engine.id) },
      el('span', { class: 'engine-icon', 'aria-hidden': 'true' }),
      el(
        'span',
        { class: 'engine-titles' },
        el('span', { class: 'engine-name' }, engine.name, engine.featured ? el('span', { class: 'star', title: 'Featured test' }, '★') : null),
        el('span', { class: 'engine-by' }, engine.by),
      ),
      el('span', { class: 'engine-result' }, el('span', { class: 'engine-value' }), el('span', { class: 'engine-verdict' })),
    ),
    el('div', { class: 'engine-more', hidden: true }),
  );
  return node;
}

function pendingText(engine) {
  if (engine.model) {
    if (isDisabled(engine.model)) return 'turned off';
    const d = state.downloads[engine.model];
    if (d && d.phase !== 'done' && !state.modelsInWorker.has(engine.model)) {
      return `downloading ${Math.floor((100 * d.loaded) / d.total)}%`;
    }
    if (state.modelStatus[engine.model] === 'initializing') return 'loading model…';
  }
  return '';
}

function renderEngine(id) {
  const node = engineEls[id];
  if (!node) return;
  const engine = ENGINE_BY_ID[id];
  const r = state.results[id];
  const verdict = r ? r.verdict : 'pending';
  node.dataset.verdict = verdict === 'skipped' || verdict === 'error' ? 'na' : verdict;
  const icon = $('.engine-icon', node);
  const iconKey = verdict === 'skipped' ? 'na' : verdict;
  if (verdict === 'running') {
    if (!$('.loader', icon)) icon.replaceChildren(loader('engine', { text: '', time: false, t0: state.scanT0, quiet: true }));
  } else {
    icon.innerHTML = ICONS[iconKey] || '';
  }
  const value = $('.engine-value', node);
  const label = $('.engine-verdict', node);
  if (!r || verdict === 'pending' || verdict === 'running') {
    value.textContent = pendingText(engine) || (verdict === 'running' ? 'running…' : '');
    label.textContent = verdict === 'running' ? 'Running' : state.scanning ? 'Waiting' : '';
  } else {
    value.textContent = r.display ?? '';
    label.textContent = VERDICT_LABEL[verdict] || verdict;
  }
  if (state.openEngines.has(id)) renderMore(id);
  updateGroupCounts();
}

function toggleEngine(id) {
  const node = engineEls[id];
  const open = !state.openEngines.has(id);
  if (open) state.openEngines.add(id);
  else state.openEngines.delete(id);
  node.classList.toggle('open', open);
  $('.engine-main', node).setAttribute('aria-expanded', String(open));
  $('.engine-more', node).hidden = !open;
  if (open) renderMore(id);
}

function tagList(items, weak = false) {
  return el('div', { class: 'tags' }, items.map((t) => el('span', { class: weak ? 'tag weak' : 'tag' }, t)));
}

function refList(refs) {
  if (!refs || !refs.length) return null;
  return el('div', { class: 'refs' }, refs.map((r) => el('a', { href: r.url, target: '_blank', rel: 'noopener' }, r.label)));
}

function renderMore(id) {
  const engine = ENGINE_BY_ID[id];
  const r = state.results[id];
  const box = $('.engine-more', engineEls[id]);
  box.innerHTML = '';
  box.append(el('p', {}, engine.about));
  if (r && r.note) box.append(el('div', { class: 'note' }, r.note));
  if (r && r.detail && Object.keys(r.detail).length) {
    const dl = el('dl', { class: 'kv' });
    for (const [k, v] of Object.entries(r.detail)) dl.append(el('dt', {}, k), el('dd', {}, String(v)));
    box.append(el('h4', {}, 'This comparison'), dl);
  }
  box.append(el('h4', {}, 'How it is scored'), el('p', {}, `Measures: ${engine.metric}. ${engine.thresholds}`));
  if (engine.pipeline) box.append(el('p', { class: 'muted small' }, `Preprocessing: ${engine.pipeline}`));
  box.append(el('h4', {}, 'Survives'), tagList(engine.robust), el('h4', {}, 'Fooled or broken by'), tagList(engine.weak, true));
  const refs = refList(engine.refs);
  if (refs) box.append(el('h4', {}, 'References'), refs);
  if (r && r.ms !== undefined) box.append(el('p', { class: 'muted small' }, `Computed in ${r.ms} ms.`));
}

function updateGroupCounts() {
  for (const g of GROUPS) {
    const sec = $(`.group[data-group="${g.id}"] .group-count`);
    if (!sec) continue;
    const ids = ENGINES.filter((e) => e.group === g.id).map((e) => e.id);
    const counted = ids.filter((id) => COUNTED.has(state.results[id]?.verdict));
    const flagged = counted.filter((id) => FLAGGED.has(state.results[id].verdict)).length;
    sec.innerHTML = counted.length ? `<b>${flagged}</b> / ${counted.length} flagged` : '';
    for (const dot of $$(`.group[data-group="${g.id}"] .gdot`)) {
      const v = state.results[dot.dataset.id]?.verdict || 'pending';
      dot.dataset.verdict = v === 'skipped' || v === 'error' ? 'na' : v;
    }
  }
}

// ------------------------------------------------------------------ spotlight

const SPOTLIGHT = [
  {
    id: 'sscd',
    title: 'SSCD',
    sub: 'Meta’s copy detector (official setting)',
    ticks: [0.5, 0.75],
    zones: [
      ['partial', 0.5, 0.75],
      ['match', 0.75, 1],
    ],
  },
  {
    id: 'sscdAligned',
    title: 'SSCD after alignment',
    sub: 'crop, rotation or mirror undone first',
    ticks: [0.5, 0.75],
    zones: [
      ['partial', 0.5, 0.75],
      ['match', 0.75, 1],
    ],
  },
  {
    id: 'sscdLarge',
    title: 'SSCD · Somepalli et al.',
    sub: 'replication threshold 0.5',
    ticks: [0.5, 0.7],
    zones: [['match', 0.5, 1]],
  },
  {
    id: 'dreamsim',
    title: 'DreamSim (for contrast)',
    sub: 'how alike people would call them',
    // a distance: plotted as 1 − d so that further right means more alike
    invert: true,
    ticks: [0.4, 0.12],
    zones: [
      ['partial', 0.12, 0.4],
      ['match', 0, 0.12],
    ],
  },
  {
    id: 'clip',
    title: 'CLIP (for contrast)',
    sub: 'semantic, not copy detection',
    ticks: [0.75, 0.95],
    zones: [
      ['partial', 0.75, 0.95],
      ['match', 0.95, 1],
    ],
  },
];

function renderSpotlight() {
  const box = $('#spotlight');
  box.innerHTML = '';
  box.append(
    el('h3', {}, '★ Copy detection with SSCD'),
    el(
      'p',
      { class: 'spotlight-intro' },
      'SSCD (Pizzi et al., CVPR 2022) is trained to recognise edited copies and is the measure Somepalli et al. used to find training-data replication in Stable Diffusion. Scores are cosine similarities; the bars show the published thresholds. For contrast, DreamSim shows how alike people would judge the two images (a distance, so further right is more alike) and CLIP whether they depict similar things — neither is copy detection.',
    ),
  );
  for (const m of SPOTLIGHT) {
    const track = el('div', { class: 'meter-track' });
    const pos = (x) => (m.invert ? 1 - x : x);
    for (const [kind, from, to] of m.zones) {
      const [l, r] = [pos(from), pos(to)].sort((a, b) => a - b);
      track.append(el('div', { class: `meter-zone ${kind}`, style: `left:${l * 100}%;width:${(r - l) * 100}%` }));
    }
    for (const t of m.ticks) track.append(el('div', { class: 'meter-tick', style: `left:${pos(t) * 100}%` }, el('span', {}, t.toFixed(2))));
    track.append(el('div', { class: 'meter-marker', style: 'left:0%', hidden: true }));
    box.append(
      el(
        'div',
        { class: 'meter', 'data-id': m.id, 'data-verdict': 'pending' },
        el('div', { class: 'meter-name' }, m.title, el('small', {}, m.sub)),
        track,
        el('div', { class: 'meter-value' }, el('span', { class: 'meter-status' }, 'waiting')),
      ),
    );
    updateSpotlight(m.id);
  }
}

function updateSpotlight(id) {
  const row = $(`.meter[data-id="${id}"]`);
  if (!row) return;
  const r = state.results[id];
  const marker = $('.meter-marker', row);
  const value = $('.meter-value', row);
  const engine = ENGINE_BY_ID[id];
  if (r && typeof r.value === 'number' && COUNTED.has(r.verdict)) {
    row.dataset.verdict = r.verdict;
    marker.hidden = false;
    const m = SPOTLIGHT.find((x) => x.id === id);
    const p = m?.invert ? 1 - r.value : r.value;
    marker.style.left = `${Math.max(0, Math.min(1, p)) * 100}%`;
    value.textContent = r.value.toFixed(3);
  } else {
    row.dataset.verdict = 'pending';
    marker.hidden = true;
    let text = 'waiting';
    if (r?.verdict === 'skipped') text = 'turned off';
    else if (r?.verdict === 'error') text = 'error';
    else if (engine.model && isDisabled(engine.model)) text = 'turned off';
    value.innerHTML = '';
    if (r?.verdict === 'running') value.append(loader('meter', { text: 'running', time: false, t0: state.scanT0, label: `${engine.name} is running` }));
    else value.append(el('span', { class: 'meter-status' }, text));
  }
}

// ------------------------------------------------------------------ summary

function headline(results) {
  const v = (id) => results[id]?.value;
  const verdict = (id) => results[id]?.verdict;
  const has = (id) => typeof v(id) === 'number' && COUNTED.has(verdict(id));
  const whole = has('sscd') ? v('sscd') : null;
  const aligned = has('sscdAligned') ? v('sscdAligned') : null;
  // after undoing a crop or rotation, SSCD can see a copy the whole-image score misses
  const useAligned = aligned !== null && (whole === null || aligned > whole + 0.05);
  const sscd = useAligned ? aligned : whole;
  const how = useAligned ? ` after lining B up with A (whole images: ${whole === null ? '—' : whole.toFixed(3)})` : '';
  const geomMatch = ['orb', 'akaze', 'brisk'].some((id) => verdict(id) === 'match');
  if (verdict('sha256') === 'identical') {
    return { tone: 'identical', title: 'Identical files', sub: 'The two files are byte-for-byte the same — a verbatim copy.' };
  }
  const stripped = results.cmi?.display === 'removed in B';
  if (verdict('payload') === 'identical') {
    return {
      tone: 'identical',
      title: stripped ? 'Same image, credits stripped' : 'Same image, different metadata',
      sub: `The compressed picture is byte-for-byte identical; only the metadata differs${stripped ? ' — and B no longer carries A’s creator and copyright fields' : ''}.`,
    };
  }
  if (verdict('pixels') === 'identical') {
    return { tone: 'identical', title: 'Identical pixels', sub: 'The files differ, but every decoded pixel is the same: the same image saved with different metadata or encoding.' };
  }
  if (verdict('crop') === 'identical') {
    return { tone: 'identical', title: 'Verbatim crop', sub: `One image is an unaltered cut-out of the other (${results.crop.detail?.['best placement'] || ''}).` };
  }
  // blank or pure-noise images: the learned detectors have nothing to go on
  const odd = ['a', 'b'].filter((side) => state.info?.[side]?.structure && state.info[side].structure !== 'normal');
  if (odd.length) {
    const what = odd.map((side) => `${side.toUpperCase()} is ${state.info[side].structure === 'flat' ? 'almost a single colour' : 'random noise'}`).join(' and ');
    return {
      tone: 'none',
      title: 'Too little structure to judge',
      sub: `${what}, so the copy detectors and other learned scores are not meaningful here (they can rate any two blank or noisy images as alike). Only the exact and pixel tests below apply.`,
    };
  }
  const credits = stripped ? ' B also drops A’s creator and copyright fields.' : '';
  const derived =
    verdict('c2pa') === 'match'
      ? ` Its signed Content Credentials agree: ${results.c2pa.display}.`
      : verdict('lineage') === 'match'
        ? ` Its edit history agrees: ${results.lineage.display}.`
        : '';
  if (sscd !== null && sscd >= 0.75) {
    return { tone: 'match', title: 'Copy detected', sub: `SSCD scores ${sscd.toFixed(3)}${how}, above the 0.75 copy threshold: B looks like an edited copy of A.${derived}${credits}` };
  }
  if (sscd !== null && sscd >= 0.5) {
    return { tone: 'partial', title: 'Possible partial copy', sub: `SSCD scores ${sscd.toFixed(3)}${how}: above 0.5, where Somepalli et al. found strong visual similarity and likely partial copies, but below the 0.75 copy threshold.${derived}${credits}` };
  }
  if (sscd === null && (verdict('pdq') === 'match' || verdict('pdqDihedral') === 'match' || geomMatch)) {
    return { tone: 'match', title: 'Likely copy', sub: 'Perceptual hashes or keypoint geometry indicate the same image (SSCD did not run).' };
  }
  if (geomMatch) {
    return { tone: 'partial', title: 'Shared content', sub: 'Keypoint matching finds a region the images have in common, although SSCD does not consider the whole image a copy.' };
  }
  const clip = has('clip') ? v('clip') : null;
  const dino = has('dino') ? v('dino') : null;
  const ds = has('dreamsim') ? v('dreamsim') : null;
  const share = state.visuals?.dino?.mutualShare;
  if ((clip !== null && clip >= 0.75) || (dino !== null && dino >= 0.4) || (share !== undefined && share >= 0.1) || (ds !== null && ds <= 0.4)) {
    const people = ds !== null && ds <= 0.4 ? ` DreamSim, trained on human judgments, puts them ${ds.toFixed(2)} apart — people would call them alike.` : '';
    return { tone: 'similar', title: 'Similar subject, not a copy', sub: `Semantic models (CLIP / DINOv2) see related content, but the copy detectors do not flag it.${people}` };
  }
  return { tone: 'none', title: 'No meaningful similarity', sub: 'The copy detectors, hashes and keypoints all treat these as different images.' };
}

// The similarity spectrum: seven increasingly loose meanings of "the same".
const LEVELS = [
  {
    id: 'file',
    label: 'Same file',
    tests: 'SHA-256, SHA-1, MD5',
    means: 'A byte-for-byte duplicate: the file itself was copied.',
  },
  {
    id: 'pixels',
    label: 'Same pixels',
    tests: 'Pixel-exact match',
    means: 'The same image saved as a different file (new metadata or a lossless format).',
  },
  {
    id: 'resaved',
    label: 'Re-saved',
    tests: 'PDQ, pHash, dHash, SSIM, PSNR',
    means: 'The same image after compression or resizing — still a reproduction of the whole work.',
  },
  {
    id: 'edited',
    label: 'Edited copy',
    tests: 'SSCD ≥ 0.75, PDQ + rotations, keypoints',
    means: 'A reproduction with changes such as filters, captions or crops. The question shifts to what the changes add — derivative work, parody, fair use.',
  },
  {
    id: 'part',
    label: 'Shared part',
    tests: 'Crop search, keypoints, SSCD 0.5–0.75, objects',
    means: 'Part of one image appears in the other. Courts then ask whether that part is protected expression, and whether it is a substantial part.',
  },
  {
    id: 'subject',
    label: 'Similar subject',
    tests: 'DINOv2, CLIP, matching parts',
    means: 'The same kind of subject, pose, composition or style without copied pixels. Ideas, subjects and styles are not protected on their own, although an original selection and arrangement can be.',
  },
  {
    id: 'none',
    label: 'Unrelated',
    tests: 'every test below its threshold',
    means: 'As far as these tests can tell, these are different works.',
  },
];

function similarityLevel(R, V) {
  const val = (id) => (typeof R[id]?.value === 'number' && COUNTED.has(R[id].verdict) ? R[id].value : null);
  const is = (id, ...vs) => vs.includes(R[id]?.verdict);
  const sscd = val('sscd');
  if (is('sha256', 'identical')) return { id: 'file', why: 'the SHA-256 digests are identical' };
  if (is('payload', 'identical')) return { id: 'pixels', why: 'the compressed image data is byte-identical; only the metadata differs' };
  if (is('pixels', 'identical')) return { id: 'pixels', why: 'every decoded pixel is identical' };
  const odd = ['a', 'b'].some((side) => state.info?.[side]?.structure && state.info[side].structure !== 'normal');
  if (odd) return { id: 'none', why: 'one of the images is blank or random noise, so the similarity tests have nothing meaningful to compare' };
  if ((is('pdq', 'match') || (is('phash', 'match') && is('dhash', 'match'))) && (is('msssim', 'match') || is('ssim', 'match'))) {
    return { id: 'resaved', why: 'the perceptual hashes match and the pixels line up' };
  }
  // SSCD after undoing a crop, rotation or mirror: a copy of the whole, or of a part?
  const aligned = val('sscdAligned');
  const ov = V?.annotations?.overlap;
  if (aligned !== null && aligned >= 0.75 && (sscd === null || aligned > sscd + 0.05)) {
    const cover = ov ? Math.min(ov.coverA, ov.coverB) : 1;
    if (cover >= 0.6) return { id: 'edited', why: `once B is lined up with A, SSCD scores ${aligned.toFixed(2)}` };
    return { id: 'part', why: `SSCD scores ${aligned.toFixed(2)} on the region the two share (${Math.round(100 * (ov?.coverA ?? 1))}% of A)` };
  }
  if (sscd !== null && sscd >= 0.75) return { id: 'edited', why: `SSCD scores ${sscd.toFixed(2)}, above the 0.75 copy threshold` };
  if (is('pdqDihedral', 'match')) return { id: 'edited', why: 'PDQ matches once B is rotated or mirrored' };
  if (is('crop', 'identical', 'match')) return { id: 'part', why: 'one image appears inside the other' };
  if (sscd !== null && sscd >= 0.5) return { id: 'part', why: `SSCD scores ${sscd.toFixed(2)}, the range Somepalli et al. associate with partial copies` };
  // as in the headline: alignment never lifted an unrelated benchmark pair to 0.5
  if (aligned !== null && aligned >= 0.5 && (sscd === null || aligned > sscd + 0.05)) {
    return { id: 'part', why: `once B is lined up with A, SSCD scores ${aligned.toFixed(2)} on the region they share, the range Somepalli et al. associate with partial copies` };
  }
  const kp = ['orb', 'akaze', 'brisk'].filter((id) => is(id, 'match'));
  if (kp.length) return { id: 'part', why: `${kp.map((k) => k.toUpperCase()).join(', ')} keypoints match in one consistent geometry` };
  const dino = val('dino');
  const clip = val('clip');
  const share = V?.dino?.mutualShare;
  const dsv = val('dreamsim');
  const sem = [
    dsv !== null && dsv <= 0.4 ? `DreamSim distance ${dsv.toFixed(2)} (people would call them alike)` : null,
    dino !== null && dino >= 0.4 ? `DINOv2 ${dino.toFixed(2)}` : null,
    clip !== null && clip >= 0.75 ? `CLIP ${clip.toFixed(2)}` : null,
    share !== undefined && share >= 0.1 ? `${Math.round(share * 100)}% of patches find a counterpart` : null,
  ].filter(Boolean);
  if (sem.length) return { id: 'subject', why: `${sem.join(', ')} — but the copy detectors stay below their thresholds` };
  return { id: 'none', why: 'no test found a meaningful resemblance' };
}

function renderSpectrum() {
  const box = $('#spectrum');
  if (!box) return;
  const done = !state.scanning && Object.keys(state.results).length > 0;
  box.hidden = !done && !state.scanning;
  if (box.hidden) return;
  const label = el('div', { class: 'spectrum-label' }, 'Where this pair sits on the similarity spectrum');
  if (!done) {
    // the rungs, waiting; built once per scan so the indicator keeps running
    if (box.dataset.pending) return;
    box.dataset.pending = '1';
    box.replaceChildren(
      label,
      el('ol', { class: 'spectrum-steps pending' }, LEVELS.map((l) => el('li', { title: `${l.label}: detected by ${l.tests}` }, l.label))),
      loader('spectrum', { t0: state.scanT0, label: 'Placing the pair on the similarity spectrum' }),
    );
    return;
  }
  delete box.dataset.pending;
  const level = similarityLevel(state.results, state.visuals);
  const info = LEVELS.find((l) => l.id === level.id);
  box.innerHTML = '';
  box.append(
    label,
    el(
      'ol',
      { class: 'spectrum-steps' },
      LEVELS.map((l) => el('li', { class: l.id === level.id ? 'active' : '', title: `${l.label}: detected by ${l.tests}` }, l.label)),
    ),
    el('p', { class: 'spectrum-means' }, el('b', {}, `${info.label}: `), `${level.why}. `, info.means),
  );
}

function renderSpectrumGuide() {
  const box = $('#spectrumGuide');
  if (!box) return;
  box.append(
    el('h3', {}, 'Seven meanings of “similar”'),
    el(
      'p',
      { class: 'muted' },
      'From most to least literal. Each scan places the pair on this scale using the most specific level that its tests support.',
    ),
    el(
      'ol',
      { class: 'guide-steps' },
      LEVELS.map((l) => el('li', {}, el('strong', {}, l.label), el('span', { class: 'guide-tests' }, l.tests), el('p', {}, l.means))),
    ),
  );
}

const KEY_CHIPS = ['sscd', 'sscdLarge', 'pdq', 'phash', 'ssim', 'orb', 'clip'];

function updateSummary() {
  const results = state.results;
  const finished = Object.values(results).filter((r) => r.verdict !== 'running');
  const counted = finished.filter((r) => COUNTED.has(r.verdict));
  const flagged = counted.filter((r) => FLAGGED.has(r.verdict)).length;
  const partial = counted.filter((r) => r.verdict === 'partial').length;
  const total = counted.length;

  $('#gaugeNum').textContent = flagged;
  $('#gaugeDen').textContent = `/ ${total}`;
  const C = 2 * Math.PI * 52;
  const fFlag = total ? flagged / total : 0;
  const fPart = total ? partial / total : 0;
  const flagCircle = $('.gauge-flag');
  const partCircle = $('.gauge-partial');
  flagCircle.style.strokeDasharray = `${fFlag * C} ${C}`;
  partCircle.style.strokeDasharray = `${fPart * C} ${C}`;
  partCircle.style.strokeDashoffset = `${-fFlag * C}`;

  const n = ENGINES.length;
  $('#progressBar').style.width = `${Math.floor(scanProgress() * 100)}%`;
  const summary = $('.summary');
  if (state.scanning) {
    $('#summaryKicker').textContent = `Scanning… ${finished.length} of ${n} tests`;
    $('#progressText').textContent = downloadSummary() || `${partial} partial · ${flagged} flagged so far`;
  } else if (finished.length) {
    $('#summaryKicker').textContent = `${flagged} of ${total} tests flagged these images as similar${partial ? ` · ${partial} partial` : ''}`;
    $('#progressText').textContent = `Finished in ${(state.elapsed / 1000).toFixed(1)} s`;
  }
  if (finished.length) {
    const h = headline(results);
    const tiny = ['a', 'b'].filter((k) => state.slots[k] && Math.min(state.slots[k].w, state.slots[k].h) < 64);
    const caveat = tiny.length
      ? ` Caution: image ${tiny.map((k) => k.toUpperCase()).join(' and ')} is under 64 px on one side, so most scores are unreliable.`
      : '';
    summary.dataset.tone = h.tone;
    $('#summaryTitle').textContent = state.scanning && !results.sscd && h.tone === 'none' ? 'Running tests…' : h.title;
    $('#summarySub').textContent = (state.scanning && !results.sscd && h.tone === 'none' ? 'Fast tests first; neural models follow.' : h.sub) + caveat;
  } else {
    $('#summaryTitle').textContent = 'Running tests…';
    $('#summarySub').textContent = '';
  }
  renderSpectrum();
  const chips = $('#summaryChips');
  chips.innerHTML = '';
  for (const id of KEY_CHIPS) {
    const r = results[id];
    if (!r || !COUNTED.has(r.verdict)) continue;
    const e = ENGINE_BY_ID[id];
    chips.append(el('span', { class: 'chip', 'data-verdict': r.verdict }, e.name.replace(' copy detector', '').replace(' · replication-study setting', ' (Somepalli)').replace(' keypoints + RANSAC', ''), el('b', {}, r.display)));
  }
}

function downloadSummary() {
  const active = Object.entries(state.downloads).filter(([, d]) => d.phase !== 'done');
  if (!active.length) return '';
  const [key, d] = active[0];
  return `Downloading ${MODEL_INFO[key].name}: ${formatBytes(d.loaded)} / ${formatBytes(d.total)}`;
}

function renderDownloadStatus() {
  if (state.scanning) $('#progressText').textContent = downloadSummary() || $('#progressText').textContent;
}

// ------------------------------------------------------------------ visual comparison

// Lenses for the "Where it's similar" tab, grouped by the question they answer.
const LENS_GROUPS = [
  {
    title: 'Marked up',
    note: 'Circles, numbers and arrows that point at what changed and what was carried over.',
    ids: ['diff', 'evidence', 'regions', 'two', 'pose', 'probe', 'cover'],
  },
  {
    title: 'Same content?',
    note: 'Learned features. They survive crops, flips and redrawing — and also respond to mere resemblance.',
    ids: ['parts', 'heat'],
  },
  {
    title: 'Same details?',
    note: 'Distinctive local patterns in one consistent geometry: strong evidence that one image was made from the other.',
    ids: ['matches', 'aligned'],
  },
  {
    title: 'Same pixels?',
    note: 'Location-by-location comparison after stretching B onto A. Only meaningful when the images line up.',
    ids: ['deltaE', 'ssim', 'swipe', 'blink', 'side'],
  },
];

const VIEWS = {
  diff: {
    label: 'Differences',
    need: (v) => v.annotations?.aligned,
    what: 'B is laid over A exactly: first with the keypoint transform, then with a fine, local alignment that undoes small shifts from scanning or resizing. Every spot where nothing nearby in the other image has the same colour is circled, and each number marks the same place in both images.',
    means: 'This is “spot the difference”: what was added, removed or changed between two versions of one picture. A handful of small changes means B is essentially A; the copyright question is then whether those changes add anything of substance.',
  },
  regions: {
    label: 'Matching regions',
    need: (v) => v.annotations?.regions?.length,
    what: 'DINOv2 patches of A and B that are each other’s best match are grouped into regions that move together. Each region is circled in its own colour in both images and joined by an arrow from A to B, labelled with what the object detector sees there and how alike the patches are on average.',
    means: 'Arrows between the same figures, objects or arrangement show which elements B shares with A — the “what was taken?” question — even when B was redrawn, restaged or rearranged. Two works can also share mere subject matter (two different cats match too), so ask whether the match lies in protectable expression.',
    ref: { label: 'Amir et al., Deep ViT Features as Dense Visual Descriptors (2021)', url: 'https://arxiv.org/abs/2112.05814' },
  },
  parts: {
    label: 'Matching parts',
    need: (v) => v.dino,
    what: 'DINOv2 (Meta AI) describes every 14×14-pixel patch of each image with a feature vector. Each patch of A is matched to the patch of B it most resembles, keeping only pairs that choose each other (mutual nearest neighbours). A dot of the same colour marks the two halves of each match; the colour comes from the dot’s position in A.',
    means: 'Matches that land on the same figures, objects or arrangement show which elements the two works share — the “what was taken?” question. They also connect things that are merely the same kind of thing (two different cats match eye to eye), so shared subject matter shows up here too; on its own that is not evidence of copied expression.',
    ref: { label: 'Amir et al., Deep ViT Features as Dense Visual Descriptors (2021)', url: 'https://arxiv.org/abs/2112.05814' },
  },
  heat: {
    label: 'Similarity heat map',
    need: (v) => v.dino,
    what: 'Each patch is coloured by how close its best match in the other image is: bright means something very similar exists in the other image, uncoloured means nothing like it does.',
    means: 'For a crop, the copied area lights up and the rest stays clear; for an edited copy, the edits show up as gaps. Plain textures such as sky or walls can light up even in unrelated photos.',
    ref: { label: 'Oquab et al., DINOv2 (2023)', url: 'https://arxiv.org/abs/2304.07193' },
  },
  evidence: {
    label: 'Copy evidence (SSCD)',
    need: (v) => v.sscd?.links,
    what: 'SSCD’s copy score is one number, but because the model adds up evidence from every location of each image, the score splits exactly into pairs: a location in A together with a location in B. Each part of A is joined to the part of B it pairs with most strongly; neighbouring parts that move together form one link, drawn in one colour with an arrow whose width is its share of the score. The bar underneath adds the links up to the score.',
    means: 'This is what the copy detector itself relied on — not a separate guess. Arrows that run parallel mean B reproduces A in place; arrows that cross mean a mirror image; arrows that fan out from a small part of A mean B is an enlarged crop. Evidence on a watermark, caption or border means the score may reflect that shared element more than the work itself.',
    ref: { label: 'Eberle et al., Building and Interpreting Deep Similarity Models (BiLRP), TPAMI 2020', url: 'https://arxiv.org/abs/2003.05431' },
  },
  two: {
    label: 'Copied or similar?',
    need: (v) => v.dino && v.sscd,
    what: 'Two models look at every region. The copy detector (SSCD) says where its evidence for “B is a copy of A” comes from; the look-alike model (DINOv2) says whether the region has a close counterpart in the other image. Red marks regions where both agree; amber marks regions that look alike but carry no copy evidence.',
    means: 'This is the idea/expression line in pictures. Amber regions share a subject, pose or composition — the kind of similarity two independent photographers can produce, and which copyright usually leaves free. Red regions are where the copy detector sees reproduced expression. The thresholds are heuristic: treat the map as a prompt for the filtration step, not as its answer.',
    ref: { label: 'Somepalli et al., Diffusion Art or Digital Forgery? (CVPR 2023)', url: 'https://arxiv.org/abs/2212.03860' },
  },
  pose: {
    label: 'Pose',
    need: (v) => v.pose,
    what: 'ViTPose finds 17 body joints for each person the object detector found. The best-matching pair of people is drawn in blue, with B’s pose laid over A’s (dashed orange) after removing differences in position, size, rotation and — if it fits better — mirroring. Joint dots show how closely each joint agrees.',
    means: 'Courts treat a pose on its own as an idea: in Rentmeester v. Nike, Jordan’s grand-jeté pose was free for Nike to use, and the comparison turned on how the photographs selected and arranged everything else. A high pose score therefore says the same idea was used, not that expression was copied.',
    ref: { label: 'Xu et al., ViTPose (NeurIPS 2022)', url: 'https://arxiv.org/abs/2204.12484' },
  },
  cover: {
    label: 'Cover-up test',
    need: (v) => v.sscd?.pairs,
    what: 'Paint over parts of either image; the painted areas are filled with the image’s average colour and SSCD scores the covered pair again. Beside the real score you see the prediction from the evidence split: what the score would be if the covered locations simply stopped contributing.',
    means: 'This is filtration by hand. Courts first set aside what is not protected — ideas, poses, stock elements, scènes à faire — and only then compare what is left. Cover the unprotected parts and see whether any copy evidence survives. When the prediction and the real score agree, the explanation is faithful for that cover-up; when they disagree, the network is using context around the covered area.',
    ref: { label: 'Petsiuk et al., RISE: randomized input sampling for explanation (2018) — the deletion test', url: 'https://arxiv.org/abs/1806.07421' },
  },
  probe: {
    label: 'Point and compare',
    need: (v) => v.dino?.featsA || v.sscd?.pairs,
    what: 'Point at any spot in A or B. The other image lights up wherever something resembles that spot, and an arrow lands on the closest match. Choose DINOv2 features (what things look like, robust to redrawing) or SSCD’s pairwise copy evidence (what the copy detector pairs with that spot).',
    means: 'A hands-on test of correspondence: if the girl’s face in A lands on the girl’s face in B, and the background lands on the background, the two works share that element. If points land somewhere random, with dim glows, nothing specific corresponds.',
    ref: { label: 'Amir et al., Deep ViT Features as Dense Visual Descriptors (2021)', url: 'https://arxiv.org/abs/2112.05814' },
  },
  matches: {
    label: 'Keypoint matches',
    need: (v) => v.matches,
    what: 'Distinctive corners and blobs found in both images, joined when their descriptors match and when one geometric transform (a homography) explains them all.',
    means: 'A large, geometrically consistent set of identical details is hard to produce by coincidence: it suggests one image was made from the other, or both from the same source — even after cropping, scaling or rotation.',
  },
  aligned: {
    label: 'Aligned overlay',
    need: (v) => v.aligned,
    what: 'B warped onto A with the keypoint transform. Fade between them, or show what still differs.',
    means: 'Once a crop, resize or rotation is undone, whatever still differs is the actual edit.',
  },
  deltaE: {
    label: 'Colour difference',
    need: (v) => v.deltaE,
    what: 'Perceptual colour difference (CIEDE2000) at every pixel after stretching B to A’s size; bright means a large change (ΔE ≥ 25).',
    means: 'Shows exactly which pixels changed — but only when the images line up. A crop or flip makes everything look “different”.',
  },
  ssim: {
    label: 'SSIM map',
    need: (v) => v.ssim,
    what: 'Local structural similarity (SSIM) in small windows; bright where brightness, contrast or structure differ.',
    means: 'Pixel-level metrics are how compression and image-quality research define “the same image”; they ignore meaning entirely.',
  },
  swipe: { label: 'Swipe', need: (v) => v.pairA, what: 'Drag across to wipe between A and B (B resized to A’s dimensions).', means: '' },
  blink: { label: 'Blink', need: (v) => v.pairA, what: 'Flicker between A and B; differences jump out as motion.', means: '' },
  side: { label: 'Side by side', need: () => true, what: 'The original files, scaled to fit.', means: '' },
};

function renderViewModes() {
  const box = $('#viewModes');
  box.innerHTML = '';
  const v = state.visuals || {};
  const ok = (id) => (id === 'side' ? !!(state.slots.a && state.slots.b) : !!VIEWS[id].need(v));
  const available = Object.keys(VIEWS).filter(ok);
  if (!state.viewChosen) state.view = defaultLens(v, available) || 'side';
  if (!available.includes(state.view)) state.view = available[0] || 'side';
  for (const g of LENS_GROUPS) {
    const ids = g.ids.filter(ok);
    if (!ids.length) continue;
    box.append(
      el(
        'div',
        { class: 'lens-group' },
        el('div', { class: 'lens-q' }, el('strong', {}, g.title), el('span', {}, g.note)),
        el(
          'div',
          { class: 'lens-pills' },
          ids.map((id) =>
            el(
              'button',
              {
                type: 'button',
                class: 'pill',
                'aria-pressed': String(id === state.view),
                onclick: () => {
                  state.view = id;
                  state.viewChosen = true;
                  renderViewModes();
                  renderVisual();
                },
              },
              VIEWS[id].label,
            ),
          ),
        ),
      ),
    );
  }
}

function renderLensExplain() {
  const box = $('#lensExplain');
  const lens = VIEWS[state.view];
  box.innerHTML = '';
  box.hidden = !lens || !state.visuals;
  if (box.hidden) return;
  box.append(el('h4', {}, 'What you’re seeing'), el('p', {}, lens.what));
  if (lens.means) box.append(el('h4', {}, 'What it means for copying'), el('p', {}, lens.means));
  if (lens.ref) box.append(el('p', { class: 'small' }, 'Method: ', el('a', { href: lens.ref.url, target: '_blank', rel: 'noopener' }, lens.ref.label)));
}

/** Side-by-side canvas of the two original images at a common height. */
function pairCanvas(maxCssWidth) {
  const A = state.slots.a.bitmap;
  const B = state.slots.b.bitmap;
  const H = 560;
  const wa = Math.round((A.width / A.height) * H);
  const wb = Math.round((B.width / B.height) * H);
  const gap = 36;
  const canvas = el('canvas', { width: wa + gap + wb, height: H });
  const g = canvas.getContext('2d');
  g.imageSmoothingQuality = 'high';
  g.drawImage(A, 0, 0, wa, H);
  g.drawImage(B, wa + gap, 0, wb, H);
  canvas.style.width = `${Math.min(maxCssWidth, canvas.width)}px`;
  return { canvas, g, ra: { x: 0, y: 0, w: wa, h: H }, rb: { x: wa + gap, y: 0, w: wb, h: H } };
}

/** Upsampled heat overlay of a gw × gh grid; `toV` maps a cell value to 0..1. */
function heatOverlay(g, grid, rect, toV, strength = 1) {
  const c = el('canvas', { width: grid.gw, height: grid.gh });
  const gc = c.getContext('2d');
  const id = gc.createImageData(grid.gw, grid.gh);
  for (let i = 0; i < grid.gw * grid.gh; i++) {
    const v = Math.max(0, Math.min(1, toV(grid.values[i])));
    const [r, gg, b] = lutColor(0.25 + 0.75 * v);
    id.data[i * 4] = r;
    id.data[i * 4 + 1] = gg;
    id.data[i * 4 + 2] = b;
    id.data[i * 4 + 3] = Math.round(255 * strength * 0.62 * v);
  }
  gc.putImageData(id, 0, 0);
  g.save();
  g.imageSmoothingEnabled = true;
  g.imageSmoothingQuality = 'high';
  g.drawImage(c, rect.x, rect.y, rect.w, rect.h);
  g.restore();
}

const cellCenter = (idx, grid, rect) => [rect.x + ((idx % grid.gw) + 0.5) * (rect.w / grid.gw), rect.y + (Math.floor(idx / grid.gw) + 0.5) * (rect.h / grid.gh)];
const dinoV = (sim) => (sim - 0.35) / 0.5;

function drawParts(d, maxCssWidth, withHeat, withLines) {
  const p = pairCanvas(maxCssWidth);
  if (withHeat) {
    heatOverlay(p.g, d.a, p.ra, dinoV, 0.8);
    heatOverlay(p.g, d.b, p.rb, dinoV, 0.8);
  }
  // Colour each matched pair by where it sits in A, so the same colour marks
  // the same part in both images (as in dense-correspondence papers).
  const cell = Math.min(p.ra.w / d.a.gw, p.ra.h / d.a.gh);
  const r = Math.max(4, cell * 0.32);
  const colour = (i) => {
    const x = (i % d.a.gw) / Math.max(1, d.a.gw - 1);
    const y = Math.floor(i / d.a.gw) / Math.max(1, d.a.gh - 1);
    return `hsl(${Math.round(300 * x)}, 90%, ${Math.round(38 + 30 * y)}%)`;
  };
  const pts = d.lines.map(([i, j, sim]) => ({ a: cellCenter(i, d.a, p.ra), b: cellCenter(j, d.b, p.rb), c: colour(i), sim }));
  if (withLines) {
    p.g.lineWidth = Math.max(1, p.canvas.width / 1100);
    for (const q of pts) {
      p.g.strokeStyle = q.c;
      p.g.globalAlpha = 0.25 + 0.5 * Math.max(0, Math.min(1, (q.sim - 0.5) / 0.45));
      p.g.beginPath();
      p.g.moveTo(...q.a);
      p.g.lineTo(...q.b);
      p.g.stroke();
    }
    p.g.globalAlpha = 1;
  }
  for (const q of pts) {
    for (const [x, y] of [q.a, q.b]) {
      p.g.beginPath();
      p.g.arc(x, y, r, 0, Math.PI * 2);
      p.g.fillStyle = q.c;
      p.g.fill();
      p.g.lineWidth = Math.max(1.5, r / 3);
      p.g.strokeStyle = 'rgba(255,255,255,0.9)';
      p.g.stroke();
    }
  }
  return p.canvas;
}

/** The most telling marked-up lens for these results. */
function defaultLens(v, available) {
  const ann = v?.annotations;
  const prefer = [];
  if (ann?.aligned && (ann.differences.length || ann.global)) prefer.push('diff');
  if (ann?.overlap && ann.overlap.coverA < 0.92) prefer.push('diff');
  if (v?.sscd?.links?.links.length && v.sscd.score >= 0.3) prefer.push('evidence');
  if (ann?.regions?.length) prefer.push('regions');
  if (ann?.aligned) prefer.push('diff');
  prefer.push('probe', 'parts', 'evidence', 'heat', 'swipe');
  return prefer.find((id) => available.includes(id));
}

const ANN_LENSES = new Set(['diff', 'regions', 'evidence', 'probe', 'two', 'cover', 'pose']);

function evidenceUnderlay(p) {
  const e = state.visuals.sscd;
  const peak = (grid) => Math.max(1e-9, ...grid.values);
  const strength = 0.45 * Math.max(0.15, Math.min(1, (e.score - 0.15) / 0.45));
  heatOverlay(p.g, e.a, p.a, (x) => x / peak(e.a), strength);
  heatOverlay(p.g, e.b, p.b, (x) => x / peak(e.b), strength);
}

/** Text under a marked-up lens. */
function annotationNote(id) {
  const v = state.visuals;
  const ann = v.annotations || {};
  if (id === 'diff') {
    const o = ann.overlap;
    const mirrored = ann.mirrored ? 'B is a mirror image of A: it was flipped back before comparing. ' : '';
    const crop = o && o.coverA < 0.92 ? ` The dashed outline marks the ${Math.round(o.coverA * 100)}% of A that B shows; the arrows run from its corners to B’s corners.` : '';
    if (ann.global) return `${mirrored}After alignment B differs from A almost everywhere (recoloured, filtered or redrawn), so there are no isolated spots to circle.${crop}`;
    if (!ann.differences.length) return `${mirrored}After alignment no part of B differs from A beyond tiny shifts: as far as the pixels go, B is a faithful copy of A.${crop}`;
    const n = ann.differences.length;
    return `${mirrored}${n} difference${n === 1 ? '' : 's'} circled, largest first.${crop} Point at (or tap) a number to single it out.`;
  }
  if (id === 'regions') {
    const n = ann.regions.length;
    const whole = ann.regions[0] && ann.regions[0].share >= 0.6;
    const hatched = v.dino && !ann.aligned ? ' Hatched areas have nothing like them in the other image — what B did not take from A, or added.' : '';
    return `${whole ? 'Nearly all of A reappears in B. ' : ''}${n} matching region${n === 1 ? '' : 's'}, joined A → B; the percentage is how alike their patches are.${hatched} Point at (or tap) a region to follow its arrow.`;
  }
  if (id === 'probe') return 'Point at any part of either image (or tap it). The other image lights up wherever something resembles that spot, and the arrow lands on the closest match. Click to pin a point.';
  if (id === 'cover') return 'Filtration by hand: paint over what you think is unprotectable (a pose, a background, a stock element) in either image and see how much copy evidence is left.';
  if (id === 'pose') {
    const b = v.pose.best;
    return `Best-matching people: pose similarity ${b.similarity.toFixed(2)}${b.mirrored ? ' (as a mirror image)' : ''} — ${b.similarity >= 0.8 ? 'the same pose' : b.similarity >= 0.55 ? 'a similar pose' : 'different poses'}. A pose alone is an idea, not protected expression (Rentmeester v. Nike).`;
  }
  if (id === 'two') return `Copy detector score ${v.sscd.score.toFixed(2)}${v.sscd.score < 0.4 ? ' — too low for any region to count as copied, so every match shows as “looks alike only”' : ''}. Red: copied; amber: looks alike only; clear: no counterpart.`;
  const e = v.sscd;
  const n = e.links?.links.length || 0;
  if (!n) return `SSCD score ${e.score.toFixed(3)}: ${e.score < 0.1 ? 'no copy evidence to trace.' : 'the evidence is spread thinly, with no part of A clearly supporting a part of B.'}`;
  return `SSCD score ${e.score.toFixed(3)}, split exactly by which part of A pairs with which part of B. ${state.evidenceMode === 'colour' ? 'Colour map: each part of B takes the colour of the parts of A it pairs with — a reversed rainbow means a mirror image, a stretched slice means a crop.' : 'Each arrow’s width is its share of the score; matching colours mark the two halves of each pair.'}${e.score < 0.5 ? ' The score is low, so treat these as weak hints.' : ''}`;
}

/** Where a box sits in its image, in words ("top left", "centre"). */
function placeName(box, img) {
  const x = (box[0] + box[2]) / 2 / img.w;
  const y = (box[1] + box[3]) / 2 / img.h;
  const v = y < 1 / 3 ? 'top' : y > 2 / 3 ? 'bottom' : '';
  const h = x < 1 / 3 ? 'left' : x > 2 / 3 ? 'right' : '';
  return [v, h].filter(Boolean).join(' ') || 'centre';
}

/** What the pattern of copy-evidence arrows says about how B was made from A. */
function arrowPattern(links, overlap) {
  const L = links.filter((k) => k.weight > 0);
  if (overlap && overlap.coverA < 0.9) return `The arrows spread out from the ${Math.round(overlap.coverA * 100)}% of A that B shows: B is a crop of A.`;
  if (L.length < 3) return '';
  const w = L.map((k) => k.weight);
  const W = w.reduce((t, x) => t + x, 0);
  const mean = (f) => L.reduce((t, k, i) => t + w[i] * f(k), 0) / W;
  const corr = (fa, fb) => {
    const ma = mean(fa);
    const mb = mean(fb);
    const cov = mean((k) => (fa(k) - ma) * (fb(k) - mb));
    const va = mean((k) => (fa(k) - ma) ** 2);
    const vb = mean((k) => (fb(k) - mb) ** 2);
    return { r: cov / Math.sqrt(Math.max(1e-9, va * vb)), spread: Math.sqrt(vb / Math.max(1e-9, va)) };
  };
  const x = corr((k) => k.ca[0], (k) => k.cb[0]);
  const y = corr((k) => k.ca[1], (k) => k.cb[1]);
  if (x.r < -0.6 && y.r > 0.6) return 'The arrows swap left and right: B is a mirror image of A.';
  if (y.r < -0.6 && x.r > 0.6) return 'The arrows swap top and bottom: B is upside down relative to A.';
  if (x.r > 0.6 && y.r > 0.6) {
    const grow = Math.sqrt(x.spread * y.spread);
    if (grow > 1.2) return 'The arrows fan out from a smaller area of A: B enlarges part of A (a crop).';
    if (grow < 0.75) return 'The arrows converge on a smaller area of B: A appears shrunk inside B.';
    return 'The arrows keep their places: each part of A reappears in the same position in B.';
  }
  return 'The arrows do not follow one simple pattern: B rearranges what it shares with A.';
}

/** "0.78 = 0.31 + 0.20 + … " — the copy score as the sum of its links. */
function scoreBreakdown(sscd) {
  const L = sscd.links;
  if (!L || !L.links.length) return '';
  const pos = Math.max(1e-9, L.positive);
  const bar = el(
    'div',
    { class: 'sb-bar', role: 'img', 'aria-label': 'Copy score split into its parts' },
    L.links.map((k, i) => el('i', { style: `width:${(100 * Math.max(0, k.weight)) / pos}%;background:${AV.PALETTE[i % AV.PALETTE.length]}`, title: `${i + 1}: +${k.weight.toFixed(3)}` })),
    L.rest > 0 ? el('i', { class: 'sb-rest', style: `width:${(100 * L.rest) / pos}%`, title: `elsewhere: +${L.rest.toFixed(3)}` }) : '',
  );
  const terms = [];
  L.links.forEach((k, i) => {
    if (i) terms.push(' + ');
    terms.push(el('b', { class: 'sb-chip', style: `background:${AV.PALETTE[i % AV.PALETTE.length]}` }, `${i + 1}`), ` ${k.weight.toFixed(2)}`);
  });
  if (L.rest > 0.005) terms.push(` + ${L.rest.toFixed(2)} elsewhere`);
  if (L.negative < -0.005) terms.push(` − ${(-L.negative).toFixed(2)} from parts that differ`);
  const pattern = arrowPattern(L.links, state.visuals.annotations?.overlap);
  return el('div', { class: 'score-break' }, bar, el('p', {}, el('strong', {}, `${sscd.score.toFixed(2)} ≈ `), ...terms), pattern ? el('p', { class: 'sb-pattern' }, pattern) : '');
}

/** Point-and-compare: the controls and the live canvas. */
function probeView(cssWidth) {
  const v = state.visuals;
  const A = state.slots.a.bitmap;
  const B = state.slots.b.bitmap;
  const sources = [];
  if (v.dino?.featsA) {
    const d = v.dino;
    const D = d.dims;
    const row = (side, idx) => {
      const [src, dst] = side === 'a' ? [d.featsA, d.featsB] : [d.featsB, d.featsA];
      const n = dst.length / D;
      const out = new Float32Array(n);
      const o = idx * D;
      for (let j = 0; j < n; j++) {
        let sum = 0;
        const q = j * D;
        for (let k = 0; k < D; k++) sum += src[o + k] * dst[q + k];
        out[j] = sum;
      }
      return out;
    };
    sources.push({ id: 'dino', label: 'What it looks like (DINOv2)', ga: d.a, gb: d.b, row, lo: () => 0.3, hi: () => 0.85, word: 'closest match', format: (x) => x.toFixed(2) });
  }
  if (v.sscd?.pairs) {
    const e = v.sscd;
    const nA = e.a.gw * e.a.gh;
    const nB = e.b.gw * e.b.gh;
    const row = (side, idx) => {
      const out = new Float32Array(side === 'a' ? nB : nA);
      if (side === 'a') for (let m = 0; m < nB; m++) out[m] = e.pairs[idx * nB + m];
      else for (let l = 0; l < nA; l++) out[l] = e.pairs[l * nB + idx];
      return out;
    };
    sources.push({
      id: 'sscd',
      label: 'Copy evidence (SSCD)',
      ga: e.a,
      gb: e.b,
      row,
      lo: () => 0,
      hi: (vals) => Math.max(1e-6, ...vals),
      word: 'strongest pairing',
      format: (x) => `${x >= 0 ? '+' : ''}${x.toFixed(3)}`,
    });
  }
  if (!sources.length) return el('p', { class: 'muted small' }, 'Turn on DINOv2 or SSCD in Settings to point and compare.');
  if (!sources.some((x) => x.id === state.probeSource)) state.probeSource = sources[0].id;
  const holder = el('div', { class: 'ann-stage' });
  const readout = el('p', { class: 'probe-readout' }, '');
  let pinned = null;
  let st = null;
  const build = () => {
    const src = sources.find((x) => x.id === state.probeSource);
    st = AV.probeStage(A, B, src, { cssWidth });
    holder.replaceChildren(st.canvas);
    st.canvas.style.cursor = 'crosshair';
    st.canvas.style.touchAction = 'none';
    const at = (e) => {
      const r = st.canvas.getBoundingClientRect();
      const x = ((e.clientX - r.left) / r.width) * st.canvas.width;
      const y = ((e.clientY - r.top) / r.height) * st.canvas.height;
      for (const side of ['a', 'b']) {
        const R = st[side];
        if (x >= R.x && x < R.x + R.w && y >= R.y && y < R.y + R.h) return { side, fx: (x - R.x) / R.w, fy: (y - R.y) / R.h };
      }
      return null;
    };
    const show = (pt) => {
      const res = st.draw(pt);
      readout.textContent = res
        ? `Pointing at ${res.side.toUpperCase()}: the ${src.word} in ${res.side === 'a' ? 'B' : 'A'} is ${src.format(res.value)}${src.id === 'dino' ? ' (cosine of DINOv2 patch features; 1 = identical, unrelated photos rarely pass 0.5)' : ' of the copy score, from this one pair of locations'}.`
        : '';
    };
    st.canvas.onpointermove = (e) => {
      if (!pinned) show(at(e));
    };
    st.canvas.onpointerleave = () => {
      if (!pinned) show(null);
    };
    st.canvas.onclick = (e) => {
      const pt = at(e);
      pinned = pinned || !pt ? null : pt;
      show(pt);
    };
    show(pinned);
  };
  const pills =
    sources.length > 1
      ? el(
          'div',
          { class: 'lens-pills probe-pills' },
          sources.map((src) =>
            el(
              'button',
              {
                type: 'button',
                class: 'pill',
                'aria-pressed': String(src.id === state.probeSource),
                onclick: (e) => {
                  state.probeSource = src.id;
                  for (const b of e.currentTarget.parentElement.children) b.setAttribute('aria-pressed', String(b === e.currentTarget));
                  build();
                },
              },
              src.label,
            ),
          ),
        )
      : '';
  build();
  return el('div', { class: 'ann-view ann-probe' }, pills, holder, readout);
}

/**
 * The cover-up test: paint over parts of either image and SSCD re-scores the
 * covered images. The first-order prediction from the pairwise split is
 * shown beside the real score, so students can see where the explanation
 * holds and where the network is more than the sum of its parts.
 */
function coverView(cssWidth) {
  const v = state.visuals;
  const bms = { a: state.slots.a.bitmap, b: state.slots.b.bitmap };
  const MAX = 640;
  const size = (bm) => {
    const s = Math.min(1, MAX / Math.max(bm.width, bm.height));
    return [Math.max(8, Math.round(bm.width * s)), Math.max(8, Math.round(bm.height * s))];
  };
  const masks = {};
  for (const side of ['a', 'b']) {
    const [w, h] = size(bms[side]);
    masks[side] = el('canvas', { width: w, height: h });
  }
  // mean colour of each image: what covered areas are filled with
  const mean = {};
  for (const side of ['a', 'b']) {
    const c = el('canvas', { width: 1, height: 1 });
    const g = c.getContext('2d', { willReadFrequently: true });
    g.drawImage(bms[side], 0, 0, 1, 1);
    const d = g.getImageData(0, 0, 1, 1).data;
    mean[side] = `rgb(${d[0]},${d[1]},${d[2]})`;
  }
  let brush = 0.07;
  const undo = [];
  const p = AV.stage(bms.a, bms.b, { cssWidth, gap: 34 });
  const holder = el('div', { class: 'ann-stage' }, p.canvas);
  p.canvas.style.cursor = 'crosshair';
  p.canvas.style.touchAction = 'none';
  const readout = el('p', { class: 'cover-readout' }, `SSCD on the uncovered images: ${v.sscd.score.toFixed(3)}. Paint over any part of either image.`);
  const redraw = () => {
    p.paint();
    for (const side of ['a', 'b']) {
      const r = p[side];
      const t = el('canvas', { width: masks[side].width, height: masks[side].height });
      const tg = t.getContext('2d');
      tg.drawImage(masks[side], 0, 0);
      tg.globalCompositeOperation = 'source-in';
      tg.fillStyle = 'rgba(239, 35, 60, 0.6)';
      tg.fillRect(0, 0, t.width, t.height);
      p.g.drawImage(t, r.x, r.y, r.w, r.h);
    }
  };
  const at = (e) => {
    const rect = p.canvas.getBoundingClientRect();
    const x = ((e.clientX - rect.left) / rect.width) * p.canvas.width;
    const y = ((e.clientY - rect.top) / rect.height) * p.canvas.height;
    for (const side of ['a', 'b']) {
      const r = p[side];
      if (x >= r.x && x < r.x + r.w && y >= r.y && y < r.y + r.h) return { side, fx: (x - r.x) / r.w, fy: (y - r.y) / r.h };
    }
    return null;
  };
  let stroke = null;
  const dab = (pt) => {
    const m = masks[pt.side];
    const g = m.getContext('2d');
    const rad = brush * Math.max(m.width, m.height);
    const x = pt.fx * m.width;
    const y = pt.fy * m.height;
    g.fillStyle = '#000';
    g.strokeStyle = '#000';
    g.lineCap = 'round';
    g.lineWidth = 2 * rad;
    if (stroke?.last && stroke.last.side === pt.side) {
      g.beginPath();
      g.moveTo(stroke.last.fx * m.width, stroke.last.fy * m.height);
      g.lineTo(x, y);
      g.stroke();
    }
    g.beginPath();
    g.arc(x, y, rad, 0, Math.PI * 2);
    g.fill();
    stroke.last = pt;
  };
  const snapshot = () => undo.push(Object.fromEntries(['a', 'b'].map((sd) => [sd, masks[sd].getContext('2d').getImageData(0, 0, masks[sd].width, masks[sd].height)])));
  p.canvas.onpointerdown = (e) => {
    const pt = at(e);
    if (!pt) return;
    p.canvas.setPointerCapture(e.pointerId);
    snapshot();
    stroke = { last: null };
    dab(pt);
    redraw();
  };
  p.canvas.onpointermove = (e) => {
    if (!stroke) return;
    const pt = at(e);
    if (pt) {
      dab(pt);
      redraw();
    }
  };
  p.canvas.onpointerup = () => {
    if (!stroke) return;
    stroke = null;
    score();
  };
  // fraction of each SSCD grid cell that is covered
  const coverage = (side) => {
    const grid = v.sscd[side];
    const c = el('canvas', { width: grid.gw, height: grid.gh });
    const g = c.getContext('2d', { willReadFrequently: true });
    g.imageSmoothingEnabled = true;
    g.imageSmoothingQuality = 'high';
    g.drawImage(masks[side], 0, 0, grid.gw, grid.gh);
    const d = g.getImageData(0, 0, grid.gw, grid.gh).data;
    return Float32Array.from({ length: grid.gw * grid.gh }, (_, i) => d[i * 4 + 3] / 255);
  };
  const predicted = () => {
    const R = v.sscd.pairs;
    if (!R) return null;
    const wa = coverage('a');
    const wb = coverage('b');
    const nA = wa.length;
    const nB = wb.length;
    let removed = 0;
    for (let l = 0; l < nA; l++) {
      for (let m = 0; m < nB; m++) {
        const keep = (1 - wa[l]) * (1 - wb[m]);
        removed += (1 - keep) * R[l * nB + m];
      }
    }
    return v.sscd.score - removed;
  };
  let seq = 0;
  async function score() {
    const my = ++seq;
    const covered = ['a', 'b'].some((sd) => coverage(sd).some((x) => x > 0.01));
    if (!covered) {
      readout.textContent = `SSCD on the uncovered images: ${v.sscd.score.toFixed(3)}. Paint over any part of either image.`;
      return;
    }
    readout.replaceChildren(loader('rescore', { text: 'Re-scoring with SSCD…', label: 'Re-scoring' }));
    const image = (side) => {
      const m = masks[side];
      const c = el('canvas', { width: m.width, height: m.height });
      const g = c.getContext('2d', { willReadFrequently: true });
      g.drawImage(bms[side], 0, 0, m.width, m.height);
      const t = el('canvas', { width: m.width, height: m.height });
      const tg = t.getContext('2d');
      tg.drawImage(m, 0, 0);
      tg.globalCompositeOperation = 'source-in';
      tg.fillStyle = mean[side];
      tg.fillRect(0, 0, t.width, t.height);
      g.drawImage(t, 0, 0);
      return { w: m.width, h: m.height, data: g.getImageData(0, 0, m.width, m.height).data };
    };
    try {
      const value = await rescore(image('a'), image('b'));
      if (my !== seq) return;
      const pred = predicted();
      const drop = v.sscd.score - value;
      readout.replaceChildren(
        el('strong', {}, `SSCD with your cover-ups: ${value.toFixed(3)}`),
        ` (uncovered ${v.sscd.score.toFixed(3)}, ${drop >= 0 ? 'down' : 'up'} ${Math.abs(drop).toFixed(3)}). `,
        pred === null ? '' : `The evidence split predicted ${pred.toFixed(3)}${Math.abs(pred - value) < 0.05 ? ' — close: the covered parts carried that much of the score.' : ' — the network reacts to more than the sum of the parts here (covering changes the features around the covered area too).'}`,
        value < 0.5 && v.sscd.score >= 0.5 ? ' Below 0.5, SSCD no longer treats B as a copy.' : '',
      );
    } catch (err) {
      readout.textContent = `Could not re-score: ${err.message}`;
    }
  }
  const btn = (label, fn, extra = {}) => el('button', { type: 'button', class: 'pill', onclick: fn, ...extra }, label);
  const sizes = el(
    'span',
    { class: 'cover-sizes' },
    [
      ['S', 0.035],
      ['M', 0.07],
      ['L', 0.13],
    ].map(([label, val]) =>
      btn(
        label,
        (e) => {
          brush = val;
          for (const b of e.currentTarget.parentElement.children) b.setAttribute('aria-pressed', String(b === e.currentTarget));
        },
        { 'aria-pressed': String(val === brush), title: `Brush ${label}` },
      ),
    ),
  );
  const coverTop = () => {
    const L = v.sscd.links?.links?.[0];
    if (!L) return;
    snapshot();
    for (const [side, cells] of [
      ['a', L.a],
      ['b', L.b],
    ]) {
      const grid = v.sscd[side];
      const m = masks[side];
      const g = m.getContext('2d');
      g.fillStyle = '#000';
      const cw = m.width / grid.gw;
      const ch = m.height / grid.gh;
      for (const i of cells) g.fillRect((i % grid.gw) * cw - 1, Math.floor(i / grid.gw) * ch - 1, cw + 2, ch + 2);
    }
    redraw();
    score();
  };
  const controls = el(
    'div',
    { class: 'lens-pills probe-pills cover-controls' },
    el('span', { class: 'muted small' }, 'Brush'),
    sizes,
    btn('Undo', () => {
      const last = undo.pop();
      if (!last) return;
      for (const sd of ['a', 'b']) masks[sd].getContext('2d').putImageData(last[sd], 0, 0);
      redraw();
      score();
    }),
    btn('Clear', () => {
      snapshot();
      for (const sd of ['a', 'b']) masks[sd].getContext('2d').clearRect(0, 0, masks[sd].width, masks[sd].height);
      redraw();
      score();
    }),
    v.sscd.links?.links?.length ? btn('Cover the strongest evidence', coverTop) : '',
  );
  redraw();
  return el('div', { class: 'ann-view ann-cover' }, controls, holder, readout);
}

/** A marked-up view: annotated canvas plus a numbered list of close-ups. */
function annotatedView(id, cssWidth) {
  if (id === 'probe') return probeView(cssWidth);
  if (id === 'pose') {
    const st = AV.drawPose(state.slots.a.bitmap, state.slots.b.bitmap, state.visuals.pose, { cssWidth });
    const legend = el(
      'div',
      { class: 'two-legend' },
      el('span', {}, el('i', { style: 'background:#2563eb' }), el('b', {}, 'Blue'), ' — each image’s own pose (the best-matching people; others in grey).'),
      el('span', {}, el('i', { style: 'background:#f59e0b' }), el('b', {}, 'Dashed orange on A'), ' — B’s pose laid over A’s after matching size, position and rotation.'),
      el('span', {}, el('i', { style: 'background:#16a34a' }), el('b', {}, 'Joint dots on A'), ' — green where B’s joint lands close, amber further, red far.'),
    );
    return el('div', { class: 'ann-view ann-pose' }, el('div', { class: 'ann-stage' }, st.canvas), legend);
  }
  if (id === 'cover') return coverView(cssWidth);
  if (id === 'two') return twoLensView(cssWidth);
  const v = state.visuals;
  const ann = v.annotations || { differences: [], regions: [] };
  const A = state.slots.a.bitmap;
  const B = state.slots.b.bitmap;
  const holder = el('div', { class: 'ann-stage' });
  let focus = null;
  let stageNow = null;
  const score = v.sscd?.score ?? 0;
  // what has no counterpart in the other image (DINOv2 best match below 0.42)
  const unmatched =
    id === 'regions' && v.dino && !ann.aligned
      ? {
          a: { grid: v.dino.a, pieces: AV.unmatchedPieces(v.dino.a, v.dino.a.values, 0.42) },
          b: { grid: v.dino.b, pieces: AV.unmatchedPieces(v.dino.b, v.dino.b.values, 0.42) },
        }
      : null;
  const items =
    id === 'diff'
      ? ann.differences.map((d) => {
          const kind = AV.KIND_TEXT[d.kind] || 'changed';
          return {
            n: d.n,
            color: AV.RED,
            title: kind[0].toUpperCase() + kind.slice(1),
            sub: [d.label, placeName(d.a, state.slots.a)].filter(Boolean).join(' · '),
            thumbs: d.thumbs,
          };
        })
      : id === 'regions'
        ? ann.regions.map((r, k) => ({
            n: r.n,
            color: AV.PALETTE[k % AV.PALETTE.length],
            title: AV.regionName(r),
            sub: `${Math.round(r.sim * 100)}% alike · ${Math.max(1, Math.round(r.share * 100))}% of A`,
            thumbs: r.thumbs,
          }))
        : (v.sscd?.links?.links || []).map((k, i) => {
            const extra = ann.copyLinks?.[i];
            const where = `${placeName(k.boxA, { w: 1, h: 1 })} of A → ${placeName(k.boxB, { w: 1, h: 1 })} of B`;
            const name = extra && (extra.labelA || extra.labelB) ? `${AV.regionName({ labelA: extra.labelA, labelB: extra.labelB, share: 0 })} · ${where}` : where;
            return {
              n: i + 1,
              color: AV.PALETTE[i % AV.PALETTE.length],
              title: name,
              sub: `+${k.weight.toFixed(2)} of the ${score.toFixed(2)} score${score > 0.05 ? ` (${Math.round((100 * k.weight) / score)}%)` : ''}`,
              thumbs: extra?.thumbs,
            };
          });
  const list = el(
    'ol',
    { class: 'ann-list' },
    items.map((it) =>
      el(
        'li',
        {
          'data-n': it.n,
          tabindex: '0',
          onmouseenter: () => setFocus(it.n),
          onmouseleave: () => setFocus(null),
          onfocus: () => setFocus(it.n),
          onblur: () => setFocus(null),
          onclick: () => setFocus(focus === it.n ? null : it.n),
        },
        el('span', { class: 'ann-num', style: `background:${it.color}` }, String(it.n)),
        el('span', { class: 'ann-thumbs' }, it.thumbs?.a ? canvasFrom(it.thumbs.a) : '', el('span', { class: 'ann-to', style: `color:${it.color}` }, '→'), it.thumbs?.b ? canvasFrom(it.thumbs.b) : ''),
        el('span', { class: 'ann-text' }, el('strong', {}, it.title), el('span', {}, it.sub)),
      ),
    ),
    unmatched
      ? ['a', 'b']
          .filter((side) => unmatched[side].pieces.length)
          .map((side) => {
            const share = unmatched[side].pieces.reduce((t, q) => t + q.share, 0);
            const where = [...new Set(unmatched[side].pieces.slice(0, 3).map((q) => placeName(q.box, { w: 1, h: 1 })))].join(', ');
            return el(
              'li',
              { class: 'ann-unmatched' },
              el('span', { class: 'ann-hatch', 'aria-hidden': 'true' }),
              el(
                'span',
                { class: 'ann-text' },
                el('strong', {}, side === 'a' ? 'Only in A' : 'Only in B'),
                el('span', {}, `${Math.round(share * 100)}% of ${side.toUpperCase()} has nothing like it in ${side === 'a' ? 'B' : 'A'} · ${where}`),
              ),
            );
          })
      : '',
  );
  const draw = () => {
    if (id === 'diff') stageNow = AV.drawDifferences(A, B, ann, { cssWidth, focus });
    else if (id === 'regions') stageNow = AV.drawRegions(A, B, ann, { cssWidth, focus, unmatched });
    else if (state.evidenceMode === 'colour' && v.sscd.pairs) stageNow = AV.drawColourMap(A, B, v.sscd, { cssWidth });
    else stageNow = AV.drawCopyLinks(A, B, v.sscd, { cssWidth, focus, underlay: evidenceUnderlay });
    stageNow.canvas.onclick = (e) => {
      const n = AV.hitTest(stageNow, e);
      setFocus(n === focus ? null : n);
      if (n) list.querySelector(`[data-n="${n}"]`)?.scrollIntoView({ block: 'nearest', behavior: 'smooth' });
    };
    if (stageNow.hits.length) stageNow.canvas.style.cursor = 'pointer';
    holder.replaceChildren(stageNow.canvas);
    for (const li of list.children) li.classList.toggle('on', Number(li.dataset.n) === focus);
  };
  function setFocus(n) {
    if (n === focus) return;
    focus = n;
    draw();
  }
  draw();
  const modes =
    id === 'evidence' && v.sscd?.pairs
      ? el(
          'div',
          { class: 'lens-pills probe-pills' },
          [
            ['arrows', 'Arrows'],
            ['colour', 'Colour map'],
          ].map(([m, label]) =>
            el(
              'button',
              {
                type: 'button',
                class: 'pill',
                'aria-pressed': String((state.evidenceMode || 'arrows') === m),
                onclick: (e) => {
                  state.evidenceMode = m;
                  for (const b of e.currentTarget.parentElement.children) b.setAttribute('aria-pressed', String(b === e.currentTarget));
                  draw();
                },
              },
              label,
            ),
          ),
        )
      : '';
  return el('div', { class: `ann-view ann-${id}` }, modes, holder, id === 'evidence' ? scoreBreakdown(v.sscd) : '', items.length ? list : '');
}

/** Copied or just similar? The copy detector's evidence against the look-alike model's matches. */
function twoLensView(cssWidth) {
  const v = state.visuals;
  const st = AV.drawTwoLenses(state.slots.a.bitmap, state.slots.b.bitmap, v, { cssWidth });
  const pct = (x) => `${Math.round(x * 100)}%`;
  const legend = el(
    'div',
    { class: 'two-legend' },
    el('span', {}, el('i', { class: 'sw-copied' }), el('b', {}, 'Copied'), ` — the copy detector’s evidence sits here and the look-alike model finds a counterpart (${pct(st.shares.copied[0])} of A, ${pct(st.shares.copied[1])} of B)`),
    el('span', {}, el('i', { class: 'sw-similar' }), el('b', {}, 'Looks alike only'), ` — a counterpart exists, but the copy detector does not count it as copied: same subject, pose or idea (${pct(st.shares.similar[0])} of A, ${pct(st.shares.similar[1])} of B)`),
  );
  return el('div', { class: 'ann-view ann-two' }, el('div', { class: 'ann-stage' }, st.canvas), legend);
}

const WHERE_LENSES = [
  { id: 'diff', label: 'Differences', need: (v) => v.annotations?.aligned },
  { id: 'evidence', label: 'Copy evidence', need: (v) => v.sscd?.links },
  { id: 'regions', label: 'Matching regions', need: (v) => v.annotations?.regions?.length },
  { id: 'two', label: 'Copied or similar?', need: (v) => v.dino && v.sscd },
  { id: 'probe', label: 'Point and compare', need: (v) => v.dino?.featsA || v.sscd?.pairs },
  { id: 'pose', label: 'Pose', need: (v) => v.pose },
  { id: 'cover', label: 'Cover-up test', need: (v) => v.sscd?.pairs },
];

/** Compact "where" panel at the top of the Detection tab. */
function renderWhereCard() {
  const box = $('#whereCard');
  if (!box) return;
  box.innerHTML = '';
  const v = state.visuals;
  box.append(el('div', { class: 'where-head' }, el('h3', {}, 'Where the similarity comes from'), el('button', { type: 'button', class: 'linklike', onclick: () => $('#whereBtn').click() }, 'More views →')));
  const lenses = v ? WHERE_LENSES.filter((l) => l.need(v)) : [];
  if (!lenses.length) {
    box.append(
      state.scanning
        ? loader('visuals', {
            panel: true,
            bar: true,
            t0: state.scanT0,
            label: 'Preparing the marked-up views',
            note: 'Marked-up views of where the similarity comes from appear here when the scan finishes.',
          })
        : el('p', { class: 'muted small' }, 'Turn on the DINOv2 or SSCD models in Settings to see where the similarity comes from.'),
    );
    return;
  }
  const ids = lenses.map((l) => l.id);
  if (!state.whereChosen || !ids.includes(state.whereLens)) state.whereLens = defaultLens(v, ids) || ids[0];
  const lens = lenses.find((l) => l.id === state.whereLens);
  box.append(
    el(
      'div',
      { class: 'lens-pills' },
      lenses.map((l) =>
        el(
          'button',
          {
            type: 'button',
            class: 'pill',
            'aria-pressed': String(l.id === state.whereLens),
            onclick: () => {
              state.whereLens = l.id;
              state.whereChosen = true;
              renderWhereCard();
            },
          },
          l.label,
        ),
      ),
    ),
  );
  const width = Math.max(300, box.clientWidth - 36);
  box.append(...safeView(lens.id, width));
}

/** A marked-up view and its note; if drawing fails, say so instead of breaking the page. */
function safeView(id, width) {
  try {
    return [annotatedView(id, width), el('p', { class: 'where-note' }, annotationNote(id))];
  } catch (err) {
    console.error(`view ${id} failed`, err);
    return [el('p', { class: 'muted small' }, `This view could not be drawn for these images (${err.message || err}). The other views and all test results are unaffected.`)];
  }
}

function drawHeatPair(left, right, toV, strength, maxCssWidth) {
  const p = pairCanvas(maxCssWidth);
  heatOverlay(p.g, left, p.ra, toV.a || toV, strength);
  heatOverlay(p.g, right, p.rb, toV.b || toV, strength);
  return p.canvas;
}

function canvasFrom(img) {
  const c = el('canvas', { width: img.w, height: img.h });
  c.getContext('2d').putImageData(new ImageData(img.data, img.w, img.h), 0, 0);
  return c;
}

let blinkTimer = null;

function renderVisual() {
  clearInterval(blinkTimer);
  const viewer = $('#viewer');
  const caption = $('#viewerCaption');
  viewer.innerHTML = '';
  caption.textContent = '';
  const v = state.visuals;
  const { a, b } = state.slots;
  renderLensExplain();
  if (state.view === 'side' || !v) {
    if (!a || !b) return;
    if (!v && state.scanning) {
      viewer.append(
        loader('visuals', {
          panel: true,
          bar: true,
          t0: state.scanT0,
          label: 'Preparing more views',
          note: 'Keypoint matches, heat maps and marked-up differences appear here when the scan finishes.',
        }),
      );
    }
    viewer.append(
      el(
        'div',
        { class: 'view-pair' },
        el('figure', {}, el('img', { src: a.url, alt: 'Image A' }), el('figcaption', {}, `A · ${a.w}×${a.h}`)),
        el('figure', {}, el('img', { src: b.url, alt: 'Image B' }), el('figcaption', {}, `B · ${b.w}×${b.h}`)),
      ),
    );
    caption.textContent = v || state.scanning ? '' : 'More views appear when the scan finishes.';
    return;
  }
  const displayWidth = Math.min(viewer.clientWidth - 32, Math.max(v.pairA.w, 640));
  const fullWidth = viewer.clientWidth - 32;
  const colorbar = (lo, hi) => el('div', { class: 'colorbar' }, lo, el('i'), hi);
  if (state.view === 'parts') {
    const holder = el('div');
    const draw = () => {
      holder.innerHTML = '';
      holder.append(drawParts(v.dino, fullWidth, !!state.partsHeat, !!state.partsLines));
    };
    draw();
    const toggle = (key, text) =>
      el(
        'label',
        { class: 'ctl-check' },
        el('input', {
          type: 'checkbox',
          checked: !!state[key],
          onchange: (e) => {
            state[key] = e.target.checked;
            draw();
          },
        }),
        text,
      );
    viewer.append(
      el('div', { class: 'lens-view' }, holder, el('div', { class: 'viewer-controls' }, toggle('partsLines', 'connect matches with lines'), toggle('partsHeat', 'show the similarity heat map underneath'))),
    );
    caption.textContent = v.dino.lines.length
      ? `Dots of the same colour mark matching parts: the ${v.dino.lines.length} strongest mutual matches, spread across the image. ${Math.round(v.dino.mutualShare * 100)}% of A’s patches have a mutual match in B with cosine ≥ 0.5 (unrelated photos: under 5%).`
      : 'No patch pairs pass the cosine 0.5 bar: nothing in these images corresponds.';
    return;
  }
  if (state.view === 'heat') {
    viewer.append(el('div', { class: 'lens-view' }, drawHeatPair(v.dino.a, v.dino.b, dinoV, 1, fullWidth), colorbar('no counterpart', 'close match')));
    caption.textContent = 'Colour shows each patch’s best-match cosine similarity in the other image, from 0.35 (no colour) to 0.85 and above (brightest).';
    return;
  }
  if (ANN_LENSES.has(state.view)) {
    const [view, note] = safeView(state.view, fullWidth);
    viewer.append(el('div', { class: 'lens-view' }, view));
    caption.textContent = note ? note.textContent : '';
    return;
  }
  if (state.view === 'swipe') {
    const base = canvasFrom(v.pairB);
    const top = canvasFrom(v.pairA);
    base.style.width = `${displayWidth}px`;
    top.style.width = `${displayWidth}px`;
    const topWrap = el('div', { class: 'swipe-top', style: 'width:50%' }, top);
    const handle = el('div', { class: 'swipe-handle', style: 'left:50%' });
    const box = el('div', { class: 'swipe', style: `width:${displayWidth}px` }, base, topWrap, handle, el('span', { class: 'swipe-label', style: 'left:8px' }, 'A'), el('span', { class: 'swipe-label', style: 'right:8px' }, 'B'));
    const move = (e) => {
      const rect = box.getBoundingClientRect();
      const f = Math.max(0, Math.min(1, (e.clientX - rect.left) / rect.width));
      topWrap.style.width = `${f * 100}%`;
      handle.style.left = `${f * 100}%`;
    };
    box.addEventListener('pointerdown', (e) => {
      box.setPointerCapture(e.pointerId);
      move(e);
      box.onpointermove = move;
    });
    box.addEventListener('pointerup', () => (box.onpointermove = null));
    viewer.append(box);
    caption.textContent = '';
  } else if (state.view === 'blink') {
    const ca = canvasFrom(v.pairA);
    const cb = canvasFrom(v.pairB);
    for (const c of [ca, cb]) c.style.width = `${displayWidth}px`;
    const label = el('span', { class: 'swipe-label', style: 'left:8px' }, 'A');
    const box = el('div', { class: 'swipe', style: `width:${displayWidth}px` }, ca, label);
    let showA = true;
    blinkTimer = setInterval(() => {
      showA = !showA;
      box.replaceChild(showA ? ca : cb, showA ? cb : ca);
      label.textContent = showA ? 'A' : 'B';
    }, 700);
    viewer.append(box);
    caption.textContent = '';
  } else if (state.view === 'deltaE' || state.view === 'ssim') {
    const c = canvasFrom(v[state.view]);
    c.style.width = `${displayWidth}px`;
    viewer.append(el('div', {}, c, el('div', { class: 'colorbar' }, 'similar', el('i'), 'different')));
    caption.textContent = '';
  } else if (state.view === 'matches') {
    viewer.append(drawMatches(v.matches, displayWidth));
    caption.textContent = `${v.matches.kind} keypoints: each line joins a feature in A to its match in B. Lines are RANSAC inliers (${v.matches.inliers} total)${v.matches.sane ? ', consistent with a single geometric transform' : ' — but no plausible single transform, so likely coincidental'}.`;
  } else if (state.view === 'aligned') {
    viewer.append(drawAligned(v, displayWidth));
    caption.textContent = `B warped onto A using the ${v.aligned.via} homography. Slide to fade between A and aligned B, or show where they still differ.`;
  }
}

function drawMatches(m, displayWidth) {
  const gap = 24;
  const W = m.a.w + gap + m.b.w;
  const H = Math.max(m.a.h, m.b.h);
  const c = el('canvas', { width: W, height: H });
  const g = c.getContext('2d');
  g.drawImage(state.slots.a.bitmap, 0, 0, m.a.w, m.a.h);
  g.drawImage(state.slots.b.bitmap, m.a.w + gap, 0, m.b.w, m.b.h);
  g.lineWidth = Math.max(1, W / 900);
  for (const [xb, yb, xa, ya] of m.lines) {
    g.strokeStyle = m.sane ? 'rgba(40, 220, 120, 0.75)' : 'rgba(255, 170, 0, 0.75)';
    g.beginPath();
    g.moveTo(xa, ya);
    g.lineTo(m.a.w + gap + xb, yb);
    g.stroke();
    g.fillStyle = '#fff';
    g.fillRect(xa - 1.5, ya - 1.5, 3, 3);
    g.fillRect(m.a.w + gap + xb - 1.5, yb - 1.5, 3, 3);
  }
  c.style.width = `${Math.min(displayWidth * 1.25, W, $('#viewer').clientWidth - 32)}px`;
  return c;
}

function drawAligned(v, displayWidth) {
  const base = canvasFrom(v.pairA);
  const warped = canvasFrom(v.aligned.warped);
  const diff = canvasFrom(v.aligned.diff);
  const out = el('canvas', { width: v.pairA.w, height: v.pairA.h });
  out.style.width = `${displayWidth}px`;
  const g = out.getContext('2d');
  let mode = 'fade';
  let alpha = 0.5;
  const draw = () => {
    g.clearRect(0, 0, out.width, out.height);
    if (mode === 'diff') {
      g.drawImage(diff, 0, 0);
      return;
    }
    g.globalAlpha = 1;
    g.drawImage(base, 0, 0);
    g.globalAlpha = alpha;
    g.drawImage(warped, 0, 0);
    g.globalAlpha = 1;
  };
  draw();
  const slider = el('input', {
    type: 'range',
    min: 0,
    max: 100,
    value: 50,
    oninput: (e) => {
      alpha = e.target.value / 100;
      draw();
    },
  });
  const toggle = el('label', { class: 'ctl-check' }, el('input', { type: 'checkbox', onchange: (e) => ((mode = e.target.checked ? 'diff' : 'fade'), draw()) }), 'show remaining differences');
  return el('div', {}, out, el('div', { class: 'viewer-controls' }, 'A', slider, 'aligned B', toggle));
}

// ------------------------------------------------------------------ objects

const OBJECT_VERDICTS = {
  aligned: { match: 'Unchanged', partial: 'Modified', none: 'Different' },
  appearance: { match: 'Copy', partial: 'Similar', none: 'Different' },
};
const cssVar = (name) => getComputedStyle(document.documentElement).getPropertyValue(name).trim();
const objectColor = (verdict) => cssVar(verdict === 'match' ? '--none' : verdict === 'partial' ? '--partial' : '--match');

function objectMetricsText(m, mode) {
  const parts = [];
  if (mode === 'aligned' && m.changed !== undefined) parts.push(`${Math.round(m.changed * 100)}% of pixels changed`);
  if (m.deltaE !== undefined) parts.push(`ΔE ${m.deltaE.toFixed(1)}`);
  if (m.ssim !== undefined && Number.isFinite(m.ssim)) parts.push(`SSIM ${m.ssim.toFixed(2)}`);
  if (m.colour !== undefined) parts.push(`colour ${m.colour.toFixed(2)}`);
  if (m.sscd !== undefined) parts.push(`SSCD ${m.sscd.toFixed(2)}`);
  return parts.join(' · ');
}

/** Boxes to draw on one image: paired objects (numbered) and unpaired ones. */
function objectMarks(o, side) {
  const dets = o[side].dets;
  const marks = o.pairs.map((p) => ({
    box: dets[side === 'a' ? p.i : p.j].box,
    color: objectColor(p.verdict),
    tag: `${p.n} ${dets[side === 'a' ? p.i : p.j].label}`,
    n: p.n,
  }));
  for (const x of side === 'a' ? o.onlyA : o.onlyB) {
    const d = dets[side === 'a' ? x.i : x.j];
    marks.push({ box: d.box, color: cssVar('--match'), tag: `${d.label} · only in ${side.toUpperCase()}`, dashed: true, n: `${side}${side === 'a' ? x.i : x.j}` });
  }
  return marks;
}

function drawObjectCanvas(canvas, slot, marks) {
  const bm = state.slots[slot].bitmap;
  const s = Math.min(1, 1100 / Math.max(bm.width, bm.height));
  canvas.width = Math.round(bm.width * s);
  canvas.height = Math.round(bm.height * s);
  const g = canvas.getContext('2d');
  g.drawImage(bm, 0, 0, canvas.width, canvas.height);
  if (state.objectView === 'original') return;
  const lw = Math.max(2, canvas.width / 320);
  const font = Math.max(11, Math.round(canvas.width / 48));
  g.font = `600 ${font}px ${cssVar('--sans') || 'sans-serif'}`;
  g.textBaseline = 'top';
  const hi = state.objectHighlight;
  for (const m of marks) {
    const [x1, y1, x2, y2] = m.box.map((v) => v * s);
    g.globalAlpha = hi === null || hi === m.n ? 1 : 0.25;
    g.lineWidth = hi === m.n ? lw * 2 : lw;
    g.strokeStyle = m.color;
    g.setLineDash(m.dashed ? [lw * 3, lw * 2] : []);
    g.strokeRect(x1, y1, x2 - x1, y2 - y1);
    g.setLineDash([]);
    const tw = g.measureText(m.tag).width + font * 0.6;
    const ty = y1 - font * 1.35 >= 0 ? y1 - font * 1.35 : y1;
    g.fillStyle = m.color;
    g.fillRect(x1, ty, tw, font * 1.35);
    g.fillStyle = '#fff';
    g.fillText(m.tag, x1 + font * 0.3, ty + font * 0.18);
  }
  g.globalAlpha = 1;
}

function renderObjects() {
  const panel = $('#objectsPanel');
  if (!panel) return;
  panel.innerHTML = '';
  const o = state.visuals?.objects;
  const r = state.results.objects;
  if (!o) {
    if (state.scanning && !isDisabled('dfine') && r?.verdict !== 'skipped' && r?.verdict !== 'error') {
      panel.append(
        loader('objects', {
          panel: true,
          bar: true,
          t0: state.scanT0,
          label: 'Detecting objects',
          note: 'Objects are detected near the end of the scan; they appear here, paired up and compared, when it finishes.',
        }),
      );
      return;
    }
    let msg = 'Objects are detected near the end of the scan; they appear here when it finishes.';
    if (r?.verdict === 'skipped' || isDisabled('dfine')) msg = 'The object detector is turned off in Settings.';
    else if (r?.verdict === 'error') msg = `Object detection failed: ${r.note}`;
    else if (!state.scanning && !Object.keys(state.results).length) msg = 'Scan two images to see their objects.';
    panel.append(el('p', { class: 'muted objects-empty' }, msg));
    return;
  }
  const words = OBJECT_VERDICTS[o.mode] || OBJECT_VERDICTS.aligned;
  const count = (v) => o.pairs.filter((p) => p.verdict === v).length;
  const how =
    o.mode === 'aligned'
      ? 'The images were aligned with keypoints, so objects are paired by position and compared pixel by pixel.'
      : o.mode === 'appearance'
        ? 'The images could not be aligned, so objects are paired by kind and appearance.'
        : '';
  panel.append(
    el(
      'div',
      { class: 'objects-head' },
      el(
        'p',
        {},
        el('b', {}, `${o.a.dets.length} objects in A, ${o.b.dets.length} in B. `),
        `${o.pairs.length} paired: ${count('match')} ${words.match.toLowerCase()}, ${count('partial')} ${words.partial.toLowerCase()}, ${count('none')} ${words.none.toLowerCase()}; ${o.onlyA.length} only in A, ${o.onlyB.length} only in B. `,
        el('span', { class: 'muted' }, how),
      ),
      el(
        'div',
        { class: 'objects-controls' },
        ['markup', 'original'].map((v) =>
          el(
            'button',
            {
              type: 'button',
              class: 'pill',
              'aria-pressed': String(state.objectView === v),
              onclick: () => {
                state.objectView = v;
                renderObjects();
              },
            },
            v === 'markup' ? 'Marked up' : 'Original',
          ),
        ),
        el('span', { class: 'objects-legend' }, el('i', { class: 'dot none' }), words.match, el('i', { class: 'dot partial' }), words.partial, el('i', { class: 'dot match' }), `${words.none} / unpaired`),
      ),
    ),
  );
  const canvases = {};
  const figures = ['a', 'b'].map((side) => {
    canvases[side] = el('canvas', { class: 'objects-canvas' });
    return el('figure', {}, canvases[side], el('figcaption', {}, `Image ${side.toUpperCase()} · ${o[side].dets.length} objects`));
  });
  panel.append(el('div', { class: 'objects-images' }, figures));
  const redraw = () => {
    drawObjectCanvas(canvases.a, 'a', objectMarks(o, 'a'));
    drawObjectCanvas(canvases.b, 'b', objectMarks(o, 'b'));
  };
  redraw();
  const hover = (n) => {
    state.objectHighlight = n;
    redraw();
  };

  const list = el('div', { class: 'objects-list' });
  for (const p of o.pairs) {
    list.append(
      el(
        'div',
        { class: 'obj-row', 'data-verdict': p.verdict, onmouseenter: () => hover(p.n), onmouseleave: () => hover(null) },
        el('span', { class: 'obj-n' }, p.n),
        el(
          'div',
          { class: 'obj-thumbs' },
          el('figure', {}, canvasFrom(p.thumbs.a), el('figcaption', {}, 'A')),
          el('figure', {}, canvasFrom(p.thumbs.b), el('figcaption', {}, o.mode === 'aligned' ? 'B, aligned' : 'B')),
          p.thumbs.diff ? el('figure', {}, canvasFrom(p.thumbs.diff), el('figcaption', {}, 'difference')) : null,
        ),
        el('div', { class: 'obj-info' }, el('strong', {}, p.label), el('div', { class: 'obj-metrics' }, objectMetricsText(p.metrics, o.mode))),
        el('span', { class: 'obj-verdict' }, words[p.verdict]),
      ),
    );
  }
  for (const side of ['a', 'b']) {
    for (const x of side === 'a' ? o.onlyA : o.onlyB) {
      const d = o[side].dets[side === 'a' ? x.i : x.j];
      const other = side === 'a' ? 'B' : 'A';
      let note = `Not found in ${other}.`;
      if (x.changed !== null && x.changed !== undefined) {
        note += x.changed > 0.3 ? ` That spot changed in ${other} (${Math.round(x.changed * 100)}% of pixels).` : ` That spot looks unchanged in ${other}, so the detector may simply have missed it.`;
      }
      const n = `${side}${side === 'a' ? x.i : x.j}`;
      list.append(
        el(
          'div',
          { class: 'obj-row', 'data-verdict': 'none', onmouseenter: () => hover(n), onmouseleave: () => hover(null) },
          el('span', { class: 'obj-n' }, '–'),
          el('div', { class: 'obj-thumbs' }, x.thumb ? el('figure', {}, canvasFrom(x.thumb), el('figcaption', {}, side.toUpperCase())) : null),
          el('div', { class: 'obj-info' }, el('strong', {}, d.label), el('div', { class: 'obj-metrics' }, `${note} Detector confidence ${d.score.toFixed(2)}.`)),
          el('span', { class: 'obj-verdict' }, `Only in ${side.toUpperCase()}`),
        ),
      );
    }
  }
  panel.append(list);
}

// ------------------------------------------------------------------ details table

const SIGNATURES = [
  [[0x89, 0x50, 0x4e, 0x47], 'PNG'],
  [[0xff, 0xd8, 0xff], 'JPEG'],
  [[0x47, 0x49, 0x46, 0x38], 'GIF'],
  [[0x42, 0x4d], 'BMP'],
  [[0x00, 0x00, 0x01, 0x00], 'ICO'],
  [[0x49, 0x49, 0x2a, 0x00], 'TIFF'],
  [[0x4d, 0x4d, 0x00, 0x2a], 'TIFF'],
];

async function sniffFormat(file) {
  const head = new Uint8Array(await file.slice(0, 32).arrayBuffer());
  for (const [sig, name] of SIGNATURES) if (sig.every((b, i) => head[i] === b)) return name;
  const ascii = String.fromCharCode(...head);
  if (ascii.startsWith('RIFF') && ascii.slice(8, 12) === 'WEBP') return 'WebP';
  if (ascii.slice(4, 8) === 'ftyp') {
    const brand = ascii.slice(8, 12);
    if (brand.startsWith('avi')) return 'AVIF';
    if (/hei|hev|mif/.test(brand)) return 'HEIC/HEIF';
    return `ISO-BMFF (${brand})`;
  }
  if (/<svg|<\?xml/i.test(ascii)) return 'SVG';
  return 'unknown';
}

async function renderDetails() {
  const table = $('#detailsTable');
  const { a, b } = state.slots;
  if (!a || !b) return;
  const info = state.info || { a: {}, b: {} };
  const [fa, fb] = await Promise.all([sniffFormat(a.file), sniffFormat(b.file)]);
  const rows = [
    ['section', 'File'],
    ['Name', a.file.name, b.file.name],
    ['Format (from file signature)', `${fa}${a.file.type ? ` · ${a.file.type}` : ''}`, `${fb}${b.file.type ? ` · ${b.file.type}` : ''}`],
    ['File size', `${formatBytes(a.file.size)} (${a.file.size.toLocaleString()} bytes)`, `${formatBytes(b.file.size)} (${b.file.size.toLocaleString()} bytes)`],
    ['Last modified', fmtTime(a.file.lastModified), fmtTime(b.file.lastModified)],
    ['MD5', info.a.md5, info.b.md5],
    ['SHA-256', info.a.sha256, info.b.sha256],
    ['section', 'Pixels'],
    ['Dimensions', info.a.dimensions || `${a.w} × ${a.h}`, info.b.dimensions || `${b.w} × ${b.h}`],
    ['Aspect ratio', ratio(a.w, a.h), ratio(b.w, b.h)],
    ['Megapixels', info.a.megapixels, info.b.megapixels],
    ['Transparency', info.a.transparency, info.b.transparency],
    ['section', 'Perceptual hashes (hex)'],
    ['PDQ', info.a.pdq, info.b.pdq],
    ['pHash', info.a.phash, info.b.phash],
    ['dHash', info.a.dhash, info.b.dhash],
    ['aHash', info.a.ahash, info.b.ahash],
    ['wHash', info.a.whash, info.b.whash],
    ['Blockhash', info.a.blockhash, info.b.blockhash],
  ];
  const ea = info.a.exif || {};
  const eb = info.b.exif || {};
  const keys = [...new Set([...Object.keys(ea), ...Object.keys(eb)])];
  rows.push(['section', 'Embedded metadata (EXIF / XMP)']);
  if (!keys.length) rows.push(['Metadata', state.info ? 'none found' : '…', state.info ? 'none found' : '…']);
  for (const k of keys) rows.push([k, ea[k] ?? '—', eb[k] ?? '—']);

  table.innerHTML = '';
  table.append(el('thead', {}, el('tr', {}, el('th', {}, 'Property'), el('th', {}, 'Image A'), el('th', {}, 'Image B'))));
  const body = el('tbody');
  for (const row of rows) {
    if (row[0] === 'section') {
      body.append(el('tr', { class: 'section' }, el('th', { colspan: 3 }, row[1])));
      continue;
    }
    const [k, va, vb] = row;
    const same = va !== undefined && va !== null && va !== '—' && va === vb && !['Name', 'Last modified'].includes(k);
    body.append(el('tr', { class: same ? 'same' : '' }, el('th', {}, k), el('td', {}, va ?? '…'), el('td', {}, vb ?? '…')));
  }
  table.append(body);
}

// ------------------------------------------------------------------ export

function report() {
  const { a, b } = state.slots;
  const results = {};
  for (const e of ENGINES) {
    const r = state.results[e.id];
    if (!r) continue;
    results[e.id] = {
      test: e.name,
      group: GROUPS.find((g) => g.id === e.group).title,
      verdict: r.verdict,
      display: r.display,
      value: typeof r.value === 'number' && Number.isFinite(r.value) ? r.value : r.value === Infinity ? 'Infinity' : null,
      thresholds: e.better === 'equal' ? 'exact equality' : { better: e.better, match: e.match, partial: e.partial ?? null },
      detail: r.detail || undefined,
      note: r.note || undefined,
      milliseconds: r.ms,
    };
  }
  const counted = Object.values(state.results).filter((r) => COUNTED.has(r.verdict));
  const h = headline(state.results);
  return {
    tool: 'Image Similarity Scanner',
    generated: new Date().toISOString(),
    images: {
      A: { name: a.file.name, bytes: a.file.size, width: a.w, height: a.h, md5: state.info?.a?.md5, sha256: state.info?.a?.sha256 },
      B: { name: b.file.name, bytes: b.file.size, width: b.w, height: b.h, md5: state.info?.b?.md5, sha256: state.info?.b?.sha256 },
    },
    summary: {
      headline: h.title,
      explanation: h.sub,
      flagged: counted.filter((r) => FLAGGED.has(r.verdict)).length,
      partial: counted.filter((r) => r.verdict === 'partial').length,
      applicable: counted.length,
    },
    results,
    modelsTurnedOff: state.settings.disabled,
  };
}

function exportJson() {
  const blob = new Blob([JSON.stringify(report(), null, 2)], { type: 'application/json' });
  const url = URL.createObjectURL(blob);
  const a = el('a', { href: url, download: 'image-similarity-report.json' });
  document.body.append(a);
  a.click();
  a.remove();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}

async function copySummary() {
  const r = report();
  const lines = [
    `Image Similarity Scanner — ${r.images.A.name} vs ${r.images.B.name}`,
    `${r.summary.headline}: ${r.summary.explanation}`,
    `${r.summary.flagged} of ${r.summary.applicable} tests flagged similarity (${r.summary.partial} partial).`,
    '',
  ];
  for (const g of GROUPS) {
    lines.push(`${g.title}:`);
    for (const e of ENGINES.filter((x) => x.group === g.id)) {
      const res = state.results[e.id];
      if (res) lines.push(`  ${e.name}: ${res.display} (${VERDICT_LABEL[res.verdict] || res.verdict})`);
    }
  }
  try {
    await navigator.clipboard.writeText(lines.join('\n'));
    toast('Summary copied to the clipboard.');
  } catch {
    toast('Could not access the clipboard.');
  }
}

// ------------------------------------------------------------------ settings

function renderSettings() {
  const list = $('#modelList');
  list.innerHTML = '';
  for (const key of MODEL_ORDER) {
    const info = MODEL_INFO[key];
    const sizeCell = el('span', { class: 'model-size' }, formatBytes(MODELS[key].bytes));
    const input = el('input', {
      type: 'checkbox',
      checked: !isDisabled(key),
      onchange: (e) => {
        const off = new Set(state.settings.disabled);
        if (e.target.checked) off.delete(key);
        else off.add(key);
        state.settings.disabled = [...off];
        saveSettings();
        refreshDownloadNote();
        for (const eng of ENGINES) if (eng.model === key) renderEngine(eng.id);
        for (const m of SPOTLIGHT) updateSpotlight(m.id);
      },
    });
    list.append(el('label', { class: 'model-row' }, input, el('span', {}, el('strong', {}, info.name), el('small', {}, info.note)), sizeCell));
    store.isCached(key).then((cached) => {
      if (cached) sizeCell.append(el('span', { class: 'cached' }, 'downloaded'));
    });
  }
}

async function refreshDownloadNote() {
  const note = $('#downloadNote');
  let bytes = 0;
  for (const key of MODEL_ORDER) {
    if (isDisabled(key) || state.modelsInWorker.has(key)) continue;
    if (!(await store.isCached(key))) bytes += MODELS[key].bytes;
  }
  note.hidden = bytes === 0;
  note.textContent = `The first scan downloads ${formatBytes(bytes)} of neural network models (cached afterwards). Large models can be turned off in Settings.`;
}

// ------------------------------------------------------------------ transform lab

const CONTROLS = [
  { section: 'Geometry' },
  { key: 'crop', label: 'Keep (crop)', min: 10, max: 100, unit: '%' },
  { key: 'cropX', label: 'Crop X position', min: 0, max: 100, unit: '%' },
  { key: 'cropY', label: 'Crop Y position', min: 0, max: 100, unit: '%' },
  { key: 'scale', label: 'Scale', min: 10, max: 200, unit: '%' },
  { key: 'rotate', label: 'Rotate', min: -180, max: 180, unit: '°' },
  { key: 'flip', label: 'Mirror horizontally', type: 'check' },
  { section: 'Colour' },
  { key: 'brightness', label: 'Brightness', min: -100, max: 100, unit: '' },
  { key: 'contrast', label: 'Contrast', min: -100, max: 100, unit: '' },
  { key: 'saturation', label: 'Saturation', min: 0, max: 200, unit: '%' },
  { key: 'hue', label: 'Hue shift', min: -180, max: 180, unit: '°' },
  { key: 'grayscale', label: 'Black & white', type: 'check' },
  { section: 'Quality' },
  { key: 'blur', label: 'Blur', min: 0, max: 12, step: 0.5, unit: 'px' },
  { key: 'noise', label: 'Noise', min: 0, max: 60, unit: '' },
  { key: 'format', label: 'Save as', type: 'select', options: [['jpeg', 'JPEG (lossy)'], ['png', 'PNG (lossless)']] },
  { key: 'quality', label: 'JPEG quality', min: 1, max: 100, unit: '' },
  { section: 'Overlay' },
  { key: 'text', label: 'Caption', type: 'text', placeholder: 'e.g. a meme caption or watermark' },
];

const lab = { params: { ...DEFAULTS }, blob: null, url: null, timer: null, inputs: {}, seq: 0 };

function buildLab() {
  const box = $('#labControls');
  const presets = $('#labPresets');
  for (const p of PRESETS) {
    presets.append(el('button', { type: 'button', class: 'pill', onclick: () => setLabParams({ ...DEFAULTS, ...p.params }) }, p.label));
  }
  for (const c of CONTROLS) {
    if (c.section) {
      box.append(el('div', { class: 'ctl-section' }, c.section));
      continue;
    }
    let input;
    if (c.type === 'check') {
      input = el('input', { type: 'checkbox', onchange: () => updateLabParam(c.key, input.checked) });
      box.append(el('label', { class: 'ctl-check' }, input, c.label));
    } else if (c.type === 'select') {
      input = el('select', { onchange: () => updateLabParam(c.key, input.value) }, c.options.map(([v, t]) => el('option', { value: v }, t)));
      box.append(el('label', { class: 'ctl' }, el('span', {}, c.label), input));
    } else if (c.type === 'text') {
      input = el('input', { type: 'text', placeholder: c.placeholder, oninput: () => updateLabParam(c.key, input.value) });
      box.append(el('label', { class: 'ctl' }, el('span', {}, c.label), input));
    } else {
      const out = el('output');
      input = el('input', {
        type: 'range',
        min: c.min,
        max: c.max,
        step: c.step || 1,
        oninput: () => {
          out.textContent = `${input.value}${c.unit}`;
          updateLabParam(c.key, Number(input.value));
        },
      });
      input._out = out;
      input._unit = c.unit;
      box.append(el('label', { class: 'ctl' }, el('span', {}, c.label), input, out));
    }
    lab.inputs[c.key] = input;
  }
  $('#labReset').addEventListener('click', () => setLabParams({ ...DEFAULTS }));
  $('#labUse').addEventListener('click', async () => {
    if (!lab.blob) return;
    const base = (state.slots.a.file.name || 'image').replace(/\.[^.]+$/, '');
    const ext = lab.params.format === 'png' ? 'png' : 'jpg';
    const file = new File([lab.blob], `${base}-edited.${ext}`, { type: lab.blob.type });
    $('#labDialog').close();
    await setSlot('b', file);
    startScan();
  });
}

function setLabParams(params) {
  lab.params = { ...params };
  for (const [k, input] of Object.entries(lab.inputs)) {
    const v = lab.params[k];
    if (input.type === 'checkbox') input.checked = !!v;
    else input.value = v;
    if (input._out) input._out.textContent = `${v}${input._unit}`;
  }
  refreshLab();
}

function updateLabParam(key, value) {
  lab.params[key] = value;
  clearTimeout(lab.timer);
  lab.timer = setTimeout(refreshLab, 120);
}

async function refreshLab() {
  const a = state.slots.a;
  if (!a) return;
  const seq = ++lab.seq;
  const blob = await applyTransform(a.bitmap, lab.params);
  if (seq !== lab.seq) return;
  lab.blob = blob;
  if (lab.url) URL.revokeObjectURL(lab.url);
  lab.url = URL.createObjectURL(blob);
  $('#labPreview').src = lab.url;
  $('#labCaption').textContent = `${describe(lab.params)} · ${formatBytes(blob.size)}`;
}

function openLab() {
  if (!state.slots.a) {
    toast('Add image A first.');
    return;
  }
  $('#labDialog').showModal();
  setLabParams(lab.params);
}

// ------------------------------------------------------------------ samples

const SAMPLES = [
  { label: 'Identical file', a: 'astronaut', b: 'astronaut' },
  { label: 'Heavy JPEG compression', a: 'coffee', params: { quality: 12 } },
  { label: 'Crop + filter + caption', a: 'chelsea', params: { crop: 72, cropX: 30, cropY: 35, saturation: 145, hue: -15, contrast: 15, text: 'NOT A COPY', quality: 82 } },
  { label: 'Lossless crop', a: 'rocket', params: { crop: 45, cropX: 55, cropY: 30, format: 'png' } },
  { label: 'Mirrored', a: 'astronaut', params: { flip: true } },
  { label: 'Unrelated images', a: 'coffee', b: 'chelsea' },
  { label: 'Spot the difference (drawing)', a: 'farm-a.png', b: 'farm-b.png' },
];

async function fetchSample(name) {
  const file = name.includes('.') ? name : `${name}.jpg`;
  const res = await fetch(new URL(`assets/samples/${file}`, ROOT));
  if (!res.ok) throw new Error(`HTTP ${res.status} for ${file}`);
  const blob = await res.blob();
  return new File([blob], file, { type: file.endsWith('.png') ? 'image/png' : 'image/jpeg', lastModified: 0 });
}

async function loadSample(s) {
  try {
    const fa = await fetchSample(s.a);
    await setSlot('a', fa);
    if (s.params) {
      const blob = await applyTransform(state.slots.a.bitmap, s.params);
      const ext = s.params.format === 'png' ? 'png' : 'jpg';
      await setSlot('b', new File([blob], `${s.a}-edited.${ext}`, { type: blob.type, lastModified: 0 }));
    } else {
      await setSlot('b', s.b === s.a ? fa : await fetchSample(s.b));
    }
    startScan();
  } catch (err) {
    console.error(err);
    toast('Could not load the example images.');
  }
}

// ------------------------------------------------------------------ copyright cases

const OUTCOME = {
  liable: { short: 'Infringement', long: 'Holding: infringement' },
  fairuse: { short: 'Fair use', long: 'Holding: no infringement (fair use)' },
  nosim: { short: 'No infringement', long: 'Holding: no infringement' },
};
// a case may word its own holding when the result is mixed or not final
const outcomeShort = (c) => c.outcome || OUTCOME[c.group].short;
const outcomeLong = (c) => c.holding || OUTCOME[c.group].long;

const caseImageUrl = (path) => new URL(`assets/cases/${path}`, ROOT).href;

async function loadCaseManifest() {
  try {
    const res = await fetch(new URL('assets/cases/manifest.json', ROOT));
    if (res.ok) {
      for (const [id, entry] of Object.entries(await res.json())) {
        if (!id.startsWith('_') && entry.a && entry.b) state.caseImages[id] = entry;
      }
    }
  } catch {
    // no case images published
  }
  renderCases();
}

function renderCases() {
  const root = $('#caseGroups');
  root.innerHTML = '';
  // only cases whose works are published here; the rest stay listed in cases.js
  const ready = CASES.filter((c) => state.caseImages[c.id]);
  for (const g of CASE_GROUPS) {
    const cases = ready.filter((c) => c.group === g.id);
    if (!cases.length) continue;
    root.append(
      el(
        'section',
        { class: 'case-group' },
        el('div', { class: 'case-group-head' }, el('h3', {}, g.title), el('span', { class: 'muted small' }, `${cases.length} case${cases.length === 1 ? '' : 's'}`)),
        el('p', { class: 'case-group-blurb' }, g.blurb),
        el('div', { class: 'case-grid' }, cases.map(caseCard)),
      ),
    );
  }
  $('#cases').hidden = !ready.length;
  $('.cases-pointer').hidden = !ready.length;
  for (const a of document.querySelectorAll('.topbar a[href="#cases"]')) a.hidden = !ready.length;
  const link = $('#casesLink');
  link.replaceChildren(...(ready.length === 1 ? [el('i', {}, ready[0].name), ' ↓'] : [`all ${ready.length} cases ↓`]));
  for (const pick of document.querySelectorAll('.case-pick')) fillCasePicker(pick, ready);
}

/** A drop-down of the cases with images, grouped by holding. */
function fillCasePicker(select, ready) {
  select.replaceChildren(el('option', { value: '' }, 'Choose a case…'));
  for (const g of CASE_GROUPS) {
    const cases = ready.filter((c) => c.group === g.id);
    if (!cases.length) continue;
    select.append(el('optgroup', { label: g.title }, cases.map((c) => el('option', { value: c.id }, `${c.name} — ${outcomeShort(c)}`))));
  }
  select.value = state.case?.id || '';
}

function pickCase(id) {
  const c = CASES.find((x) => x.id === id);
  if (c) loadCase(c);
}

// ------------------------------------------------------------------ all cases at once

/** Resolves when the current scan finishes (or fails). */
function scanFinished() {
  return new Promise((resolve) => state.scanWaiters.push(resolve));
}

function settleScanWaiters() {
  const waiters = state.scanWaiters.splice(0);
  for (const w of waiters) w();
}

const levelLabel = (id) => LEVELS.find((l) => l.id === id)?.label || '—';

/** One row of the all-cases table, from the finished scan on the page. */
function caseRow(c) {
  const R = state.results;
  const counted = Object.values(R).filter((r) => COUNTED.has(r.verdict));
  const num = (id) => (typeof R[id]?.value === 'number' && COUNTED.has(R[id].verdict) ? R[id] : null);
  const h = headline(R);
  return {
    c,
    title: h.title,
    tone: h.tone,
    level: similarityLevel(R, state.visuals),
    flagged: counted.filter((r) => FLAGGED.has(r.verdict)).length,
    total: counted.length,
    sscd: num('sscd'),
    dino: num('dinoParts'),
    dreamsim: num('dreamsim'),
  };
}

async function runAllCases() {
  const btn = $('#runAllCases');
  const status = $('#runAllStatus');
  if (state.batch) {
    state.batch.stop = true;
    status.textContent = 'Stopping after this case…';
    return;
  }
  const ready = CASES.filter((c) => state.caseImages[c.id]);
  state.batch = { stop: false, rows: [], queue: ready, current: null, t0: 0 };
  btn.textContent = 'Stop';
  const bar = $('#runAllBar');
  setBar(bar, 0);
  bar.hidden = false;
  for (const [i, c] of ready.entries()) {
    if (state.batch.stop) break;
    status.textContent = `Scanning ${i + 1} of ${ready.length}: ${c.name}…`;
    state.batch.current = c;
    state.batch.t0 = performance.now();
    renderCaseTable();
    const done = scanFinished();
    if (!(await loadCase(c))) {
      settleScanWaiters();
      continue;
    }
    await done;
    state.batch.rows.push(caseRow(c));
  }
  const n = state.batch.rows.length;
  state.lastBatch = state.batch.rows;
  state.batch = null;
  bar.hidden = true;
  renderCaseTable();
  btn.textContent = 'Run all cases again';
  status.textContent = n ? `Finished ${n} case${n === 1 ? '' : 's'}. Click a row to see its full results.` : '';
}

function renderCaseTable() {
  const box = $('#caseTable');
  const batch = state.batch;
  const rows = batch?.rows || state.lastBatch || [];
  box.hidden = !batch && !rows.length;
  if (box.hidden) return;
  // during a run, the cases still to come are listed too: the one being scanned, then the rest
  const scanned = new Set(rows.map((r) => r.c.id));
  const toCome = batch ? batch.queue.filter((c) => !scanned.has(c.id)) : [];
  // opening a case mid-run would replace the scan the run is waiting for
  const open = (c) => !state.batch && loadCase(c);
  const fmt = (r, f) => (r ? f(r.value) : '—');
  // infringement or not; the line under it names fair use or explains a mixed, unfinished result
  const court = (c) => {
    const detail = c.holding ? c.holding.replace(/^Holding: /, '') : c.group === 'fairuse' ? 'fair use' : null;
    return [
      el('span', { class: 'case-outcome', 'data-group': c.group, 'data-final': String(c.final !== false) }, c.final === false ? outcomeShort(c) : c.group === 'liable' ? 'Infringement' : 'No infringement'),
      detail ? el('div', { class: 'muted small' }, detail) : null,
    ];
  };
  box.replaceChildren(
    el('h3', {}, 'All cases at a glance'),
    el(
      'p',
      { class: 'muted small' },
      'Each pair scanned in turn with every test, beside the court’s holding. Expect the two to part ways: computer similarity is at most evidence of copying, while infringement also turns on what was protectable, how much was taken and whether the use was fair.',
    ),
    el(
      'div',
      { class: 'case-table-wrap' },
      el(
        'table',
        { class: 'case-table' },
        el(
          'thead',
          {},
          el('tr', {}, ['Case', 'Court', 'Scanner’s verdict', 'Similarity level', 'Tests flagged', 'SSCD copy score', 'DINOv2 shared parts', 'DreamSim distance'].map((t) => el('th', {}, t))),
        ),
        el(
          'tbody',
          {},
          rows.map((r) =>
            el(
              'tr',
              { tabindex: '0', title: 'Show this case’s full results', onclick: () => open(r.c), onkeydown: (e) => e.key === 'Enter' && open(r.c) },
              el('td', {}, el('i', {}, r.c.name), el('div', { class: 'muted small' }, r.c.cite)),
              el('td', {}, court(r.c)),
              el('td', { 'data-tone': r.tone }, r.title),
              el('td', {}, levelLabel(r.level?.id)),
              el('td', { class: 'num' }, `${r.flagged} / ${r.total}`),
              el('td', { class: 'num' }, fmt(r.sscd, (v) => v.toFixed(3))),
              el('td', { class: 'num' }, fmt(r.dino, (v) => `${Math.round(v * 100)}%`)),
              el('td', { class: 'num' }, fmt(r.dreamsim, (v) => v.toFixed(3))),
            ),
          ),
          toCome.map((c) => {
            const now = c === batch.current;
            return el(
              'tr',
              { class: now ? 'case-row-now' : 'case-row-waiting' },
              el('td', {}, el('i', {}, c.name), el('div', { class: 'muted small' }, c.cite)),
              el('td', {}, court(c)),
              el(
                'td',
                { colspan: '6' },
                now
                  ? loader('scan', { text: 'Fetching the works…', bar: true, t0: batch.t0, label: `Scanning ${c.name}` })
                  : el('span', { class: 'muted small' }, 'Waiting…'),
              ),
            );
          }),
        ),
      ),
    ),
  );
}

function caseCard(c) {
  const imgs = state.caseImages[c.id];
  const thumb = (side) =>
    imgs
      ? el('img', { src: caseImageUrl(imgs[side]), alt: imgs[`${side}Label`] || `Image ${side.toUpperCase()}`, loading: 'lazy' })
      : el('div', { class: 'case-thumb-empty' }, side.toUpperCase());
  return el(
    'article',
    { class: 'case-card' },
    el('div', { class: 'case-thumbs' }, thumb('a'), el('span', { class: 'case-arrow', 'aria-hidden': 'true' }, '→'), thumb('b')),
    el(
      'div',
      { class: 'case-body' },
      el('span', { class: 'case-outcome', 'data-group': c.group, 'data-final': String(c.final !== false) }, outcomeShort(c)),
      el('h4', {}, el('i', {}, c.name)),
      el('div', { class: 'case-cite' }, c.cite),
      el('p', { class: 'case-pairing' }, c.pairing),
      c.note ? el('p', { class: 'case-note' }, c.note) : null,
      c.caution ? el('p', { class: 'case-caution' }, c.caution) : null,
    ),
    el(
      'div',
      { class: 'case-foot' },
      imgs
        ? el('button', { type: 'button', class: 'btn small', onclick: () => loadCase(c) }, 'Compare the works')
        : el('span', { class: 'muted small' }, 'Images not added yet'),
    ),
  );
}

async function fetchCaseFile(c, side) {
  const path = state.caseImages[c.id][side];
  const res = await fetch(caseImageUrl(path));
  if (!res.ok) throw new Error(`HTTP ${res.status} for ${path}`);
  const blob = await res.blob();
  const ext = (path.match(/\.[a-z0-9]+$/i) || ['.jpg'])[0];
  const role = side === 'a' ? 'plaintiff' : 'defendant';
  return new File([blob], `${c.id}-${role}${ext}`, { type: blob.type || 'image/jpeg', lastModified: 0 });
}

async function loadCase(c) {
  try {
    const [fa, fb] = await Promise.all([fetchCaseFile(c, 'a'), fetchCaseFile(c, 'b')]);
    await setSlot('a', fa, { keepCase: true });
    await setSlot('b', fb, { keepCase: true });
    state.case = c;
    for (const pick of document.querySelectorAll('.case-pick')) pick.value = c.id;
    startScan();
    return true;
  } catch (err) {
    console.error(err);
    toast('Could not load the images for this case.');
    return false;
  }
}

function renderCaseBanner() {
  const box = $('#caseBanner');
  const c = state.case;
  box.hidden = !c;
  box.innerHTML = '';
  if (!c) return;
  const imgs = state.caseImages[c.id] || {};
  const labels = imgs.aLabel || imgs.bLabel ? el('p', { class: 'case-banner-labels' }, el('b', {}, 'A '), imgs.aLabel || 'plaintiff’s work', el('b', {}, ' · B '), imgs.bLabel || 'defendant’s work') : null;
  const parts = [
    el(
      'div',
      { class: 'case-banner-top' },
      el('span', { class: 'case-outcome', 'data-group': c.group, 'data-final': String(c.final !== false) }, outcomeLong(c)),
      el('span', { class: 'case-banner-name' }, el('i', {}, c.name), `, ${c.cite}`),
    ),
    el('p', { class: 'case-banner-pairing' }, c.pairing, c.note ? ` — ${c.note}` : ''),
    labels,
    c.caution ? el('p', { class: 'case-caution' }, c.caution) : null,
    el(
      'p',
      { class: 'muted small' },
      'Computer similarity is not legal similarity. Do the tests below line up with the court? They measure resemblance between pixels or learned features; what counts as protectable expression, how much was taken and whether the use was fair are questions they do not answer.',
    ),
  ];
  box.append(...parts.filter(Boolean));
}

// ------------------------------------------------------------------ catalogue

function renderCatalogue() {
  const root = $('#catalogue');
  root.append(
    el('h2', {}, `The ${ENGINES.length} tests`),
    el('p', {}, 'Grouped by what they look at. Each card says what the test measures, how its score is read, and which edits it survives.'),
  );
  for (const g of GROUPS) {
    const grid = el('div', { class: 'cat-grid' });
    for (const e of ENGINES.filter((x) => x.group === g.id)) {
      grid.append(
        el(
          'article',
          { class: 'cat-card' },
          el('h4', {}, e.name),
          el('div', { class: 'by' }, e.by),
          el('p', {}, e.about),
          el('p', { class: 'thr' }, e.thresholds),
          el('div', { class: 'tags' }, e.robust.slice(0, 4).map((t) => el('span', { class: 'tag' }, t)), e.weak.slice(0, 3).map((t) => el('span', { class: 'tag weak' }, t))),
          refList(e.refs),
        ),
      );
    }
    root.append(
      el(
        'details',
        { class: 'cat-group' },
        el('summary', {}, el('span', { class: 'group-chevron', 'aria-hidden': 'true' }), el('h3', {}, g.title), el('span', { class: 'muted small' }, `${grid.children.length} test${grid.children.length === 1 ? '' : 's'}`)),
        el('p', {}, g.blurb),
        grid,
      ),
    );
  }
}

// ------------------------------------------------------------------ misc

function formatBytes(n) {
  if (n === undefined || n === null) return '';
  if (n < 1024) return `${n} B`;
  if (n < 1024 * 1024) return `${(n / 1024).toFixed(1)} KB`;
  return `${(n / 1024 / 1024).toFixed(1)} MB`;
}

function fmtTime(t) {
  return t ? new Date(t).toLocaleString() : '—';
}

function ratio(w, h) {
  const g = (x, y) => (y ? g(y, x % y) : x);
  const d = g(w, h);
  const r = `${w / d}:${h / d}`;
  return r.length <= 9 ? `${r} (${(w / h).toFixed(3)})` : (w / h).toFixed(3);
}

let toastTimer = null;
function toast(message) {
  $('.toast')?.remove();
  const t = el('div', { class: 'toast', role: 'status' }, message);
  document.body.append(t);
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => t.remove(), 4000);
}

function setupTabs() {
  for (const tab of $$('.tab')) {
    tab.addEventListener('click', () => {
      for (const t of $$('.tab')) t.setAttribute('aria-selected', String(t === tab));
      for (const p of $$('.tabpanel')) p.hidden = p.id !== `tab-${tab.dataset.tab}`;
      if (tab.dataset.tab === 'visual') renderVisual();
      if (tab.dataset.tab === 'objects') renderObjects();
    });
  }
}

function init() {
  setupDrops();
  setupTabs();
  buildLab();
  renderCatalogue();
  renderSpectrumGuide();
  renderCases();
  loadCaseManifest();
  for (const s of SAMPLES) $('#samples').append(el('button', { type: 'button', class: 'pill', onclick: () => loadSample(s) }, s.label));
  for (const pick of document.querySelectorAll('.case-pick')) pick.addEventListener('change', () => pickCase(pick.value));
  $('#runAllCases').addEventListener('click', runAllCases);
  $('#scanBtn').addEventListener('click', startScan);
  $('#labBtn').addEventListener('click', openLab);
  $('#labLink').addEventListener('click', (e) => {
    e.stopPropagation();
    openLab();
  });
  $('#whereBtn').addEventListener('click', () => {
    $('.tab[data-tab="visual"]').click();
    $('.tabs').scrollIntoView({ behavior: 'smooth', block: 'start' });
  });
  $('#expandAll').addEventListener('click', () => setAllGroups(true));
  $('#collapseAll').addEventListener('click', () => setAllGroups(false));
  $('#exportBtn').addEventListener('click', exportJson);
  $('#copyBtn').addEventListener('click', copySummary);
  $('#settingsBtn').addEventListener('click', () => {
    renderSettings();
    $('#settingsDialog').showModal();
  });
  $('#clearCacheBtn').addEventListener('click', async () => {
    await store.clear();
    renderSettings();
    refreshDownloadNote();
    toast('Downloaded models deleted from this browser.');
  });
  let resizeTimer = null;
  window.addEventListener('resize', () => {
    clearTimeout(resizeTimer);
    resizeTimer = setTimeout(() => {
      if (!$('#tab-visual').hidden) renderVisual();
      if (!$('#tab-objects').hidden) renderObjects();
      if (!$('#tab-detection').hidden) renderWhereCard();
    }, 200);
  });
  refreshDownloadNote();
  updateButtons();
  if ($('#testCount')) $('#testCount').textContent = String(ENGINES.length);
  if (new URLSearchParams(location.search).has('debug')) {
    // test hook: draw every marked-up view and every tab view, report failures
    const lensCheck = () => {
      const out = {};
      const v = state.visuals;
      for (const l of WHERE_LENSES) {
        if (!v || !l.need(v)) {
          out[l.id] = 'n/a';
          continue;
        }
        try {
          annotatedView(l.id, 640);
          out[l.id] = 'ok';
        } catch (err) {
          out[l.id] = `ERROR ${err.message}`;
        }
      }
      const keep = state.view;
      for (const id of Object.keys(VIEWS)) {
        if (!v || !(id === 'side' || VIEWS[id].need(v))) continue;
        try {
          state.view = id;
          renderVisual();
          out[`tab:${id}`] = $('#viewer').textContent.includes('could not be drawn') ? 'FALLBACK' : 'ok';
        } catch (err) {
          out[`tab:${id}`] = `ERROR ${err.message}`;
        }
      }
      state.view = keep;
      renderVisual();
      return out;
    };
    window.scanner = { state, report, lensCheck, setSlot, startScan };
  }
  // the catalogue is built at runtime, so honour #anchors once it exists
  if (location.hash) document.querySelector(location.hash)?.scrollIntoView();
}

init();
