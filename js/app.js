import { ENGINES, ENGINE_BY_ID, GROUPS, VERDICT_LABEL } from './engines.js';
import { MODELS } from './lib/neural.js';
import { ModelStore } from './lib/modelstore.js';
import { applyTransform, DEFAULTS, PRESETS, describe } from './transform.js';
import { CASES, CASE_GROUPS } from './cases.js';

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
const MODEL_ORDER = ['sscd', 'lpips', 'dino', 'sscdLarge', 'clip'];
const MODEL_INFO = {
  sscd: { name: 'SSCD (ResNet-50)', note: 'The main copy detector. Recommended.' },
  sscdLarge: { name: 'SSCD large (ResNeXt-101)', note: 'Replication-study setting from Somepalli et al.' },
  dino: { name: 'DINOv2 small', note: 'General visual similarity.' },
  clip: { name: 'CLIP ViT-B/32', note: 'Semantic similarity; the largest download.' },
  lpips: { name: 'LPIPS (AlexNet)', note: 'Perceptual distance; tiny.' },
};
const ICONS = {
  identical: '<svg viewBox="0 0 24 24"><path d="M6 9.5h12M6 14.5h12"/></svg>',
  match: '<svg viewBox="0 0 24 24"><path d="M12 6v8m0 4h.01"/></svg>',
  partial: '<svg viewBox="0 0 24 24"><path d="M6 12h12"/></svg>',
  none: '<svg viewBox="0 0 24 24"><path d="M5 12.5l4.5 4.5L19 7.5"/></svg>',
  na: '<svg viewBox="0 0 24 24"><path d="M7 12h10"/></svg>',
  error: '<svg viewBox="0 0 24 24"><path d="M8 8l8 8m0-8-8 8"/></svg>',
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
  case: null, // copyright case whose works are loaded in A and B
  caseImages: {}, // case id -> image paths, from assets/cases/manifest.json
  openEngines: new Set(),
  warmed: false,
};

const isDisabled = (key) => state.settings.disabled.includes(key);

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

function onWorkerMessage(e) {
  const m = e.data;
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
      break;
    case 'running':
      state.results[m.id] = { verdict: 'running' };
      renderEngine(m.id);
      break;
    case 'result':
      state.results[m.id] = m.result;
      renderEngine(m.id);
      updateSpotlight(m.id);
      updateSummary();
      break;
    case 'visuals':
      state.visuals = m.visuals;
      renderViewModes();
      renderVisual();
      break;
    case 'done':
      state.scanning = false;
      state.elapsed = m.ms;
      updateSummary();
      updateButtons();
      break;
    case 'fatal':
      fail(m.message);
      break;
    default:
  }
}

function fail(message) {
  state.scanning = false;
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
  state.openEngines.clear();
  $('#results').hidden = false;
  const s = $('.summary');
  delete s.dataset.tone;
  renderCaseBanner();
  renderGroups();
  renderSpotlight();
  renderViewModes();
  renderVisual();
  renderDetails();
  updateSummary();
  updateButtons();
  requestAnimationFrame(() => $('#results').scrollIntoView({ behavior: 'smooth', block: 'start' }));
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
    root.append(
      el(
        'section',
        { class: 'group card', 'data-group': g.id },
        el('div', { class: 'group-head' }, el('h3', {}, g.title), el('span', { class: 'group-count' })),
        el('p', { class: 'group-blurb' }, g.blurb),
        list,
      ),
    );
    for (const engine of engines) renderEngine(engine.id);
  }
  updateGroupCounts();
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
  icon.innerHTML = ICONS[iconKey] || '';
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
    id: 'sscdLarge',
    title: 'SSCD · Somepalli et al.',
    sub: 'replication threshold 0.5',
    ticks: [0.5, 0.7],
    zones: [['match', 0.5, 1]],
  },
  {
    id: 'clip',
    title: 'CLIP (for contrast)',
    sub: 'semantic, not copy detection',
    ticks: [0.85, 0.95],
    zones: [
      ['partial', 0.85, 0.95],
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
      'SSCD (Pizzi et al., CVPR 2022) is trained to recognise edited copies and is the measure Somepalli et al. used to find training-data replication in Stable Diffusion. Scores are cosine similarities; the bars show the published thresholds. CLIP is shown for contrast: it measures whether images depict similar things.',
    ),
  );
  for (const m of SPOTLIGHT) {
    const track = el('div', { class: 'meter-track' });
    for (const [kind, from, to] of m.zones) {
      track.append(el('div', { class: `meter-zone ${kind}`, style: `left:${from * 100}%;width:${(to - from) * 100}%` }));
    }
    for (const t of m.ticks) track.append(el('div', { class: 'meter-tick', style: `left:${t * 100}%` }, el('span', {}, t.toFixed(2))));
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
    marker.style.left = `${Math.max(0, Math.min(1, r.value)) * 100}%`;
    value.textContent = r.value.toFixed(3);
  } else {
    row.dataset.verdict = 'pending';
    marker.hidden = true;
    let text = 'waiting';
    if (r?.verdict === 'skipped') text = 'turned off';
    else if (r?.verdict === 'error') text = 'error';
    else if (r?.verdict === 'running') text = 'running…';
    else if (engine.model && isDisabled(engine.model)) text = 'turned off';
    value.innerHTML = '';
    value.append(el('span', { class: 'meter-status' }, text));
  }
}

// ------------------------------------------------------------------ summary

function headline(results) {
  const v = (id) => results[id]?.value;
  const verdict = (id) => results[id]?.verdict;
  const has = (id) => typeof v(id) === 'number' && COUNTED.has(verdict(id));
  const sscd = has('sscd') ? v('sscd') : null;
  const geomMatch = ['orb', 'akaze', 'brisk'].some((id) => verdict(id) === 'match');
  if (verdict('sha256') === 'identical') {
    return { tone: 'identical', title: 'Identical files', sub: 'The two files are byte-for-byte the same — a verbatim copy.' };
  }
  if (verdict('pixels') === 'identical') {
    return { tone: 'identical', title: 'Identical pixels', sub: 'The files differ, but every decoded pixel is the same: the same image saved with different metadata or encoding.' };
  }
  if (verdict('crop') === 'identical') {
    return { tone: 'identical', title: 'Verbatim crop', sub: `One image is an unaltered cut-out of the other (${results.crop.detail?.['best placement'] || ''}).` };
  }
  if (sscd !== null && sscd >= 0.75) {
    return { tone: 'match', title: 'Copy detected', sub: `SSCD scores ${sscd.toFixed(3)}, above the 0.75 copy threshold: B looks like an edited copy of A.` };
  }
  if (sscd !== null && sscd >= 0.5) {
    return { tone: 'partial', title: 'Possible partial copy', sub: `SSCD scores ${sscd.toFixed(3)}: above 0.5, where Somepalli et al. found strong visual similarity and likely partial copies, but below the 0.75 copy threshold.` };
  }
  if (sscd === null && (verdict('pdq') === 'match' || verdict('pdqDihedral') === 'match' || geomMatch)) {
    return { tone: 'match', title: 'Likely copy', sub: 'Perceptual hashes or keypoint geometry indicate the same image (SSCD did not run).' };
  }
  if (geomMatch) {
    return { tone: 'partial', title: 'Shared content', sub: 'Keypoint matching finds a region the images have in common, although SSCD does not consider the whole image a copy.' };
  }
  const clip = has('clip') ? v('clip') : null;
  const dino = has('dino') ? v('dino') : null;
  if ((clip !== null && clip >= 0.85) || (dino !== null && dino >= 0.6)) {
    return { tone: 'similar', title: 'Similar subject, not a copy', sub: 'Semantic models (CLIP / DINOv2) see related content, but the copy detectors do not flag it.' };
  }
  return { tone: 'none', title: 'No meaningful similarity', sub: 'The copy detectors, hashes and keypoints all treat these as different images.' };
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
  const pct = Math.round((finished.length / n) * 100);
  $('#progressBar').style.width = `${pct}%`;
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

const VIEWS = [
  { id: 'swipe', label: 'Swipe', need: (v) => v.pairA },
  { id: 'side', label: 'Side by side', need: () => true },
  { id: 'blink', label: 'Blink', need: (v) => v.pairA },
  { id: 'deltaE', label: 'Colour difference', need: (v) => v.deltaE },
  { id: 'ssim', label: 'SSIM map', need: (v) => v.ssim },
  { id: 'matches', label: 'Keypoint matches', need: (v) => v.matches },
  { id: 'aligned', label: 'Aligned overlay', need: (v) => v.aligned },
];

function renderViewModes() {
  const box = $('#viewModes');
  box.innerHTML = '';
  const v = state.visuals || {};
  const available = VIEWS.filter((m) => (m.id === 'side' ? state.slots.a && state.slots.b : m.need(v)));
  if (!state.viewChosen && available.some((m) => m.id === 'swipe')) state.view = 'swipe';
  if (!available.some((m) => m.id === state.view)) state.view = available[0]?.id || 'side';
  for (const m of available) {
    box.append(
      el(
        'button',
        {
          type: 'button',
          class: 'pill',
          'aria-pressed': String(m.id === state.view),
          onclick: () => {
            state.view = m.id;
            state.viewChosen = true;
            renderViewModes();
            renderVisual();
          },
        },
        m.label,
      ),
    );
  }
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
  if (state.view === 'side' || !v) {
    if (!a || !b) return;
    viewer.append(
      el(
        'div',
        { class: 'view-pair' },
        el('figure', {}, el('img', { src: a.url, alt: 'Image A' }), el('figcaption', {}, `A · ${a.w}×${a.h}`)),
        el('figure', {}, el('img', { src: b.url, alt: 'Image B' }), el('figcaption', {}, `B · ${b.w}×${b.h}`)),
      ),
    );
    caption.textContent = v ? 'Original files, scaled to fit.' : 'More views appear when the scan finishes.';
    return;
  }
  const displayWidth = Math.min(viewer.clientWidth - 32, Math.max(v.pairA.w, 640));
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
    caption.textContent = 'Drag across the image. B is resized to A’s dimensions, exactly as the pixel and structural tests see it.';
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
    caption.textContent = 'Flicker comparison: differences jump out as motion.';
  } else if (state.view === 'deltaE' || state.view === 'ssim') {
    const c = canvasFrom(v[state.view]);
    c.style.width = `${displayWidth}px`;
    viewer.append(el('div', {}, c, el('div', { class: 'colorbar' }, 'similar', el('i'), 'different')));
    caption.textContent =
      state.view === 'deltaE'
        ? 'Perceptual colour difference (CIEDE2000) at each pixel; bright = large change (ΔE ≥ 25).'
        : 'Local SSIM: bright areas are where luminance, contrast or structure differ.';
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
];

async function fetchSample(name) {
  const res = await fetch(new URL(`assets/samples/${name}.jpg`, ROOT));
  const blob = await res.blob();
  return new File([blob], `${name}.jpg`, { type: 'image/jpeg', lastModified: 0 });
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
  liable: { short: 'Infringement', long: 'Court: infringement' },
  fairuse: { short: 'Fair use', long: 'Court: fair use' },
  nosim: { short: 'No infringement', long: 'Court: no infringement' },
};

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
  for (const g of CASE_GROUPS) {
    const cases = CASES.filter((c) => c.group === g.id);
    root.append(
      el(
        'section',
        { class: 'case-group' },
        el('div', { class: 'case-group-head' }, el('h3', {}, g.title), el('span', { class: 'muted small' }, `${cases.length} cases`)),
        el('p', { class: 'case-group-blurb' }, g.blurb),
        el('div', { class: 'case-grid' }, cases.map(caseCard)),
      ),
    );
  }
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
      el('span', { class: 'case-outcome', 'data-group': c.group }, OUTCOME[c.group].short),
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
    startScan();
  } catch (err) {
    console.error(err);
    toast('Could not load the images for this case.');
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
  box.append(
    el(
      'div',
      { class: 'case-banner-top' },
      el('span', { class: 'case-outcome', 'data-group': c.group }, OUTCOME[c.group].long),
      el('span', { class: 'case-banner-name' }, el('i', {}, c.name), `, ${c.cite}`),
    ),
    el('p', { class: 'case-banner-pairing' }, c.pairing, c.note ? ` — ${c.note}` : ''),
    labels,
    c.caution ? el('p', { class: 'case-caution' }, c.caution) : null,
    el(
      'p',
      { class: 'muted small' },
      'Do the tests below line up with the court? Similarity scores measure resemblance between pixels or learned features; what counts as protectable expression, how much was taken and whether the use was fair are questions they do not measure.',
    ),
  );
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
    root.append(el('section', { class: 'cat-group' }, el('h3', {}, g.title), el('p', {}, g.blurb), grid));
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
    });
  }
}

function init() {
  setupDrops();
  setupTabs();
  buildLab();
  renderCatalogue();
  renderCases();
  loadCaseManifest();
  for (const s of SAMPLES) $('#samples').append(el('button', { type: 'button', class: 'pill', onclick: () => loadSample(s) }, s.label));
  $('#scanBtn').addEventListener('click', startScan);
  $('#labBtn').addEventListener('click', openLab);
  $('#labLink').addEventListener('click', (e) => {
    e.stopPropagation();
    openLab();
  });
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
    }, 200);
  });
  refreshDownloadNote();
  updateButtons();
  if (new URLSearchParams(location.search).has('debug')) window.scanner = { state, report };
  // the catalogue is built at runtime, so honour #anchors once it exists
  if (location.hash) document.querySelector(location.hash)?.scrollIntoView();
}

init();
