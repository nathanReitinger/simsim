import { ENGINES, ENGINE_BY_ID, GROUPS, VERDICT_LABEL } from './engines.js';
import { MODELS } from './lib/neural.js';
import { ModelStore } from './lib/modelstore.js';
import { applyTransform, DEFAULTS, PRESETS, describe } from './transform.js';
import { CASES, CASE_GROUPS } from './cases.js';
import { lutColor } from './lib/colormap.js';
import * as AV from './annotate-view.js';

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
const MODEL_ORDER = ['sscd', 'lpips', 'dino', 'dfine', 'sscdLarge', 'clip'];
const MODEL_INFO = {
  sscd: { name: 'SSCD (ResNet-50)', note: 'The main copy detector. Recommended.' },
  sscdLarge: { name: 'SSCD large (ResNeXt-101)', note: 'Replication-study setting from Somepalli et al.' },
  dino: { name: 'DINOv2 small', note: 'General visual similarity.' },
  clip: { name: 'CLIP ViT-B/32', note: 'Semantic similarity; the largest download.' },
  lpips: { name: 'LPIPS (AlexNet)', note: 'Perceptual distance; tiny.' },
  dfine: { name: 'D-FINE object detector', note: 'Finds hats, pictures, tables… for the Objects tab.' },
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
  whereLens: null,
  whereChosen: false,
  openGroups: new Set(['neural', 'objects']),
  objectView: 'markup',
  objectHighlight: null,
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
      renderSpectrum();
      renderWhereCard();
      renderViewModes();
      renderVisual();
      renderObjects();
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
  const share = state.visuals?.dino?.mutualShare;
  if ((clip !== null && clip >= 0.75) || (dino !== null && dino >= 0.4) || (share !== undefined && share >= 0.1)) {
    return { tone: 'similar', title: 'Similar subject, not a copy', sub: 'Semantic models (CLIP / DINOv2) see related content, but the copy detectors do not flag it.' };
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
  if (is('pixels', 'identical')) return { id: 'pixels', why: 'every decoded pixel is identical' };
  if ((is('pdq', 'match') || (is('phash', 'match') && is('dhash', 'match'))) && (is('msssim', 'match') || is('ssim', 'match'))) {
    return { id: 'resaved', why: 'the perceptual hashes match and the pixels line up' };
  }
  if (sscd !== null && sscd >= 0.75) return { id: 'edited', why: `SSCD scores ${sscd.toFixed(2)}, above the 0.75 copy threshold` };
  if (is('pdqDihedral', 'match')) return { id: 'edited', why: 'PDQ matches once B is rotated or mirrored' };
  if (is('crop', 'identical', 'match')) return { id: 'part', why: 'one image appears inside the other' };
  if (sscd !== null && sscd >= 0.5) return { id: 'part', why: `SSCD scores ${sscd.toFixed(2)}, the range Somepalli et al. associate with partial copies` };
  const kp = ['orb', 'akaze', 'brisk'].filter((id) => is(id, 'match'));
  if (kp.length) return { id: 'part', why: `${kp.map((k) => k.toUpperCase()).join(', ')} keypoints match in one consistent geometry` };
  const dino = val('dino');
  const clip = val('clip');
  const share = V?.dino?.mutualShare;
  const sem = [
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
  box.hidden = !done;
  if (!done) return;
  const level = similarityLevel(state.results, state.visuals);
  const info = LEVELS.find((l) => l.id === level.id);
  box.innerHTML = '';
  box.append(
    el('div', { class: 'spectrum-label' }, 'Where this pair sits on the similarity spectrum'),
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
    ids: ['diff', 'regions', 'evidence'],
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
    need: (v) => v.sscd,
    what: 'SSCD’s copy score is one number, but it can be split exactly into contributions from each region of the image (plus a constant close to zero). The heat map shows that split and the circles mark its peaks: the regions that supplied most of the evidence for the score. Everything fades when the score itself is low.',
    means: 'This is what the copy detector “looked at”. Evidence on the subject means SSCD is matching the reproduced content; evidence on a watermark, caption or border means the score may reflect that shared element more than the work itself.',
    ref: { label: 'Stylianou, Souvenir & Pless, Visualizing Deep Similarity Networks (2019)', url: 'https://arxiv.org/abs/1901.00536' },
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
  if (ann?.regions?.length) prefer.push('regions');
  if (ann?.aligned) prefer.push('diff');
  prefer.push('parts', 'evidence', 'heat', 'swipe');
  return prefer.find((id) => available.includes(id));
}

const ANN_LENSES = new Set(['diff', 'regions', 'evidence']);

function evidenceUnderlay(p) {
  const e = state.visuals.sscd;
  const peak = (grid) => Math.max(1e-9, ...grid.values);
  const strength = Math.max(0.15, Math.min(1, (e.score - 0.15) / 0.45));
  heatOverlay(p.g, e.a, p.a, (x) => x / peak(e.a), strength);
  heatOverlay(p.g, e.b, p.b, (x) => x / peak(e.b), strength);
}

/** Text under a marked-up lens. */
function annotationNote(id) {
  const v = state.visuals;
  const ann = v.annotations;
  if (id === 'diff') {
    if (ann.global) return 'After alignment B differs from A almost everywhere (recoloured, filtered or redrawn), so there are no isolated spots to circle. Try the Matching regions lens.';
    if (!ann.differences.length) return 'After alignment no part of B differs from A beyond tiny shifts: as far as the pixels go, B is a faithful copy of A.';
    const n = ann.differences.length;
    return `${n} difference${n === 1 ? '' : 's'} circled, largest first. Point at (or tap) a number to single it out.`;
  }
  if (id === 'regions') {
    const n = ann.regions.length;
    const whole = ann.regions[0] && ann.regions[0].share >= 0.6;
    return `${whole ? 'Nearly all of A reappears in B. ' : ''}${n} matching region${n === 1 ? '' : 's'}, joined A → B; the percentage is how alike their patches are. Point at (or tap) one to follow its arrow.`;
  }
  const e = v.sscd;
  return `SSCD score ${e.score.toFixed(3)}: the heat adds up to this score, and the circles mark where most of it comes from.${e.score < 0.5 ? ' Everything is faint because SSCD finds little copy evidence overall.' : ''}`;
}

/** Where a box sits in its image, in words ("top left", "centre"). */
function placeName(box, img) {
  const x = (box[0] + box[2]) / 2 / img.w;
  const y = (box[1] + box[3]) / 2 / img.h;
  const v = y < 1 / 3 ? 'top' : y > 2 / 3 ? 'bottom' : '';
  const h = x < 1 / 3 ? 'left' : x > 2 / 3 ? 'right' : '';
  return [v, h].filter(Boolean).join(' ') || 'centre';
}

/** A marked-up view: annotated canvas plus a numbered list of close-ups. */
function annotatedView(id, cssWidth) {
  const v = state.visuals;
  const ann = v.annotations || { differences: [], regions: [] };
  const A = state.slots.a.bitmap;
  const B = state.slots.b.bitmap;
  const holder = el('div', { class: 'ann-stage' });
  let focus = null;
  let stageNow = null;
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
        : [];
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
  );
  const draw = () => {
    if (id === 'diff') stageNow = AV.drawDifferences(A, B, ann, { cssWidth, focus });
    else if (id === 'regions') stageNow = AV.drawRegions(A, B, ann, { cssWidth, focus });
    else stageNow = AV.drawEvidence(A, B, ann?.peaks ? ann : { peaks: { score: v.sscd.score, a: [], b: [] } }, { cssWidth, underlay: evidenceUnderlay });
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
  return el('div', { class: `ann-view ann-${id}` }, holder, items.length ? list : '', id === 'evidence' ? el('div', { class: 'colorbar' }, 'no evidence', el('i'), 'most evidence') : '');
}

const WHERE_LENSES = [
  { id: 'diff', label: 'Differences', need: (v) => v.annotations?.aligned },
  { id: 'regions', label: 'Matching regions', need: (v) => v.annotations?.regions?.length },
  { id: 'evidence', label: 'Copy evidence', need: (v) => v.sscd && v.annotations?.peaks },
  { id: 'parts', label: 'Matching parts', need: (v) => v.dino, short: 'Same-coloured dots mark parts of A and B that DINOv2 finds most alike (mutual best matches).' },
  { id: 'heat', label: 'Heat map', need: (v) => v.dino, short: 'How closely every region of each image has a counterpart in the other.' },
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
      el(
        'p',
        { class: 'muted small' },
        state.scanning
          ? 'Marked-up views appear here when the scan finishes.'
          : 'Turn on the DINOv2 or SSCD models in Settings to see where the similarity comes from.',
      ),
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
  if (ANN_LENSES.has(lens.id)) {
    box.append(annotatedView(lens.id, width), el('p', { class: 'where-note' }, annotationNote(lens.id)));
    return;
  }
  let canvas;
  let note = lens.short;
  if (lens.id === 'parts') {
    canvas = drawParts(v.dino, width, false, false);
    note += ` ${Math.round(v.dino.mutualShare * 100)}% of A’s patches have a mutual match in B.`;
  } else {
    canvas = drawHeatPair(v.dino.a, v.dino.b, dinoV, 1, width);
  }
  box.append(el('div', { class: 'lens-view' }, canvas), el('p', { class: 'where-note' }, note));
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
    viewer.append(
      el(
        'div',
        { class: 'view-pair' },
        el('figure', {}, el('img', { src: a.url, alt: 'Image A' }), el('figcaption', {}, `A · ${a.w}×${a.h}`)),
        el('figure', {}, el('img', { src: b.url, alt: 'Image B' }), el('figcaption', {}, `B · ${b.w}×${b.h}`)),
      ),
    );
    caption.textContent = v ? '' : 'More views appear when the scan finishes.';
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
    viewer.append(el('div', { class: 'lens-view' }, annotatedView(state.view, fullWidth)));
    caption.textContent = annotationNote(state.view);
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
  link.replaceChildren(...(ready.length === 1 ? [el('i', {}, ready[0].name), ' ↓'] : [`${ready.length} image cases ↓`]));
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
  const parts = [
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
  if (new URLSearchParams(location.search).has('debug')) window.scanner = { state, report };
  // the catalogue is built at runtime, so honour #anchors once it exists
  if (location.hash) document.querySelector(location.hash)?.scrollIntoView();
}

init();
