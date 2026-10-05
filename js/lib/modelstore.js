// Downloads model files with progress and keeps them in Cache Storage so a
// model is only downloaded once per browser. Runs on the page (main thread),
// which stays responsive while the worker is busy computing.

const CACHE_NAME = 'image-compare-models-v1';

async function openCache() {
  try {
    return typeof caches === 'undefined' ? null : await caches.open(CACHE_NAME);
  } catch {
    return null; // e.g. storage disabled in a private window
  }
}

function concat(chunks, total) {
  const out = new Uint8Array(total);
  let off = 0;
  for (const c of chunks) {
    out.set(c, off);
    off += c.length;
  }
  return out;
}

async function fetchFile(url, cache, onBytes) {
  if (cache) {
    const hit = await cache.match(url);
    if (hit) {
      const buf = new Uint8Array(await hit.arrayBuffer());
      onBytes(buf.length);
      return buf;
    }
  }
  const res = await fetch(url);
  if (!res.ok) throw new Error(`HTTP ${res.status} while downloading ${url}`);
  const reader = res.body.getReader();
  const chunks = [];
  let loaded = 0;
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    chunks.push(value);
    loaded += value.length;
    onBytes(loaded);
  }
  const buf = concat(chunks, loaded);
  if (cache) {
    try {
      await cache.put(url, new Response(buf, { headers: { 'content-type': 'application/octet-stream' } }));
    } catch {
      // quota exceeded: keep working without the cache
    }
  }
  return buf;
}

export class ModelStore {
  constructor(models, baseUrl, onProgress) {
    this.models = models;
    this.base = baseUrl;
    this.onProgress = onProgress;
    this.pending = new Map();
    this.ready = new Map(); // downloaded bytes waiting to be handed to the worker
    this.memory = new Map(); // fallback when Cache Storage is unavailable
  }

  urls(key) {
    return this.models[key].files.map((f) => new URL(f, this.base).href);
  }

  /** Download (or read from cache) ahead of time. */
  prefetch(key) {
    if (this.ready.has(key) || this.memory.has(key)) return Promise.resolve();
    if (!this.pending.has(key)) {
      const p = this.load(key)
        .then((bytes) => {
          if (!this.memory.has(key)) this.ready.set(key, bytes);
        })
        .finally(() => this.pending.delete(key));
      this.pending.set(key, p);
    }
    return this.pending.get(key);
  }

  /** Resolves with bytes the caller owns (and may transfer to a worker). */
  async get(key) {
    await this.prefetch(key);
    if (this.memory.has(key)) return this.memory.get(key).slice();
    const bytes = this.ready.get(key);
    this.ready.delete(key);
    return bytes || this.load(key);
  }

  async load(key) {
    const cache = await openCache();
    const total = this.models[key].bytes;
    const parts = [];
    let done = 0;
    this.onProgress(key, 0, total, 'start');
    for (const url of this.urls(key)) {
      const part = await fetchFile(url, cache, (n) => this.onProgress(key, done + n, total, 'progress'));
      done += part.length;
      parts.push(part);
    }
    const bytes = parts.length === 1 ? parts[0] : concat(parts, done);
    if (!cache) this.memory.set(key, bytes);
    this.onProgress(key, done, total, 'done');
    return bytes;
  }

  async isCached(key) {
    if (this.memory.has(key)) return true;
    const cache = await openCache();
    if (!cache) return false;
    for (const url of this.urls(key)) if (!(await cache.match(url))) return false;
    return true;
  }

  async clear() {
    this.memory.clear();
    if (typeof caches !== 'undefined') await caches.delete(CACHE_NAME);
  }
}
