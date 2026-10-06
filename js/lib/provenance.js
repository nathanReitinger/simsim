// Provenance signals that live in the file rather than in the pixels: the
// encoded image data with metadata set aside, the JPEG encoder's tables, who
// the file credits (copyright management information), and where its edit
// history says it came from.

// IJG (Annex K) luminance and chrominance tables, in zigzag order as stored.
const ZIGZAG = [
  0, 1, 8, 16, 9, 2, 3, 10, 17, 24, 32, 25, 18, 11, 4, 5, 12, 19, 26, 33, 40, 48, 41, 34, 27, 20, 13, 6, 7, 14, 21, 28, 35, 42, 49, 56, 57, 50, 43, 36, 29, 22, 15,
  23, 30, 37, 44, 51, 58, 59, 52, 45, 38, 31, 39, 46, 53, 60, 61, 54, 47, 55, 62, 63,
];
const STD_LUMA = [
  16, 11, 10, 16, 24, 40, 51, 61, 12, 12, 14, 19, 26, 58, 60, 55, 14, 13, 16, 24, 40, 57, 69, 56, 14, 17, 22, 29, 51, 87, 80, 62, 18, 22, 37, 56, 68, 109, 103, 77,
  24, 35, 55, 64, 81, 104, 113, 92, 49, 64, 78, 87, 103, 121, 120, 101, 72, 92, 95, 98, 112, 100, 103, 99,
];
const STD_CHROMA = [
  17, 18, 24, 47, 99, 99, 99, 99, 18, 21, 26, 66, 99, 99, 99, 99, 24, 26, 56, 99, 99, 99, 99, 99, 47, 66, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99,
  99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99,
];

const u16 = (b, i) => (b[i] << 8) | b[i + 1];
const u32 = (b, i) => ((b[i] << 24) | (b[i + 1] << 16) | (b[i + 2] << 8) | b[i + 3]) >>> 0;
const ascii = (b, i, n) => String.fromCharCode(...b.subarray(i, i + n));
const has = (b, text) => {
  const t = Array.from(text, (ch) => ch.charCodeAt(0));
  outer: for (let i = 0; i + t.length <= b.length; i++) {
    for (let k = 0; k < t.length; k++) if (b[i + k] !== t[k]) continue outer;
    return true;
  }
  return false;
};

/** IJG quality whose scaled standard table is closest to `table` (in zigzag order). */
function ijgQuality(table, std) {
  let best = { q: 0, err: Infinity };
  for (let q = 1; q <= 100; q++) {
    const scale = q < 50 ? 5000 / q : 200 - 2 * q;
    let err = 0;
    for (let i = 0; i < 64; i++) {
      const v = Math.min(255, Math.max(1, Math.floor((std[ZIGZAG[i]] * scale + 50) / 100)));
      err += Math.abs(v - table[i]);
    }
    if (err < best.err) best = { q, err };
  }
  return best;
}

function parseJpeg(b) {
  const out = { format: 'jpeg', segments: [], dqt: [], payload: [], c2pa: false, progressive: false, components: [] };
  let i = 2;
  while (i + 4 <= b.length) {
    if (b[i] !== 0xff) {
      i++;
      continue;
    }
    const m = b[i + 1];
    if (m === 0xff) {
      i++;
      continue;
    }
    if (m === 0xd8 || m === 0x01 || (m >= 0xd0 && m <= 0xd7)) {
      i += 2;
      continue;
    }
    if (m === 0xd9) break;
    const len = u16(b, i + 2);
    const seg = b.subarray(i, i + 2 + len);
    const data = b.subarray(i + 4, i + 2 + len);
    const name = m >= 0xe0 && m <= 0xef ? `APP${m - 0xe0}` : m === 0xfe ? 'COM' : m === 0xdb ? 'DQT' : m === 0xc4 ? 'DHT' : m === 0xda ? 'SOS' : m === 0xdd ? 'DRI' : m >= 0xc0 && m <= 0xcf ? `SOF${m - 0xc0}` : `0x${m.toString(16)}`;
    out.segments.push(name);
    const isMeta = (m >= 0xe0 && m <= 0xef) || m === 0xfe;
    if (m === 0xeb && (has(data, 'jumb') || has(data, 'c2pa'))) out.c2pa = true;
    if (m === 0xdb) {
      let k = 0;
      while (k < data.length) {
        const pq = data[k] >> 4;
        const tq = data[k] & 15;
        const n = pq ? 128 : 64;
        const t = new Uint16Array(64);
        for (let j = 0; j < 64; j++) t[j] = pq ? u16(data, k + 1 + 2 * j) : data[k + 1 + j];
        out.dqt.push({ id: tq, table: t });
        k += 1 + n;
      }
    }
    if (m >= 0xc0 && m <= 0xcf && m !== 0xc4 && m !== 0xc8 && m !== 0xcc) {
      out.progressive = m === 0xc2 || m === 0xc6 || m === 0xca || m === 0xce;
      const nf = data[5];
      for (let c = 0; c < nf; c++) out.components.push({ id: data[6 + 3 * c], h: data[7 + 3 * c] >> 4, v: data[7 + 3 * c] & 15, tq: data[8 + 3 * c] });
    }
    if (!isMeta) out.payload.push(seg);
    if (m === 0xda) {
      // entropy-coded data runs to the next marker that is not a stuffed byte or restart
      let j = i + 2 + len;
      while (j + 1 < b.length && !(b[j] === 0xff && b[j + 1] !== 0 && !(b[j + 1] >= 0xd0 && b[j + 1] <= 0xd7))) j++;
      out.payload.push(b.subarray(i + 2 + len, j));
      i = j;
      continue;
    }
    i += 2 + len;
  }
  const luma = out.dqt.find((t) => t.id === (out.components[0]?.tq ?? 0)) || out.dqt[0];
  const chroma = out.components[1] ? out.dqt.find((t) => t.id === out.components[1].tq) : null;
  if (luma) {
    // encoders built on libjpeg scale the Annex K luminance table exactly;
    // chroma tables vary between libjpeg versions, so only luminance decides
    const ql = ijgQuality(luma.table, STD_LUMA);
    out.quality = ql.q;
    out.standardTables = ql.err === 0;
    out.chromaQuality = chroma ? ijgQuality(chroma.table, STD_CHROMA).q : null;
  }
  const y = out.components[0];
  if (out.components.length === 1) out.subsampling = 'greyscale';
  else if (y) out.subsampling = y.h === 1 && y.v === 1 ? '4:4:4' : y.h === 2 && y.v === 2 ? '4:2:0' : y.h === 2 && y.v === 1 ? '4:2:2' : `${y.h}×${y.v}`;
  return out;
}

async function inflate(bytes) {
  const ds = new DecompressionStream('deflate');
  const buf = await new Response(new Blob([bytes]).stream().pipeThrough(ds)).arrayBuffer();
  return new Uint8Array(buf);
}

async function parsePng(b) {
  const out = { format: 'png', payload: [], text: {}, c2pa: false, segments: [] };
  let i = 8;
  const latin1 = new TextDecoder('latin1');
  const utf8 = new TextDecoder('utf-8');
  while (i + 12 <= b.length) {
    const len = u32(b, i);
    const type = ascii(b, i + 4, 4);
    const data = b.subarray(i + 8, i + 8 + len);
    out.segments.push(type);
    if (['IHDR', 'PLTE', 'tRNS', 'IDAT', 'acTL', 'fcTL', 'fdAT'].includes(type)) out.payload.push(data);
    if (type === 'caBX') out.c2pa = true;
    try {
      if (type === 'tEXt') {
        const z = data.indexOf(0);
        out.text[latin1.decode(data.subarray(0, z))] = latin1.decode(data.subarray(z + 1));
      } else if (type === 'zTXt') {
        const z = data.indexOf(0);
        out.text[latin1.decode(data.subarray(0, z))] = latin1.decode(await inflate(data.subarray(z + 2)));
      } else if (type === 'iTXt') {
        const z = data.indexOf(0);
        const key = latin1.decode(data.subarray(0, z));
        const compressed = data[z + 1] === 1;
        let k = z + 3;
        k = data.indexOf(0, k) + 1; // language tag
        k = data.indexOf(0, k) + 1; // translated keyword
        const body = data.subarray(k);
        out.text[key] = utf8.decode(compressed ? await inflate(body) : body);
      }
    } catch {
      // unreadable text chunk: ignore
    }
    if (type === 'IEND') break;
    i += 12 + len;
  }
  return out;
}

function parseWebp(b) {
  const out = { format: 'webp', payload: [], c2pa: false, segments: [] };
  let i = 12;
  while (i + 8 <= b.length) {
    const type = ascii(b, i, 4);
    const len = b[i + 4] | (b[i + 5] << 8) | (b[i + 6] << 16) | (b[i + 7] << 24);
    out.segments.push(type.trim());
    if (['VP8 ', 'VP8L', 'ALPH', 'ANMF'].includes(type)) out.payload.push(b.subarray(i + 8, i + 8 + len));
    if (type === 'C2PA') out.c2pa = true;
    i += 8 + len + (len & 1);
  }
  return out;
}

/** Container-level facts about a file: format, payload parts, encoder details, text chunks. */
export async function parseContainer(bytes) {
  const b = bytes;
  if (b[0] === 0xff && b[1] === 0xd8) return parseJpeg(b);
  if (b[0] === 0x89 && ascii(b, 1, 3) === 'PNG') return parsePng(b);
  if (ascii(b, 0, 4) === 'RIFF' && ascii(b, 8, 4) === 'WEBP') return parseWebp(b);
  return { format: 'other', payload: null, c2pa: has(b.subarray(0, Math.min(b.length, 1 << 20)), 'c2pa'), segments: [] };
}

/** SHA-256 of the image data alone (metadata segments removed), or null. */
export async function payloadHash(info) {
  if (!info.payload || !info.payload.length) return null;
  const total = info.payload.reduce((t, p) => t + p.length, 0);
  const all = new Uint8Array(total);
  let o = 0;
  for (const p of info.payload) {
    all.set(p, o);
    o += p.length;
  }
  const d = await crypto.subtle.digest('SHA-256', all);
  return { hex: Array.from(new Uint8Array(d), (x) => x.toString(16).padStart(2, '0')).join(''), bytes: total };
}

/** Two JPEGs' quantization tables are identical. */
export function sameTables(a, b) {
  if (!a.dqt?.length || a.dqt.length !== b.dqt?.length) return false;
  return a.dqt.every((t, i) => t.id === b.dqt[i].id && t.table.every((v, k) => v === b.dqt[i].table[k]));
}

/** IPTC strings arrive as Latin-1; most are really UTF-8. */
function fixText(s) {
  if (typeof s !== 'string' || !/[\u0080-ÿ]/.test(s)) return s;
  try {
    return new TextDecoder('utf-8', { fatal: true }).decode(Uint8Array.from(s, (c) => c.charCodeAt(0) & 255));
  } catch {
    return s;
  }
}

const text = (v) => {
  if (v === undefined || v === null || v === '') return null;
  if (Array.isArray(v)) return v.map(text).filter(Boolean).join('; ') || null;
  if (typeof v === 'object') return text(v.value ?? v.description ?? Object.values(v).find((x) => typeof x === 'string'));
  return fixText(String(v)).trim() || null;
};

/** Credit and rights fields (copyright management information) from parsed metadata. */
export function cmiFields(meta) {
  if (!meta) return {};
  const m = meta;
  const f = {
    creator: text(m.dc?.creator) || text(m.iptc?.Byline) || text(m.ifd0?.Artist),
    rights: text(m.dc?.rights) || text(m.iptc?.CopyrightNotice) || text(m.ifd0?.Copyright),
    credit: text(m.photoshop?.Credit) || text(m.iptc?.Credit),
    source: text(m.photoshop?.Source) || text(m.iptc?.Source),
    terms: text(m.xmpRights?.UsageTerms),
    licence: text(m.xmpRights?.WebStatement) || text(m.plus?.Licensor),
  };
  for (const k of Object.keys(f)) if (!f[k]) delete f[k];
  return f;
}

/** IDs from the XMP media-management (edit history) schema. */
export function lineage(meta) {
  const mm = meta?.xmpMM || {};
  const list = (v) => (v === undefined || v === null ? [] : Array.isArray(v) ? v : [v]);
  const derived = list(mm.DerivedFrom).flatMap((d) => [d?.documentID, d?.instanceID, d?.originalDocumentID]).filter(Boolean);
  const ancestors = list(meta?.photoshop?.DocumentAncestors).flatMap((x) => (typeof x === 'object' ? Object.values(x) : [x])).filter(Boolean).map(String);
  const ingredients = list(mm.Ingredients).flatMap((d) => [d?.documentID, d?.instanceID]).filter(Boolean);
  const history = list(mm.History).map((h) => [h?.action, h?.softwareAgent, h?.when].filter(Boolean).join(' · ')).filter(Boolean);
  return {
    documentID: mm.DocumentID ? String(mm.DocumentID) : null,
    instanceID: mm.InstanceID ? String(mm.InstanceID) : null,
    originalID: mm.OriginalDocumentID ? String(mm.OriginalDocumentID) : null,
    derived: derived.map(String),
    ancestors,
    ingredients: ingredients.map(String),
    history,
    tool: text(meta?.xmp?.CreatorTool) || text(meta?.ifd0?.Software),
    sourceType: text(meta?.Iptc4xmpExt?.DigitalSourceType),
  };
}

/** Generator settings that AI image tools write into PNG text chunks. */
export function generatorText(png) {
  const t = png?.text || {};
  if (t.parameters) return { tool: 'Stable Diffusion web UI (A1111/Forge)', text: t.parameters };
  if (t.prompt || t.workflow) return { tool: 'ComfyUI', text: (t.prompt || t.workflow).slice(0, 400) };
  if (t.Dream) return { tool: 'InvokeAI', text: t.Dream };
  if (t['sd-metadata']) return { tool: 'InvokeAI', text: t['sd-metadata'].slice(0, 400) };
  return null;
}
