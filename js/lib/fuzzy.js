// Byte-level fuzzy hashes: how much two FILES share as byte sequences, the way
// malware analysts compare files. ssdeep (context-triggered piecewise hashing,
// Kornblum 2006, after Tridgell's spamsum) is ported from ppdeep (Apache-2.0)
// and gives identical hashes and scores; TLSH uses Trend Micro's own port.

import { Tlsh } from './tlsh.js';

const BLOCKSIZE_MIN = 3;
const SPAMSUM_LENGTH = 64;
const B64 = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/';
const F_TABLE = [
  0x00, 0x13, 0x26, 0x39, 0x0c, 0x1f, 0x32, 0x05, 0x18, 0x2b, 0x3e, 0x11, 0x24, 0x37, 0x0a, 0x1d, 0x30, 0x03, 0x16, 0x29, 0x3c, 0x0f, 0x22, 0x35, 0x08, 0x1b, 0x2e, 0x01, 0x14, 0x27, 0x3a, 0x0d,
  0x20, 0x33, 0x06, 0x19, 0x2c, 0x3f, 0x12, 0x25, 0x38, 0x0b, 0x1e, 0x31, 0x04, 0x17, 0x2a, 0x3d, 0x10, 0x23, 0x36, 0x09, 0x1c, 0x2f, 0x02, 0x15, 0x28, 0x3b, 0x0e, 0x21, 0x34, 0x07, 0x1a, 0x2d,
];
// byteTable[b][h] = F_TABLE[h] ^ (b & 63): the partial FNV step of spamsum
const BYTE_TABLE = Array.from({ length: 256 }, (_, b) => Uint8Array.from({ length: 64 }, (__, h) => F_TABLE[h] ^ (b & 0x3f)));

/** ssdeep hash of a byte array ("blocksize:sig1:sig2"). */
export function ssdeepHash(buf) {
  const ROLL = 7;
  const HASH_INIT = 0x27;
  let bs = BLOCKSIZE_MIN;
  while (bs * SPAMSUM_LENGTH < buf.length) bs *= 2;
  let blockSize = bs;
  for (;;) {
    const win = new Uint8Array(ROLL);
    let h1 = 0;
    let h2 = 0;
    let h3 = 0;
    let n = 0;
    let b1 = HASH_INIT;
    let b2 = HASH_INIT;
    let s1 = '';
    let s2 = '';
    let last1 = '';
    let last2 = '';
    let rh = 0;
    for (let i = 0; i < buf.length; i++) {
      const b = buf[i];
      b1 = BYTE_TABLE[b][b1];
      b2 = BYTE_TABLE[b][b2];
      h2 = h2 - h1 + ROLL * b;
      h1 = h1 + b - win[n];
      win[n] = b;
      n = (n + 1) % ROLL;
      h3 = ((h3 << 5) >>> 0) ^ b;
      rh = (h1 + h2 + h3) >>> 0;
      if (rh % blockSize === blockSize - 1) {
        last1 = B64[b1];
        if (s1.length < SPAMSUM_LENGTH - 1) {
          s1 += B64[b1];
          b1 = HASH_INIT;
          last1 = '';
        }
        if (rh % (blockSize * 2) === blockSize * 2 - 1) {
          last2 = B64[b2];
          if (s2.length < SPAMSUM_LENGTH / 2 - 1) {
            s2 += B64[b2];
            b2 = HASH_INIT;
            last2 = '';
          }
        }
      }
    }
    if (blockSize > BLOCKSIZE_MIN && s1.length < SPAMSUM_LENGTH / 2) {
      blockSize = Math.floor(blockSize / 2);
      continue;
    }
    if (rh !== 0) {
      s1 += B64[b1];
      s2 += B64[b2];
    } else {
      s1 += last1;
      s2 += last2;
    }
    return `${blockSize}:${s1}:${s2}`;
  }
}

function levenshtein(s, t) {
  if (s === t) return 0;
  if (!s.length) return t.length;
  if (!t.length) return s.length;
  let v0 = Array.from({ length: t.length + 1 }, (_, i) => i);
  let v1 = new Array(t.length + 1);
  for (let i = 0; i < s.length; i++) {
    v1[0] = i + 1;
    for (let j = 0; j < t.length; j++) v1[j + 1] = Math.min(v1[j] + 1, v0[j + 1] + 1, v0[j] + (s[i] === t[j] ? 0 : 1));
    [v0, v1] = [v1, v0];
  }
  return v0[t.length];
}

function commonSubstring(a, b) {
  for (let i = 0; i < a.length; i++) {
    for (let j = 0; j < b.length; j++) {
      let k = 0;
      while (i + k < a.length && j + k < b.length && a[i + k] === b[j + k]) k++;
      if (k >= 7) return true;
    }
  }
  return false;
}

function scoreStrings(a, b, blockSize) {
  if (!commonSubstring(a, b)) return 0;
  let score = levenshtein(a, b);
  score = Math.floor((score * SPAMSUM_LENGTH) / (a.length + b.length));
  score = Math.floor((100 * score) / SPAMSUM_LENGTH);
  score = 100 - score;
  const cap = Math.floor(blockSize / BLOCKSIZE_MIN) * Math.min(a.length, b.length);
  return score > cap ? cap : score;
}

const strip = (s) => {
  let r = s.slice(0, 3);
  for (let i = 3; i < s.length; i++) if (s[i] !== s[i - 1] || s[i] !== s[i - 2] || s[i] !== s[i - 3]) r += s[i];
  return r;
};

/** ssdeep match score 0–100 (as ppdeep.compare / ssdeep -d). */
export function ssdeepCompare(h1, h2) {
  const [bs1s, a1, a2] = h1.split(':');
  const [bs2s, b1, b2] = h2.split(':');
  const bs1 = Number(bs1s);
  const bs2 = Number(bs2s);
  if (bs1 !== bs2 && bs1 !== bs2 * 2 && bs2 !== bs1 * 2) return 0;
  const x1 = strip(a1);
  const x2 = strip(a2);
  const y1 = strip(b1);
  const y2 = strip(b2);
  if (bs1 === bs2 && x1 === y1) return 100;
  if (bs1 === bs2) return Math.max(scoreStrings(x1, y1, bs1), scoreStrings(x2, y2, bs2 * 2));
  if (bs1 === bs2 * 2) return scoreStrings(x1, y2, bs1);
  return scoreStrings(x2, y1, bs2);
}

/** TLSH digest (70 hex characters) of a byte array, or null if the file is too small or too uniform. */
export function tlshHash(buf) {
  try {
    const t = new Tlsh();
    t.finale(buf, buf.length);
    return t.hash();
  } catch {
    return null;
  }
}

/** TLSH distance between two digests (0 = identical; above ~100 = unrelated). */
export function tlshDiff(h1, h2) {
  const a = new Tlsh();
  a.fromTlshStr(h1);
  const b = new Tlsh();
  b.fromTlshStr(h2);
  return a.totalDiff(b);
}

// ---------------------------------------------------------------- Nilsimsa

// Nilsimsa (Damiani et al. 2004; the "cmeclax" reference code, as the Python nilsimsa package)
const NIL_TRAN = [
  0x02, 0xd6, 0x9e, 0x6f, 0xf9, 0x1d, 0x04, 0xab, 0xd0, 0x22, 0x16, 0x1f, 0xd8, 0x73, 0xa1, 0xac, 0x3b, 0x70, 0x62, 0x96, 0x1e, 0x6e, 0x8f, 0x39, 0x9d, 0x05, 0x14, 0x4a, 0xa6, 0xbe, 0xae, 0x0e,
  0xcf, 0xb9, 0x9c, 0x9a, 0xc7, 0x68, 0x13, 0xe1, 0x2d, 0xa4, 0xeb, 0x51, 0x8d, 0x64, 0x6b, 0x50, 0x23, 0x80, 0x03, 0x41, 0xec, 0xbb, 0x71, 0xcc, 0x7a, 0x86, 0x7f, 0x98, 0xf2, 0x36, 0x5e, 0xee,
  0x8e, 0xce, 0x4f, 0xb8, 0x32, 0xb6, 0x5f, 0x59, 0xdc, 0x1b, 0x31, 0x4c, 0x7b, 0xf0, 0x63, 0x01, 0x6c, 0xba, 0x07, 0xe8, 0x12, 0x77, 0x49, 0x3c, 0xda, 0x46, 0xfe, 0x2f, 0x79, 0x1c, 0x9b, 0x30,
  0xe3, 0x00, 0x06, 0x7e, 0x2e, 0x0f, 0x38, 0x33, 0x21, 0xad, 0xa5, 0x54, 0xca, 0xa7, 0x29, 0xfc, 0x5a, 0x47, 0x69, 0x7d, 0xc5, 0x95, 0xb5, 0xf4, 0x0b, 0x90, 0xa3, 0x81, 0x6d, 0x25, 0x55, 0x35,
  0xf5, 0x75, 0x74, 0x0a, 0x26, 0xbf, 0x19, 0x5c, 0x1a, 0xc6, 0xff, 0x99, 0x5d, 0x84, 0xaa, 0x66, 0x3e, 0xaf, 0x78, 0xb3, 0x20, 0x43, 0xc1, 0xed, 0x24, 0xea, 0xe6, 0x3f, 0x18, 0xf3, 0xa0, 0x42,
  0x57, 0x08, 0x53, 0x60, 0xc3, 0xc0, 0x83, 0x40, 0x82, 0xd7, 0x09, 0xbd, 0x44, 0x2a, 0x67, 0xa8, 0x93, 0xe0, 0xc2, 0x56, 0x9f, 0xd9, 0xdd, 0x85, 0x15, 0xb4, 0x8a, 0x27, 0x28, 0x92, 0x76, 0xde,
  0xef, 0xf8, 0xb2, 0xb7, 0xc9, 0x3d, 0x45, 0x94, 0x4b, 0x11, 0x0d, 0x65, 0xd5, 0x34, 0x8b, 0x91, 0x0c, 0xfa, 0x87, 0xe9, 0x7c, 0x5b, 0xb1, 0x4d, 0xe5, 0xd4, 0xcb, 0x10, 0xa2, 0x17, 0x89, 0xbc,
  0xdb, 0xb0, 0xe2, 0x97, 0x88, 0x52, 0xf7, 0x48, 0xd3, 0x61, 0x2c, 0x3a, 0x2b, 0xd1, 0x8c, 0xfb, 0xf1, 0xcd, 0xe4, 0x6a, 0xe7, 0xa9, 0xfd, 0xc4, 0x37, 0xc8, 0xd2, 0xf6, 0xdf, 0x58, 0x72, 0x4e,
];

const tran = (a, b, c, n) => ((NIL_TRAN[(a + n) & 255] ^ (NIL_TRAN[b] * (n + n + 1))) + NIL_TRAN[c ^ NIL_TRAN[n]]) & 255;

/**
 * Nilsimsa digest (32 bytes): counts of hashed character trigrams within a
 * sliding five-byte window, each bucket set if above the average. Built in
 * 2001 to spot near-duplicate spam e-mails.
 */
export function nilsimsaDigest(buf) {
  const acc = new Uint32Array(256);
  let w0 = -1;
  let w1 = -1;
  let w2 = -1;
  let w3 = -1;
  for (let i = 0; i < buf.length; i++) {
    const c = buf[i];
    if (w1 >= 0) acc[tran(c, w0, w1, 0)]++;
    if (w2 >= 0) {
      acc[tran(c, w0, w2, 1)]++;
      acc[tran(c, w1, w2, 2)]++;
    }
    if (w3 >= 0) {
      acc[tran(c, w0, w3, 3)]++;
      acc[tran(c, w1, w3, 4)]++;
      acc[tran(c, w2, w3, 5)]++;
      acc[tran(w3, w0, c, 6)]++;
      acc[tran(w3, w2, c, 7)]++;
    }
    [w0, w1, w2, w3] = [c, w0, w1, w2];
  }
  const n = buf.length;
  const trigrams = n === 3 ? 1 : n === 4 ? 4 : n > 4 ? 8 * n - 28 : 0;
  const threshold = trigrams / 256;
  const digest = new Uint8Array(32);
  for (let i = 0; i < 256; i++) if (acc[i] > threshold) digest[i >> 3] += 1 << (i & 7);
  return digest.reverse();
}

/** Nilsimsa score: 128 minus the number of differing bits, from −128 to 128 (128 = same digest). */
export function nilsimsaCompare(a, b) {
  let bits = 0;
  for (let i = 0; i < 32; i++) {
    let x = a[i] ^ b[i];
    while (x) {
      bits += x & 1;
      x >>= 1;
    }
  }
  return 128 - bits;
}

// ---------------------------------------------------------------- LZJD

const rotl = (x, r) => ((x << r) | (x >>> (32 - r))) | 0;
function fmix32(h) {
  h ^= h >>> 16;
  h = Math.imul(h, 0x85ebca6b);
  h ^= h >>> 13;
  h = Math.imul(h, 0xc2b2ae35);
  h ^= h >>> 16;
  return h;
}

// pyLZJD sizes its open-addressing hash set with these primes
const TWIN_PRIMES = [7, 13, 19, 43, 73, 139, 271, 523, 1033, 2083, 4129, 8221, 16453, 32803, 65539, 131113, 262153, 524353, 1048891, 2097259, 4194583, 8388619, 16777291, 33554503, 67109323, 134217781];

/** Hoare partition and quickselect exactly as pyLZJD (whose selection is only approximately the k smallest). */
function hoarePartition(arr, left, right) {
  const pivot = arr[left];
  let i = left - 1;
  let j = right;
  for (;;) {
    i++;
    while (arr[i] < pivot) i++;
    j--;
    while (arr[j] > pivot) j--;
    if (i >= j) return j;
    const t = arr[j];
    arr[j] = arr[i];
    arr[i] = t;
  }
}

function quickSelect(arr, left, right, k) {
  for (;;) {
    const p = hoarePartition(arr, left, right);
    if (k === p || right - left <= 1) return;
    if (k < p) right = p;
    else left = p + 1;
  }
}

/**
 * LZJD digest (Raff & Nicholas 2017), as pyLZJD's lzjd_f: split the bytes
 * Lempel–Ziv style into phrases never seen before, hash each phrase
 * (MurmurHash3, pushed byte by byte, quirks included), and keep about the
 * 1,024 smallest hashes — selected with pyLZJD's own hash table and
 * quickselect so the digests are identical to the reference.
 */
export function lzjdDigest(buf, k = 1024) {
  let ti = 0;
  while (TWIN_PRIMES[ti] && Math.floor(TWIN_PRIMES[ti] / 2) < buf.length) ti++;
  const size = TWIN_PRIMES[ti];
  if (!size) return null; // file too large
  const keys = new Int32Array(size);
  const used = new Uint8Array(size);
  const slot = (key) => {
    const h = key & 0x7fffffff;
    let i = h % size;
    if (!used[i] || keys[i] === key) return i;
    const c = 1 + (h % (size - 2));
    for (;;) {
      i -= c;
      if (i < 0) i += size;
      if (!used[i] || keys[i] === key) return i;
    }
  };
  const data = new Int8Array(4);
  let len = 0;
  let h1 = 0;
  let count = 0;
  const c1 = 0xcc9e2d51 | 0;
  const c2 = 0x1b873593;
  for (let i = 0; i < buf.length; i++) {
    data[len % 4] = buf[i]; // stored as a signed char, as in the C code
    len++;
    let out;
    if (len % 4 === 0) {
      let k1 = (data[0] & 0xff) | ((data[1] & 0xff) << 8) | ((data[2] & 0xff) << 16) | (data[3] << 24);
      k1 = Math.imul(k1, c1);
      k1 = rotl(k1, 15);
      k1 = Math.imul(k1, c2);
      h1 ^= k1;
      h1 = rotl(h1, 13);
      h1 = (Math.imul(h1, 5) + (0xe6546b64 | 0)) | 0;
      out = h1;
      data.fill(0);
    } else {
      // the reference masks the tail bytes with C's "and" (0 or 1), not 0xff
      let k1 = (data[0] & 0xff) | ((data[1] & (len >= 1 ? 1 : 0)) << 8) | ((data[2] & (len >= 2 ? 1 : 0)) << 16) | ((data[3] && len >= 1 ? 1 : 0) << 24);
      k1 = Math.imul(k1, c1);
      k1 = rotl(k1, 15);
      k1 = Math.imul(k1, c2);
      out = h1 ^ k1;
    }
    out = fmix32(out ^ len);
    const at = slot(out);
    if (!used[at]) {
      used[at] = 1;
      keys[at] = out;
      count++;
      len = 0;
      h1 = 0;
      data.fill(0);
    }
  }
  // the set in table order, then pyLZJD's selection and a sort of the first k
  const arr = new Int32Array(count);
  for (let i = 0, p = 0; i < size; i++) if (used[i]) arr[p++] = keys[i];
  let n = count;
  if (count > k) {
    quickSelect(arr, 0, count, k);
    n = k;
  }
  return arr.subarray(0, n).sort();
}

/** pyLZJD's similarity: shared hashes over all hashes of the two digests, 0..1. */
export function lzjdSimilarity(a, b) {
  let i = 0;
  let j = 0;
  let shared = 0;
  while (i < a.length && j < b.length) {
    if (a[i] < b[j]) i++;
    else if (a[i] > b[j]) j++;
    else {
      i++;
      j++;
      shared++;
    }
  }
  const total = a.length + b.length - shared;
  return total ? shared / total : 1;
}
