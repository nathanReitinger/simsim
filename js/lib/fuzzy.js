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
