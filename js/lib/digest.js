// File digests. SHA-1 / SHA-256 come from Web Crypto; MD5 is not offered
// there, so it is implemented here (RFC 1321).

const S = [
  7, 12, 17, 22, 7, 12, 17, 22, 7, 12, 17, 22, 7, 12, 17, 22, 5, 9, 14, 20, 5, 9, 14, 20, 5, 9, 14, 20, 5, 9, 14, 20, 4,
  11, 16, 23, 4, 11, 16, 23, 4, 11, 16, 23, 4, 11, 16, 23, 6, 10, 15, 21, 6, 10, 15, 21, 6, 10, 15, 21, 6, 10, 15, 21,
];
const K = Array.from({ length: 64 }, (_, i) => Math.floor(Math.abs(Math.sin(i + 1)) * 2 ** 32) | 0);

function toHex(bytes) {
  return Array.from(bytes, (b) => b.toString(16).padStart(2, '0')).join('');
}

export function md5(bytes) {
  const len = bytes.length;
  const blocks = ((len + 8) >>> 6) + 1;
  const buf = new Uint8Array(blocks * 64);
  buf.set(bytes);
  buf[len] = 0x80;
  const view = new DataView(buf.buffer);
  const bitLen = len * 8;
  view.setUint32(buf.length - 8, bitLen >>> 0, true);
  view.setUint32(buf.length - 4, Math.floor(bitLen / 2 ** 32), true);

  let a0 = 0x67452301;
  let b0 = 0xefcdab89 | 0;
  let c0 = 0x98badcfe | 0;
  let d0 = 0x10325476;
  const M = new Int32Array(16);
  for (let off = 0; off < buf.length; off += 64) {
    for (let i = 0; i < 16; i++) M[i] = view.getInt32(off + i * 4, true);
    let A = a0;
    let B = b0;
    let C = c0;
    let D = d0;
    for (let i = 0; i < 64; i++) {
      let F;
      let g;
      if (i < 16) {
        F = (B & C) | (~B & D);
        g = i;
      } else if (i < 32) {
        F = (D & B) | (~D & C);
        g = (5 * i + 1) & 15;
      } else if (i < 48) {
        F = B ^ C ^ D;
        g = (3 * i + 5) & 15;
      } else {
        F = C ^ (B | ~D);
        g = (7 * i) & 15;
      }
      F = (F + A + K[i] + M[g]) | 0;
      A = D;
      D = C;
      C = B;
      B = (B + ((F << S[i]) | (F >>> (32 - S[i])))) | 0;
    }
    a0 = (a0 + A) | 0;
    b0 = (b0 + B) | 0;
    c0 = (c0 + C) | 0;
    d0 = (d0 + D) | 0;
  }
  const out = new DataView(new ArrayBuffer(16));
  out.setInt32(0, a0, true);
  out.setInt32(4, b0, true);
  out.setInt32(8, c0, true);
  out.setInt32(12, d0, true);
  return toHex(new Uint8Array(out.buffer));
}

export async function sha(algorithm, bytes) {
  return toHex(new Uint8Array(await crypto.subtle.digest(algorithm, bytes)));
}

// ---------------------------------------------------------------- CRC-32

const CRC_TABLE = (() => {
  const t = new Uint32Array(256);
  for (let n = 0; n < 256; n++) {
    let c = n;
    for (let k = 0; k < 8; k++) c = c & 1 ? 0xedb88320 ^ (c >>> 1) : c >>> 1;
    t[n] = c >>> 0;
  }
  return t;
})();

/** CRC-32 (IEEE 802.3, as in zlib / PNG / ZIP), as 8 hex digits. */
export function crc32(bytes) {
  let c = 0xffffffff;
  for (let i = 0; i < bytes.length; i++) c = CRC_TABLE[(c ^ bytes[i]) & 0xff] ^ (c >>> 8);
  return ((c ^ 0xffffffff) >>> 0).toString(16).padStart(8, '0');
}

// ---------------------------------------------------------------- SHA3-256 (FIPS 202)

const RC = [
  [0x00000001, 0x00000000], [0x00008082, 0x00000000], [0x0000808a, 0x80000000], [0x80008000, 0x80000000], [0x0000808b, 0x00000000], [0x80000001, 0x00000000],
  [0x80008081, 0x80000000], [0x00008009, 0x80000000], [0x0000008a, 0x00000000], [0x00000088, 0x00000000], [0x80008009, 0x00000000], [0x8000000a, 0x00000000],
  [0x8000808b, 0x00000000], [0x0000008b, 0x80000000], [0x00008089, 0x80000000], [0x00008003, 0x80000000], [0x00008002, 0x80000000], [0x00000080, 0x80000000],
  [0x0000800a, 0x00000000], [0x8000000a, 0x80000000], [0x80008081, 0x80000000], [0x00008080, 0x80000000], [0x80000001, 0x00000000], [0x80008008, 0x80000000],
];
const ROT = [0, 1, 62, 28, 27, 36, 44, 6, 55, 20, 3, 10, 43, 25, 39, 41, 45, 15, 21, 8, 18, 2, 61, 56, 14];

/** Keccak-f[1600] on 25 lanes stored as (lo, hi) 32-bit halves. */
function keccakF(lo, hi) {
  const cLo = new Uint32Array(5);
  const cHi = new Uint32Array(5);
  const bLo = new Uint32Array(25);
  const bHi = new Uint32Array(25);
  for (let round = 0; round < 24; round++) {
    for (let x = 0; x < 5; x++) {
      cLo[x] = lo[x] ^ lo[x + 5] ^ lo[x + 10] ^ lo[x + 15] ^ lo[x + 20];
      cHi[x] = hi[x] ^ hi[x + 5] ^ hi[x + 10] ^ hi[x + 15] ^ hi[x + 20];
    }
    for (let x = 0; x < 5; x++) {
      const nLo = cLo[(x + 1) % 5];
      const nHi = cHi[(x + 1) % 5];
      const dLo = cLo[(x + 4) % 5] ^ ((nLo << 1) | (nHi >>> 31));
      const dHi = cHi[(x + 4) % 5] ^ ((nHi << 1) | (nLo >>> 31));
      for (let y = 0; y < 25; y += 5) {
        lo[y + x] ^= dLo;
        hi[y + x] ^= dHi;
      }
    }
    // rho and pi
    for (let x = 0; x < 5; x++) {
      for (let y = 0; y < 5; y++) {
        const i = x + 5 * y;
        const r = ROT[i];
        let l = lo[i];
        let h = hi[i];
        if (r >= 32) {
          [l, h] = [h, l];
        }
        const s = r % 32;
        const nl = s ? (l << s) | (h >>> (32 - s)) : l;
        const nh = s ? (h << s) | (l >>> (32 - s)) : h;
        const j = y + 5 * ((2 * x + 3 * y) % 5);
        bLo[j] = nl;
        bHi[j] = nh;
      }
    }
    // chi
    for (let y = 0; y < 25; y += 5) {
      for (let x = 0; x < 5; x++) {
        lo[y + x] = bLo[y + x] ^ (~bLo[y + ((x + 1) % 5)] & bLo[y + ((x + 2) % 5)]);
        hi[y + x] = bHi[y + x] ^ (~bHi[y + ((x + 1) % 5)] & bHi[y + ((x + 2) % 5)]);
      }
    }
    // iota
    lo[0] ^= RC[round][0];
    hi[0] ^= RC[round][1];
  }
}

/** SHA3-256 as hex. */
export function sha3_256(bytes) {
  const rate = 136;
  const lo = new Uint32Array(25);
  const hi = new Uint32Array(25);
  const padded = new Uint8Array(Math.ceil((bytes.length + 1) / rate) * rate);
  padded.set(bytes);
  padded[bytes.length] ^= 0x06;
  padded[padded.length - 1] ^= 0x80;
  const view = new DataView(padded.buffer);
  for (let off = 0; off < padded.length; off += rate) {
    for (let i = 0; i < rate / 8; i++) {
      lo[i] ^= view.getUint32(off + 8 * i, true);
      hi[i] ^= view.getUint32(off + 8 * i + 4, true);
    }
    keccakF(lo, hi);
  }
  const out = new Uint8Array(32);
  const ov = new DataView(out.buffer);
  for (let i = 0; i < 4; i++) {
    ov.setUint32(8 * i, lo[i], true);
    ov.setUint32(8 * i + 4, hi[i], true);
  }
  return toHex(out);
}

// ---------------------------------------------------------------- BLAKE2b (RFC 7693)

const B2_IV = [
  0xf3bcc908, 0x6a09e667, 0x84caa73b, 0xbb67ae85, 0xfe94f82b, 0x3c6ef372, 0x5f1d36f1, 0xa54ff53a, 0xade682d1, 0x510e527f, 0x2b3e6c1f, 0x9b05688c, 0xfb41bd6b, 0x1f83d9ab, 0x137e2179, 0x5be0cd19,
];
const SIGMA = [
  [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15], [14, 10, 4, 8, 9, 15, 13, 6, 1, 12, 0, 2, 11, 7, 5, 3], [11, 8, 12, 0, 5, 2, 15, 13, 10, 14, 3, 6, 7, 1, 9, 4],
  [7, 9, 3, 1, 13, 12, 11, 14, 2, 6, 5, 10, 4, 0, 15, 8], [9, 0, 5, 7, 2, 4, 10, 15, 14, 1, 11, 12, 6, 8, 3, 13], [2, 12, 6, 10, 0, 11, 8, 3, 4, 13, 7, 5, 15, 14, 1, 9],
  [12, 5, 1, 15, 14, 13, 4, 10, 0, 7, 6, 3, 9, 2, 8, 11], [13, 11, 7, 14, 12, 1, 3, 9, 5, 0, 15, 4, 8, 6, 2, 10], [6, 15, 14, 9, 11, 3, 0, 8, 12, 2, 13, 7, 1, 4, 10, 5],
  [10, 2, 8, 4, 7, 6, 1, 5, 15, 11, 9, 14, 3, 12, 13, 0], [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15], [14, 10, 4, 8, 9, 15, 13, 6, 1, 12, 0, 2, 11, 7, 5, 3],
];

/** BLAKE2b with a 32-byte digest, as hex. 64-bit words as (lo, hi) pairs in a Uint32Array. */
export function blake2b256(bytes) {
  const outLen = 32;
  const h = new Uint32Array(16);
  for (let i = 0; i < 16; i++) h[i] = B2_IV[i];
  h[0] ^= 0x01010000 ^ outLen; // parameter block: digest length, fanout 1, depth 1
  const v = new Uint32Array(32);
  const m = new Uint32Array(32);
  const add = (a, b) => {
    const lo = v[a] + v[b];
    v[a + 1] = v[a + 1] + v[b + 1] + (lo >= 0x100000000 ? 1 : 0);
    v[a] = lo;
  };
  const addM = (a, w) => {
    const lo = v[a] + m[w];
    v[a + 1] = v[a + 1] + m[w + 1] + (lo >= 0x100000000 ? 1 : 0);
    v[a] = lo;
  };
  const xorRot = (a, b, r) => {
    const xl = v[a] ^ v[b];
    const xh = v[a + 1] ^ v[b + 1];
    if (r === 32) {
      v[a] = xh;
      v[a + 1] = xl;
    } else if (r < 32) {
      v[a] = (xl >>> r) | (xh << (32 - r));
      v[a + 1] = (xh >>> r) | (xl << (32 - r));
    } else {
      const s = r - 32;
      v[a] = (xh >>> s) | (xl << (32 - s));
      v[a + 1] = (xl >>> s) | (xh << (32 - s));
    }
  };
  const G = (a, b, c, d, x, y) => {
    add(a, b);
    addM(a, x);
    xorRot(d, a, 32);
    add(c, d);
    xorRot(b, c, 24);
    add(a, b);
    addM(a, y);
    xorRot(d, a, 16);
    add(c, d);
    xorRot(b, c, 63);
  };
  const compress = (block, t, last) => {
    for (let i = 0; i < 16; i++) {
      v[i] = h[i];
      v[i + 16] = B2_IV[i];
    }
    v[24] ^= t >>> 0;
    v[25] ^= Math.floor(t / 0x100000000);
    if (last) {
      v[28] = ~v[28];
      v[29] = ~v[29];
    }
    const dv = new DataView(block.buffer, block.byteOffset, 128);
    for (let i = 0; i < 32; i++) m[i] = dv.getUint32(4 * i, true);
    for (let r = 0; r < 12; r++) {
      const s = SIGMA[r];
      G(0, 8, 16, 24, s[0] * 2, s[1] * 2);
      G(2, 10, 18, 26, s[2] * 2, s[3] * 2);
      G(4, 12, 20, 28, s[4] * 2, s[5] * 2);
      G(6, 14, 22, 30, s[6] * 2, s[7] * 2);
      G(0, 10, 20, 30, s[8] * 2, s[9] * 2);
      G(2, 12, 22, 24, s[10] * 2, s[11] * 2);
      G(4, 14, 16, 26, s[12] * 2, s[13] * 2);
      G(6, 8, 18, 28, s[14] * 2, s[15] * 2);
    }
    for (let i = 0; i < 16; i++) h[i] ^= v[i] ^ v[i + 16];
  };
  const n = bytes.length;
  const blocks = Math.max(1, Math.ceil(n / 128));
  for (let b = 0; b < blocks; b++) {
    const block = new Uint8Array(128);
    block.set(bytes.subarray(b * 128, Math.min(n, b * 128 + 128)));
    const last = b === blocks - 1;
    compress(block, last ? n : (b + 1) * 128, last);
  }
  const out = new Uint8Array(outLen);
  const ov = new DataView(out.buffer);
  for (let i = 0; i < outLen / 4; i++) ov.setUint32(4 * i, h[i], true);
  return toHex(out);
}
