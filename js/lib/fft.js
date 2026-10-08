// Fast Fourier transforms of any length: a Stockham (self-sorting)
// mixed-radix FFT, with Bluestein's chirp-z algorithm when the length has a
// large prime factor. Results match numpy.fft (forward unscaled, inverse
// divided by n).

const plans = new Map();

function factorize(n) {
  const f = [];
  for (const p of [4, 2, 3, 5]) {
    while (n % p === 0) {
      f.push(p);
      n /= p;
    }
  }
  for (let p = 7; p * p <= n; p += 2) {
    while (n % p === 0) {
      f.push(p);
      n /= p;
    }
  }
  if (n > 1) f.push(n);
  return f;
}

function makePlan(n, sign) {
  const key = `${n}:${sign}`;
  if (plans.has(key)) return plans.get(key);
  const factors = factorize(n);
  let plan;
  if (factors.some((p) => p > 64)) {
    plan = { kind: 'bluestein', ...bluesteinPlan(n, sign) };
  } else {
    // per stage: radix p, sub-length len, twiddles w_len^(q·r)
    const stages = [];
    let len = n;
    for (const p of factors) {
      const m = len / p;
      const tr = new Float64Array(m * p);
      const ti = new Float64Array(m * p);
      for (let q = 0; q < m; q++) {
        for (let r = 0; r < p; r++) {
          const a = (sign * 2 * Math.PI * q * r) / len;
          tr[q * p + r] = Math.cos(a);
          ti[q * p + r] = Math.sin(a);
        }
      }
      const cr = new Float64Array(p * p);
      const ci = new Float64Array(p * p);
      for (let j = 0; j < p; j++) {
        for (let r = 0; r < p; r++) {
          const a = (sign * 2 * Math.PI * ((j * r) % p)) / p;
          cr[j * p + r] = Math.cos(a);
          ci[j * p + r] = Math.sin(a);
        }
      }
      stages.push({ p, m, len, tr, ti, cr, ci });
      len = m;
    }
    plan = { kind: 'stockham', sign, stages, br: new Float64Array(n), bi: new Float64Array(n) };
  }
  plans.set(key, plan);
  return plan;
}

function stockham(plan, re, im) {
  let xr = re;
  let xi = im;
  let yr = plan.br;
  let yi = plan.bi;
  let s = 1;
  const sign = plan.sign;
  const ar = new Float64Array(64);
  const ai = new Float64Array(64);
  const h3 = Math.sqrt(3) / 2;
  // write butterfly output r of group (q, k), times the twiddle w_len^(q·r)
  const put = (tr, ti, q, p, r, o, sr, si) => {
    const wr = tr[q * p + r];
    const wi = ti[q * p + r];
    yr[o] = sr * wr - si * wi;
    yi[o] = sr * wi + si * wr;
  };
  for (const { p, m, tr, ti, cr, ci } of plan.stages) {
    const ms = m * s;
    for (let q = 0; q < m; q++) {
      for (let k = 0; k < s; k++) {
        const i0 = k + s * q;
        const o0 = k + s * p * q;
        if (p === 2) {
          const a0r = xr[i0];
          const a0i = xi[i0];
          const a1r = xr[i0 + ms];
          const a1i = xi[i0 + ms];
          put(tr, ti, q, 2, 0, o0, a0r + a1r, a0i + a1i);
          put(tr, ti, q, 2, 1, o0 + s, a0r - a1r, a0i - a1i);
        } else if (p === 4) {
          const a0r = xr[i0];
          const a0i = xi[i0];
          const a1r = xr[i0 + ms];
          const a1i = xi[i0 + ms];
          const a2r = xr[i0 + 2 * ms];
          const a2i = xi[i0 + 2 * ms];
          const a3r = xr[i0 + 3 * ms];
          const a3i = xi[i0 + 3 * ms];
          const b0r = a0r + a2r;
          const b0i = a0i + a2i;
          const b1r = a0r - a2r;
          const b1i = a0i - a2i;
          const c0r = a1r + a3r;
          const c0i = a1i + a3i;
          // (a1 − a3) times ω = e^{sign·iπ/2} = sign·i
          const dr = -sign * (a1i - a3i);
          const di = sign * (a1r - a3r);
          put(tr, ti, q, 4, 0, o0, b0r + c0r, b0i + c0i);
          put(tr, ti, q, 4, 1, o0 + s, b1r + dr, b1i + di);
          put(tr, ti, q, 4, 2, o0 + 2 * s, b0r - c0r, b0i - c0i);
          put(tr, ti, q, 4, 3, o0 + 3 * s, b1r - dr, b1i - di);
        } else if (p === 3) {
          const a0r = xr[i0];
          const a0i = xi[i0];
          const a1r = xr[i0 + ms];
          const a1i = xi[i0 + ms];
          const a2r = xr[i0 + 2 * ms];
          const a2i = xi[i0 + 2 * ms];
          const t1r = a1r + a2r;
          const t1i = a1i + a2i;
          const t2r = a0r - t1r / 2;
          const t2i = a0i - t1i / 2;
          // i·sign·(√3/2)·(a1 − a2)
          const ur = -sign * h3 * (a1i - a2i);
          const ui = sign * h3 * (a1r - a2r);
          put(tr, ti, q, 3, 0, o0, a0r + t1r, a0i + t1i);
          put(tr, ti, q, 3, 1, o0 + s, t2r + ur, t2i + ui);
          put(tr, ti, q, 3, 2, o0 + 2 * s, t2r - ur, t2i - ui);
        } else {
          for (let j = 0; j < p; j++) {
            ar[j] = xr[i0 + j * ms];
            ai[j] = xi[i0 + j * ms];
          }
          for (let r = 0; r < p; r++) {
            let sr = 0;
            let si = 0;
            for (let j = 0; j < p; j++) {
              const wr = cr[j * p + r];
              const wi = ci[j * p + r];
              sr += ar[j] * wr - ai[j] * wi;
              si += ar[j] * wi + ai[j] * wr;
            }
            put(tr, ti, q, p, r, o0 + r * s, sr, si);
          }
        }
      }
    }
    [xr, yr] = [yr, xr];
    [xi, yi] = [yi, xi];
    s *= p;
  }
  if (xr !== re) {
    re.set(xr);
    im.set(xi);
  }
}

/** The smallest length ≥ n whose prime factors are 2, 3 and 5 (fast to transform). */
export function goodSize(n) {
  for (let m = Math.max(1, n); ; m++) {
    let k = m;
    for (const p of [2, 3, 5]) while (k % p === 0) k /= p;
    if (k === 1) return m;
  }
}

function bluesteinPlan(n, sign) {
  let m = 1;
  while (m < 2 * n - 1) m *= 2;
  const wr = new Float64Array(n);
  const wi = new Float64Array(n);
  for (let k = 0; k < n; k++) {
    const a = (sign * Math.PI * ((k * k) % (2 * n))) / n;
    wr[k] = Math.cos(a);
    wi[k] = Math.sin(a);
  }
  // FFT of the conjugate chirp, wrapped into length m
  const br = new Float64Array(m);
  const bi = new Float64Array(m);
  br[0] = wr[0];
  bi[0] = -wi[0];
  for (let k = 1; k < n; k++) {
    br[k] = br[m - k] = wr[k];
    bi[k] = bi[m - k] = -wi[k];
  }
  fft(br, bi);
  return { m, wr, wi, br, bi, ur: new Float64Array(m), ui: new Float64Array(m) };
}

function bluestein(plan, re, im) {
  const n = re.length;
  const { m, wr, wi, br, bi, ur, ui } = plan;
  ur.fill(0);
  ui.fill(0);
  for (let k = 0; k < n; k++) {
    ur[k] = re[k] * wr[k] - im[k] * wi[k];
    ui[k] = re[k] * wi[k] + im[k] * wr[k];
  }
  fft(ur, ui);
  for (let k = 0; k < m; k++) {
    const r = ur[k] * br[k] - ui[k] * bi[k];
    ui[k] = ur[k] * bi[k] + ui[k] * br[k];
    ur[k] = r;
  }
  fft(ur, ui, true);
  for (let k = 0; k < n; k++) {
    re[k] = ur[k] * wr[k] - ui[k] * wi[k];
    im[k] = ur[k] * wi[k] + ui[k] * wr[k];
  }
}

/** In-place FFT of a complex sequence (Float64Arrays). The inverse is divided by n, as numpy.fft.ifft. */
export function fft(re, im, inverse = false) {
  const n = re.length;
  if (n <= 1) return;
  const plan = makePlan(n, inverse ? 1 : -1);
  if (plan.kind === 'stockham') stockham(plan, re, im);
  else bluestein(plan, re, im);
  if (inverse) {
    for (let k = 0; k < n; k++) {
      re[k] /= n;
      im[k] /= n;
    }
  }
}

/** In-place 2-D FFT of a h×w complex image stored row by row (numpy.fft.fft2 / ifft2). */
export function fft2(re, im, w, h, inverse = false) {
  const rr = new Float64Array(w);
  const ri = new Float64Array(w);
  for (let y = 0; y < h; y++) {
    const o = y * w;
    rr.set(re.subarray(o, o + w));
    ri.set(im.subarray(o, o + w));
    fft(rr, ri, inverse);
    re.set(rr, o);
    im.set(ri, o);
  }
  const cr = new Float64Array(h);
  const ci = new Float64Array(h);
  for (let x = 0; x < w; x++) {
    for (let y = 0; y < h; y++) {
      cr[y] = re[y * w + x];
      ci[y] = im[y * w + x];
    }
    fft(cr, ci, inverse);
    for (let y = 0; y < h; y++) {
      re[y * w + x] = cr[y];
      im[y * w + x] = ci[y];
    }
  }
}
