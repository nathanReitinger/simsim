// OpenCV's img_hash module (opencv_contrib, after the pHash library):
// block-mean, colour-moment, Marr–Hildreth and radial-variance hashes.
// Resizing, blurring and colour conversion go through OpenCV.js so the
// arithmetic is OpenCV's own; the rest is ported line by line from
// modules/img_hash/src/*.cpp, quirks included, so the hashes are the ones
// cv2.img_hash computes.

function rgbMat(cv, img) {
  const m = new cv.Mat(img.h, img.w, cv.CV_8UC3);
  m.data.set(img.rgb);
  return m;
}

function withMats(fn) {
  const mats = [];
  const keep = (m) => (mats.push(m), m);
  try {
    return fn(keep);
  } finally {
    for (const m of mats) m.delete();
  }
}

/** Number of differing bits between two byte arrays (cv::NORM_HAMMING). */
export function hammingBytes(a, b) {
  let d = 0;
  for (let i = 0; i < a.length; i++) {
    let x = a[i] ^ b[i];
    while (x) {
      d += x & 1;
      x >>= 1;
    }
  }
  return d;
}

/**
 * BlockMeanHash (Yang, Gu & Niu 2006): 256×256 grey image, the mean of each
 * 16×16 block (mode 0: 256 blocks; mode 1: half-overlapping, 961 blocks)
 * against the mean of the whole image. Bytes as OpenCV packs them.
 */
export function blockMeanHash(cv, img, mode = 0) {
  return withMats((keep) => {
    const resized = keep(new cv.Mat());
    cv.resize(keep(rgbMat(cv, img)), resized, new cv.Size(256, 256), 0, 0, cv.INTER_LINEAR_EXACT);
    const gray = keep(new cv.Mat());
    cv.cvtColor(resized, gray, cv.COLOR_RGB2GRAY);
    const g = gray.data;
    const step = mode === 1 ? 8 : 16;
    const means = [];
    for (let row = 0; row <= 240; row += step) {
      for (let col = 0; col <= 240; col += step) {
        let s = 0;
        for (let y = row; y < row + 16; y++) for (let x = col; x < col + 16; x++) s += g[y * 256 + x];
        means.push(s / 256);
      }
    }
    let total = 0;
    for (const v of g) total += v;
    const mean = total / g.length;
    const n = means.length;
    const out = new Uint8Array(Math.floor(n / 8) + (n % 8));
    let byte = 0;
    for (let i = 0; i < n; i++) {
      const bit = means[i] < mean ? 0 : 1;
      const r = i % 8;
      byte = (byte & ~(1 << r)) | (bit << r);
      if (r === 7) out[(i - 7) / 8] = byte;
      else if (i === n - 1) out[Math.floor(i / 8)] = bit; // OpenCV stores only the last bit here
    }
    return out;
  });
}

/** The 7 Hu moment invariants of one 8-bit channel, as cv::HuMoments(cv::moments(ch)). */
function huMoments(cv, ch, keep) {
  const m = cv.moments(ch, false);
  const hu = keep(new cv.Mat());
  cv.HuMoments(m, hu);
  return Array.from(hu.data64F);
}

/**
 * ColorMomentHash (after Tang et al.): the Hu moments of each channel
 * of the image in HSV and in YCrCb, after resizing to 512×512 and a light
 * blur — 42 numbers that survive rotation, scaling and mild noise.
 */
export function colorMomentHash(cv, img) {
  return withMats((keep) => {
    const resized = keep(new cv.Mat());
    cv.resize(keep(rgbMat(cv, img)), resized, new cv.Size(512, 512), 0, 0, cv.INTER_CUBIC);
    const blur = keep(new cv.Mat());
    cv.GaussianBlur(resized, blur, new cv.Size(3, 3), 0, 0);
    const out = [];
    for (const code of [cv.COLOR_RGB2HSV, cv.COLOR_RGB2YCrCb]) {
      const space = keep(new cv.Mat());
      cv.cvtColor(blur, space, code);
      const planes = new cv.MatVector();
      try {
        cv.split(space, planes);
        for (let i = 0; i < 3; i++) out.push(...huMoments(cv, keep(planes.get(i)), keep));
      } finally {
        planes.delete();
      }
    }
    return Float64Array.from(out);
  });
}

/** OpenCV's ColorMomentHash distance: L2 norm × 10000 (lower = more similar). */
export function colorMomentDistance(a, b) {
  let s = 0;
  for (let i = 0; i < a.length; i++) s += (a[i] - b[i]) ** 2;
  return Math.sqrt(s) * 10000;
}

/**
 * MarrHildrethHash (Zauner 2010, pHash's "MH" hash): equalised 512×512 grey
 * image filtered with a Marr–Hildreth (Laplacian of Gaussian) kernel, summed
 * over 16×16 blocks, then 64 neighbourhoods of 3×3 blocks each give 9 bits
 * (block above the neighbourhood average): 576 bits. As in OpenCV, the kernel
 * uses exp(+a/2) and the block grid is read transposed.
 */
export function marrHildrethHash(cv, img, alpha = 2, level = 1) {
  return withMats((keep) => {
    const gray = keep(new cv.Mat());
    cv.cvtColor(keep(rgbMat(cv, img)), gray, cv.COLOR_RGB2GRAY);
    const blur = keep(new cv.Mat());
    cv.GaussianBlur(gray, blur, new cv.Size(7, 7), 0);
    const resized = keep(new cv.Mat());
    cv.resize(blur, resized, new cv.Size(512, 512), 0, 0, cv.INTER_CUBIC);
    const eq = keep(new cv.Mat());
    cv.equalizeHist(resized, eq);
    const f = Math.fround;
    const sigma = Math.trunc(f(4 * f(alpha ** level)));
    const ratio = f(alpha ** -level);
    const size = 2 * sigma + 1;
    const kernel = keep(new cv.Mat(size, size, cv.CV_32F));
    const kd = kernel.data32F;
    for (let row = 0; row < size; row++) {
      const ypos = f(ratio * (row - sigma));
      const y2 = f(ypos * ypos);
      for (let col = 0; col < size; col++) {
        const xpos = f(ratio * (col - sigma));
        const a = f(f(xpos * xpos) + y2);
        kd[row * size + col] = f(f(2 - a) * f(Math.exp(a / 2)));
      }
    }
    const fre = keep(new cv.Mat());
    cv.filter2D(eq, fre, cv.CV_32F, kernel);
    return marrHildrethBits(fre.data32F);
  });
}

/** The block sums and bits of the Marr–Hildreth hash from the 512×512 filter response. */
export function marrHildrethBits(F) {
  const f = Math.fround;
  // blocks(row, col) = sum of freImg over Rect(x = row·16, y = col·16, 16, 16)
  const blocks = new Float32Array(31 * 31);
  for (let row = 0; row < 31; row++) {
    for (let col = 0; col < 31; col++) {
      let s = 0;
      for (let y = col * 16; y < col * 16 + 16; y++) for (let x = row * 16; x < row * 16 + 16; x++) s += F[y * 512 + x];
      blocks[row * 31 + col] = s;
    }
  }
  const out = new Uint8Array(72);
  let bit = 0;
  let byte = 0;
  for (let row = 0; row < 29; row += 4) {
    for (let col = 0; col < 29; col += 4) {
      let s = 0;
      for (let i = 0; i < 3; i++) for (let j = 0; j < 3; j++) s += blocks[(row + i) * 31 + col + j];
      const avg = f(s / 9);
      for (let i = 0; i < 3; i++) {
        for (let j = 0; j < 3; j++) {
          byte = ((byte << 1) | (blocks[(row + i) * 31 + col + j] > avg ? 1 : 0)) & 255;
          bit++;
          if (bit % 8 === 0) {
            out[bit / 8 - 1] = byte;
            byte = 0;
          }
        }
      }
    }
  }
  return out;
}

/**
 * RadialVarianceHash (De Roover et al. 2005, pHash's "radial" hash): the
 * variance of brightness along 180 lines through the image centre, one per
 * degree, reduced to 40 DCT coefficients scaled to bytes. Compared by peak
 * cross-correlation over circular shifts, which absorbs rotations.
 */
export function radialVarianceHash(cv, img, sigma = 1, lines = 180) {
  const input = withMats((keep) => {
    const gray = keep(new cv.Mat());
    cv.cvtColor(keep(rgbMat(cv, img)), gray, cv.COLOR_RGB2GRAY);
    const blur = keep(new cv.Mat());
    cv.GaussianBlur(gray, blur, new cv.Size(0, 0), sigma, sigma);
    return { data: Uint8Array.from(blur.data), rows: blur.rows, cols: blur.cols };
  });
  return radialVarianceFromBlurred(input.data, input.rows, input.cols, lines);
}

/** The radial projections, variances and DCT of the radial-variance hash, from the blurred grey image. */
export function radialVarianceFromBlurred(data, rows, cols, lines = 180) {
  const f = Math.fround;
  const D = Math.max(rows, cols);
  const proj = new Uint8Array(lines * D);
  const perLine = new Int32Array(lines);
  const roundOff = (v) => (v >= 0 ? 0.5 : -0.5);
  const offset = (len) => {
    const centre = Math.floor(len / 2);
    return Math.floor(centre + roundOff(centre));
  };
  const xOff = offset(cols);
  const yOff = offset(rows);
  const at = (r, c) => data[r * cols + c];
  // first half of the projections
  const quarter = Math.floor(lines / 4);
  for (let k = 0; k < quarter + 1; k++) {
    const theta = f(f(k * f(3.14159)) / lines);
    const alpha = f(Math.tan(theta));
    const two = Math.floor(lines / 2) - k;
    for (let x = 0; x < D; x++) {
      const y = f(alpha * (x - xOff));
      const yd = Math.floor(f(y + roundOff(y)));
      if (yd + yOff >= 0 && yd + yOff < rows && x < cols) {
        proj[k * D + x] = at(yd + yOff, x);
        perLine[k]++;
      }
      if (yd + xOff >= 0 && yd + xOff < cols && k !== quarter && x < rows) {
        proj[two * D + x] = at(x, yd + xOff);
        perLine[two]++;
      }
    }
  }
  // second half
  const init = Math.floor((3 * lines) / 4);
  for (let k = init, j = 0; k < lines; k++, j += 2) {
    const theta = f(f(k * f(3.14159)) / lines);
    const alpha = f(Math.tan(theta));
    for (let x = 0; x < D; x++) {
      const y = f(alpha * (x - xOff));
      const yd = Math.floor(f(y + roundOff(y)));
      if (yd + yOff >= 0 && yd + yOff < rows && x < cols) {
        proj[k * D + x] = at(yd + yOff, x);
        perLine[k]++;
      }
      if (yOff - yd >= 0 && yOff - yd < cols && 2 * yOff - x >= 0 && 2 * yOff - x < rows && k !== init) {
        proj[(k - j) * D + x] = at(-(x - yOff) + yOff, -yd + yOff);
        perLine[k - j]++;
      }
    }
  }
  // variance along each line, standardised over lines
  const feat = new Float64Array(lines);
  let sum = 0;
  let sumSq = 0;
  for (let k = 0; k < lines; k++) {
    let ls = 0;
    let lss = 0;
    const n = perLine[k] + 0.00001;
    for (let i = 0; i < D; i++) {
      const v = proj[k * D + i];
      ls += v;
      lss += v * v;
    }
    feat[k] = lss / n - (ls * ls) / (n * n);
    sum += feat[k];
    sumSq += feat[k] * feat[k];
  }
  const mean = sum / lines;
  const sd = Math.sqrt(sumSq / lines - (sum * sum) / (lines * lines));
  for (let i = 0; i < lines; i++) feat[i] = (feat[i] - mean) / sd;
  // 40 DCT coefficients, scaled to 0..255
  const temp = new Float64Array(40);
  let max = 0;
  let min = 0;
  for (let k = 0; k < 40; k++) {
    let s = 0;
    for (let n = 0; n < lines; n++) s += feat[n] * Math.cos((3.14159 * (2 * n + 1) * k) / (2 * lines));
    temp[k] = k === 0 ? s / Math.sqrt(lines) : (s * 1.4142135623730951) / Math.sqrt(lines);
    if (temp[k] > max) max = temp[k];
    else if (temp[k] < min) min = temp[k];
  }
  const range = max - min;
  return Uint8Array.from(temp, (v) => (range !== 0 ? Math.trunc((255 * (v - min)) / range) : 0));
}

/** OpenCV's RadialVarianceHash comparison: peak correlation over the 40 circular shifts (1 = identical). */
export function radialVarianceCorrelation(a, b) {
  const n = a.length;
  const stats = (h) => {
    let m = 0;
    for (const v of h) m += v;
    m /= n;
    let s = 0;
    for (const v of h) s += (v - m) ** 2;
    return [m, Math.sqrt(s / n)];
  };
  const [ma, sa] = stats(a);
  const [mb, sb] = stats(b);
  const x = Float32Array.from(a, (v) => v - ma);
  const y = Float32Array.from(b, (v) => v - mb);
  let best = Number.MIN_VALUE;
  for (let shift = 0; shift < n; shift++) {
    let dot = 0;
    for (let i = 0; i < n; i++) dot += x[i] * y[(i - shift + n) % n];
    best = Math.max(best, dot / n / (sa * sb + 1e-20));
  }
  return best;
}
