// Local-feature matching and template search with OpenCV.js.
// Every OpenCV object lives in WebAssembly memory and must be deleted.

function grayMat(cv, img) {
  const m = new cv.Mat(img.h, img.w, cv.CV_8UC1);
  m.data.set(img.gray);
  return m;
}

function makeDetector(cv, kind) {
  if (kind === 'orb') return new cv.ORB(3000);
  if (kind === 'akaze') return new cv.AKAZE();
  if (kind === 'brisk') return new cv.BRISK();
  throw new Error(`unknown detector ${kind}`);
}

/**
 * A homography is plausible for "B is an edited copy of A" when it keeps
 * orientation, does not collapse or explode areas, and is close to affine.
 */
export function homographySane(H, wB, hB) {
  if (!H) return false;
  const det = H[0] * H[4] - H[1] * H[3];
  if (!(det > 1 / 50 && det < 50)) return false;
  const diag = Math.hypot(wB, hB);
  if (Math.abs(H[6]) * diag > 0.6 || Math.abs(H[7]) * diag > 0.6) return false;
  const corners = [
    [0, 0],
    [wB, 0],
    [wB, hB],
    [0, hB],
  ].map(([x, y]) => {
    const z = H[6] * x + H[7] * y + H[8];
    return [(H[0] * x + H[1] * y + H[2]) / z, (H[3] * x + H[4] * y + H[5]) / z];
  });
  // projected quadrilateral must stay convex with consistent winding
  let sign = 0;
  for (let i = 0; i < 4; i++) {
    const [a, b, c] = [corners[i], corners[(i + 1) % 4], corners[(i + 2) % 4]];
    const cross = (b[0] - a[0]) * (c[1] - b[1]) - (b[1] - a[1]) * (c[0] - b[0]);
    if (cross === 0) return false;
    if (sign === 0) sign = Math.sign(cross);
    else if (Math.sign(cross) !== sign) return false;
  }
  return true;
}

/**
 * Detect keypoints in both images, match descriptors (Lowe ratio test) and fit
 * a homography B -> A with RANSAC. A and B are { w, h, gray }.
 */
export function matchFeatures(cv, A, B, kind) {
  const del = [];
  const keep = (o) => (del.push(o), o);
  try {
    const mA = keep(grayMat(cv, A));
    const mB = keep(grayMat(cv, B));
    const det = keep(makeDetector(cv, kind));
    const none = keep(new cv.Mat());
    const kA = keep(new cv.KeyPointVector());
    const kB = keep(new cv.KeyPointVector());
    const dA = keep(new cv.Mat());
    const dB = keep(new cv.Mat());
    det.detectAndCompute(mA, none, kA, dA);
    det.detectAndCompute(mB, none, kB, dB);
    const res = { keypointsA: kA.size(), keypointsB: kB.size(), good: 0, inliers: 0, H: null, sane: false, matches: [] };
    if (dA.rows < 2 || dB.rows < 2) return res;

    const bf = keep(new cv.BFMatcher(cv.NORM_HAMMING, false));
    const knn = keep(new cv.DMatchVectorVector());
    bf.knnMatch(dB, dA, knn, 2);
    const src = [];
    const dst = [];
    for (let i = 0; i < knn.size(); i++) {
      const pair = knn.get(i);
      if (pair.size() >= 2) {
        const m = pair.get(0);
        const n = pair.get(1);
        if (m.distance < 0.8 * n.distance) {
          const pb = kB.get(m.queryIdx).pt;
          const pa = kA.get(m.trainIdx).pt;
          src.push(pb.x, pb.y);
          dst.push(pa.x, pa.y);
        }
      }
      pair.delete();
    }
    res.good = src.length / 2;
    if (res.good < 8) return res;

    const srcM = keep(cv.matFromArray(res.good, 1, cv.CV_32FC2, src));
    const dstM = keep(cv.matFromArray(res.good, 1, cv.CV_32FC2, dst));
    const mask = keep(new cv.Mat());
    const H = keep(cv.findHomography(srcM, dstM, cv.RANSAC, 5.0, mask));
    if (H.empty()) return res;
    res.H = Array.from(H.data64F);
    for (let i = 0; i < res.good; i++) {
      if (mask.data[i]) {
        res.inliers++;
        res.matches.push([src[2 * i], src[2 * i + 1], dst[2 * i], dst[2 * i + 1]]);
      }
    }
    res.sane = homographySane(res.H, B.w, B.h);
    return res;
  } finally {
    for (const o of del) o.delete();
  }
}

/**
 * Is the smaller image contained in the larger one at the same scale?
 * Inputs are grayscale images downscaled by the same factor.
 */
export function templateSearch(cv, big, small) {
  const del = [];
  const keep = (o) => (del.push(o), o);
  try {
    const img = keep(grayMat(cv, big));
    const tpl = keep(grayMat(cv, small));
    const out = keep(new cv.Mat());
    cv.matchTemplate(img, tpl, out, cv.TM_CCOEFF_NORMED);
    const mm = cv.minMaxLoc(out);
    return { score: mm.maxVal, x: mm.maxLoc.x, y: mm.maxLoc.y };
  } finally {
    for (const o of del) o.delete();
  }
}

/**
 * Warp an RGB image { w, h, rgb } into a w×h frame with homography H.
 * Returns the warped image and a mask of pixels that came from the source.
 */
export function warpRgbInto(cv, img, H, w, h) {
  const del = [];
  const keep = (o) => (del.push(o), o);
  try {
    const src = keep(new cv.Mat(img.h, img.w, cv.CV_8UC3));
    src.data.set(img.rgb);
    const Hm = keep(cv.matFromArray(3, 3, cv.CV_64F, H));
    const dsize = new cv.Size(w, h);
    const dst = keep(new cv.Mat());
    cv.warpPerspective(src, dst, Hm, dsize, cv.INTER_LINEAR, cv.BORDER_CONSTANT, new cv.Scalar(0, 0, 0, 0));
    const ones = keep(new cv.Mat(img.h, img.w, cv.CV_8UC1, new cv.Scalar(255)));
    const mask = keep(new cv.Mat());
    cv.warpPerspective(ones, mask, Hm, dsize, cv.INTER_NEAREST, cv.BORDER_CONSTANT, new cv.Scalar(0));
    const k = keep(cv.Mat.ones(3, 3, cv.CV_8U));
    cv.erode(mask, mask, k);
    return { img: { w, h, rgb: new Uint8Array(dst.data) }, mask: new Uint8Array(mask.data) };
  } finally {
    for (const o of del) o.delete();
  }
}

/**
 * Warp image B (gray) into the frame of A using homography H (B -> A coords).
 * Returns the warped plane and a validity mask eroded by `erode` pixels.
 */
export function warpInto(cv, B, H, w, h, erode = 5) {
  const del = [];
  const keep = (o) => (del.push(o), o);
  try {
    const src = keep(grayMat(cv, B));
    const Hm = keep(cv.matFromArray(3, 3, cv.CV_64F, H));
    const dsize = new cv.Size(w, h);
    const warped = keep(new cv.Mat());
    cv.warpPerspective(src, warped, Hm, dsize, cv.INTER_LINEAR, cv.BORDER_CONSTANT, new cv.Scalar(0));
    const ones = keep(new cv.Mat(B.h, B.w, cv.CV_8UC1, new cv.Scalar(255)));
    const mask = keep(new cv.Mat());
    cv.warpPerspective(ones, mask, Hm, dsize, cv.INTER_NEAREST, cv.BORDER_CONSTANT, new cv.Scalar(0));
    if (erode > 0) {
      const k = keep(cv.Mat.ones(2 * erode + 1, 2 * erode + 1, cv.CV_8U));
      cv.erode(mask, mask, k);
    }
    return { gray: new Uint8Array(warped.data), mask: new Uint8Array(mask.data) };
  } finally {
    for (const o of del) o.delete();
  }
}
