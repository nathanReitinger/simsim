// Comparing edge maps: where the outlines and contours of two aligned
// images are, and how far apart. Canny edges and exact Euclidean distance
// transforms come from OpenCV.

/** Canny edges (Canny 1986) after a light blur: Uint8Array, 255 on edges. */
export function cannyEdges(cv, gray, w, h, lo = 50, hi = 150) {
  const src = new cv.Mat(h, w, cv.CV_8UC1);
  const blur = new cv.Mat();
  const out = new cv.Mat();
  try {
    src.data.set(gray);
    cv.GaussianBlur(src, blur, new cv.Size(3, 3), 0);
    cv.Canny(blur, out, lo, hi, 3, false);
    return Uint8Array.from(out.data);
  } finally {
    src.delete();
    blur.delete();
    out.delete();
  }
}

/** Exact Euclidean distance from every pixel to the nearest edge pixel. */
export function distanceToEdges(cv, edges, w, h) {
  const src = new cv.Mat(h, w, cv.CV_8UC1);
  const dst = new cv.Mat();
  try {
    for (let i = 0; i < edges.length; i++) src.data[i] = edges[i] ? 0 : 255;
    cv.distanceTransform(src, dst, cv.DIST_L2, cv.DIST_MASK_PRECISE);
    return Float64Array.from(dst.data32F);
  } finally {
    src.delete();
    dst.delete();
  }
}

/**
 * Pratt's figure of merit (Pratt 1978), made symmetric: each edge pixel of
 * one image scores 1 / (1 + d²/9) by its distance d to the other image's
 * nearest edge, normalised by the larger edge count; the two directions are
 * averaged. 1 = the same edges in the same places.
 */
export function prattFom(ea, eb, da, db, alpha = 1 / 9) {
  let na = 0;
  let nb = 0;
  let sa = 0;
  let sb = 0;
  for (let i = 0; i < ea.length; i++) {
    if (ea[i]) {
      na++;
      sa += 1 / (1 + alpha * db[i] * db[i]);
    }
    if (eb[i]) {
      nb++;
      sb += 1 / (1 + alpha * da[i] * da[i]);
    }
  }
  if (!na || !nb) return null;
  const n = Math.max(na, nb);
  return (sa / n + sb / n) / 2;
}

/**
 * Modified Hausdorff distance (Dubuisson & Jain 1994) between two edge sets:
 * the larger of the two mean distances from one set's pixels to the other
 * set, in pixels. 0 = identical edges.
 */
export function modifiedHausdorff(ea, eb, da, db) {
  let na = 0;
  let nb = 0;
  let sa = 0;
  let sb = 0;
  for (let i = 0; i < ea.length; i++) {
    if (ea[i]) {
      na++;
      sa += db[i];
    }
    if (eb[i]) {
      nb++;
      sb += da[i];
    }
  }
  if (!na || !nb) return null;
  return Math.max(sa / na, sb / nb);
}
