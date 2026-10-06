"""SSCD after undoing B's geometry with a keypoint homography (as the site's ORB path does)."""
import json, os, sys
import numpy as np, cv2
sys.argv = [sys.argv[0]]
exec(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'evaluate.py')).read().split('results = {}')[0])
s = session(ROOT + 'sscd_disc_mixup_explain.onnx')
def emb(im): return l2(s.run(['embedding'], {'input': chw(short_edge(im, 288, Image.BILINEAR), IMN)})[0])
orb = cv2.ORB_create(3000)
def feats(im):
    g = cv2.cvtColor(np.asarray(im), cv2.COLOR_RGB2GRAY); sc = min(1, 800 / max(g.shape)); g = cv2.resize(g, None, fx=sc, fy=sc, interpolation=cv2.INTER_AREA)
    k, d = orb.detectAndCompute(g, None); return k, d, sc
F, E = {}, {}
def get(n):
    if n not in F: im = load(n); F[n] = feats(im); E[n] = emb(im)
    return F[n], E[n]
def sane(H, w, h):
    if H is None: return False
    det = H[0, 0] * H[1, 1] - H[0, 1] * H[1, 0]
    if not (1 / 50 < det < 50): return False
    if abs(H[2, 0]) * np.hypot(w, h) > 0.6 or abs(H[2, 1]) * np.hypot(w, h) > 0.6: return False
    c = cv2.perspectiveTransform(np.float32([[0, 0], [w, 0], [w, h], [0, h]])[None], H)[0]
    return cv2.isContourConvex(c.astype(np.float32))
def rectified(a, b):
    (ka, da, sa), ea = get(a); (kb, db, sb), eb = get(b)
    if da is None or db is None or len(ka) < 8 or len(kb) < 8: return None
    m = cv2.BFMatcher(cv2.NORM_HAMMING).knnMatch(db, da, k=2)
    good = [x for x in m if len(x) == 2 and x[0].distance < 0.8 * x[1].distance]
    if len(good) < 8: return None
    src = np.float32([kb[g[0].queryIdx].pt for g in good]) / sb; dst = np.float32([ka[g[0].trainIdx].pt for g in good]) / sa
    H, inl = cv2.findHomography(src, dst, cv2.RANSAC, 5.0)
    imA, imB = load(a), load(b)
    if H is None or inl.sum() < 20 or not sane(H, *imB.size): return None
    w, h = imA.size
    wb = cv2.warpPerspective(np.asarray(imB), H, (w, h), borderValue=(0, 0, 0))
    mask = cv2.warpPerspective(np.full(imB.size[::-1], 255, np.uint8), H, (w, h))
    ys, xs = np.nonzero(mask)
    if len(xs) < 0.05 * w * h: return None
    box = (xs.min(), ys.min(), xs.max() + 1, ys.max() + 1)
    A2 = imA.crop(box); B2 = Image.fromarray(wb).crop(box)
    # outside B's footprint, show A's pixels in neither: fill both with grey there
    mk = Image.fromarray(mask).crop(box)
    grey = Image.new('RGB', A2.size, (128, 128, 128))
    A2 = Image.composite(A2, grey, mk); B2 = Image.composite(B2, grey, mk)
    return float(emb(A2) @ emb(B2))
y = np.array([p[2] for p in pairs]); kinds = [p[3] for p in pairs]
glob, rect = [], []
for a, b, _, _ in pairs:
    g = float(get(a)[1] @ get(b)[1]); glob.append(g)
    r1 = rectified(a, b); r2 = rectified(b, a)
    rect.append(max([g] + [r for r in (r1, r2) if r is not None]))
def summarize(name, sc):
    sc = np.array(sc); pos, neg = sc[y == 1], sc[y == 0]; thr = np.quantile(neg, 0.99)
    ts = sorted({k for k, l in zip(kinds, y) if l == 1})
    print(f'{name:18s} TPR@1% {(pos > thr).mean():.3f} thr {thr:.3f} max-neg {neg.max():.3f} | recall@0.75 ' + ' '.join(f'{t[:6]}:{np.mean([v > 0.75 for v, k, l in zip(sc, kinds, y) if l and k == t]):.2f}' for t in ts))
summarize('global', glob); summarize('max(global,rect)', rect)
json.dump({'glob': glob, 'rect': rect}, open(os.path.join(B, 'rectify.json'), 'w'))
