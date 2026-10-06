"""Partial-copy strategies for SSCD: tiles (max over crops) vs evidence-guided crop."""
import json, os, sys
import numpy as np
sys.argv = [sys.argv[0]]
exec(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'evaluate.py')).read().split('results = {}')[0])
s = session(ROOT + 'sscd_disc_mixup_explain.onnx')
def run(im):
    e, zn, P = s.run(['embedding', 'znorm', 'contrib'], {'input': chw(short_edge(im, 288, Image.BILINEAR), IMN)})
    return l2(e), float(zn.ravel()[0]), P[0]  # P: 512 x fh x fw
cache = {}
def get(n):
    if n not in cache: cache[n] = (load(n),) + run(load(n))
    return cache[n]
def emb(im): return run(im)[0]
def tiles(im):
    w, h = im.size; out = []
    for k in (0.5, 0.7):
        tw, th = int(w * k), int(h * k)
        for fx in (0, 0.5, 1):
            for fy in (0, 0.5, 1):
                x, y = int((w - tw) * fx), int((h - th) * fy); out.append(im.crop((x, y, x + tw, y + th)))
    return out
tcache = {}
def tile_embs(n):
    if n not in tcache: tcache[n] = [emb(t) for t in tiles(load(n))]
    return tcache[n]
def evidence_crop(nA, nB):
    """Crop B to where its evidence (pairing with A) concentrates, re-embed, compare."""
    imA, eA, zA, PA = get(nA); imB, eB, zB, PB = get(nB)
    d, fh, fw = PB.shape
    ev = (eA @ PB.reshape(d, -1)) / zB  # per-cell share of cos(eA, eB) on B's side
    if ev.max() <= 0: return -1
    order = np.argsort(-ev); pos = ev[ev > 0].sum(); acc = 0; cells = []
    for i in order:
        if ev[i] <= 0 or acc >= 0.7 * pos: break
        acc += ev[i]; cells.append(i)
    ys, xs = np.array(cells) // fw, np.array(cells) % fw
    x0, x1 = max(0, xs.min() - 1), min(fw, xs.max() + 2); y0, y1 = max(0, ys.min() - 1), min(fh, ys.max() + 2)
    w, h = imB.size
    box = (int(x0 / fw * w), int(y0 / fh * h), int(x1 / fw * w), int(y1 / fh * h))
    if (box[2] - box[0]) * (box[3] - box[1]) > 0.9 * w * h: return float(eA @ eB)
    return float(eA @ emb(imB.crop(box)))
y = np.array([p[2] for p in pairs]); kinds = [p[3] for p in pairs]
def summarize(name, sc):
    sc = np.array(sc); pos, neg = sc[y == 1], sc[y == 0]; thr = np.quantile(neg, 0.99)
    auc = (pos[:, None] > neg[None, :]).mean()
    col = np.array([k == 'collage' for k in kinds]) & (y == 1); cor = np.array([k == 'corner30' for k in kinds]) & (y == 1)
    print(f'{name:16s} AUC {auc:.4f} TPR@1% {(pos > thr).mean():.3f} thr {thr:.3f} collage {(sc[col] > thr).mean():.2f} corner30 {(sc[cor] > thr).mean():.2f} | median pos {np.median(pos):.3f} neg99 {thr:.3f}')
glob = [float(get(a)[1] @ get(b)[1]) for a, b, _, _ in pairs]
summarize('global', glob)
tile = [max(glob[i], max(float(get(a)[1] @ t) for t in tile_embs(b)), max(float(get(b)[1] @ t) for t in tile_embs(a))) for i, (a, b, _, _) in enumerate(pairs)]
summarize('max(global,tiles)', tile)
evc = [max(glob[i], evidence_crop(a, b), evidence_crop(b, a)) for i, (a, b, _, _) in enumerate(pairs)]
summarize('max(global,evcrop)', evc)
