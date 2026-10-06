"""Score the site's own ONNX models on the simsim mini-benchmark.

    pip install onnxruntime pillow numpy imagehash scikit-image opencv-python-headless
    python make_benchmark.py      # writes img/ and pairs.json next to this file
    python evaluate.py            # AUC, recall at 1% false alarms, per-edit recall
    python rectify.py             # SSCD after undoing the keypoint transform
    python partial.py             # tiles and evidence-crop strategies (slow)
"""
import json, os, sys, time
import numpy as np, onnxruntime as ort, imagehash
from PIL import Image

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'models') + os.sep
B = os.path.dirname(os.path.abspath(__file__))
pairs = json.load(open(os.path.join(B, 'pairs.json')))
IMN = (np.array([0.485, 0.456, 0.406]), np.array([0.229, 0.224, 0.225]))
HALF = (np.array([0.5] * 3), np.array([0.5] * 3))
CLIPN = (np.array([0.48145466, 0.4578275, 0.40821073]), np.array([0.26862954, 0.26130258, 0.27577711]))

def short_edge(im, s, resample):
    w, h = im.size
    nw, nh = (s, int(s * h / w)) if w <= h else (int(s * w / h), s)
    return im.resize((nw, nh), resample)
def ccrop(im, s):
    w, h = im.size; l, t = int(round((w - s) / 2)), int(round((h - s) / 2)); return im.crop((l, t, l + s, t + s))
def chw(im, norm):
    a = (np.asarray(im, np.float32) / 255 - norm[0]) / norm[1]; return a.transpose(2, 0, 1)[None].astype(np.float32)
def l2(v): v = np.asarray(v, np.float32).ravel(); return v / (np.linalg.norm(v) + 1e-12)

def session(path):
    o = ort.SessionOptions(); o.intra_op_num_threads = 8
    if isinstance(path, list):
        data = b''.join(open(p, 'rb').read() for p in path); return ort.InferenceSession(data, o)
    return ort.InferenceSession(path, o)

MODELS = {
    'sscd': (lambda: session(ROOT + 'sscd_disc_mixup_explain.onnx'), lambda s, im: l2(s.run(['embedding'], {'input': chw(short_edge(im, 288, Image.BILINEAR), IMN)})[0])),
    'sscdLarge': (lambda: session(ROOT + 'sscd_disc_large.onnx'), lambda s, im: l2(s.run(['embedding'], {'input': chw(ccrop(short_edge(im, 256, Image.BILINEAR), 224), HALF)})[0])),
    'dino': (lambda: session(ROOT + 'dinov2_small.onnx'), lambda s, im: l2(s.run(['last_hidden_state'], {'pixel_values': chw(ccrop(short_edge(im, 256, Image.BICUBIC), 224), IMN)})[0][0, 0, :384])),
    'clip': (lambda: session([ROOT + 'clip_vit_b32_vision.onnx.part1', ROOT + 'clip_vit_b32_vision.onnx.part2']), lambda s, im: l2(s.run(['image_embeds'], {'pixel_values': chw(ccrop(short_edge(im, 224, Image.BICUBIC), 224), CLIPN)})[0])),
}

def load(name): return Image.open(os.path.join(B, 'img', name + '.png')).convert('RGB')
names = sorted({p[0] for p in pairs} | {p[1] for p in pairs})
extra = sys.argv[1:]  # optional: python evaluate.py module.py  (module defines NAME, setup(), embed(state, im))
for path in extra:
    ns = dict(globals()); exec(open(path).read(), ns); MODELS[ns['NAME']] = (ns['setup'], ns['embed'])

def report(name, score_fn):
    s = np.array([score_fn(a, b) for a, b, _, _ in pairs]); y = np.array([p[2] for p in pairs]); kinds = [p[3] for p in pairs]
    pos, neg = s[y == 1], s[y == 0]
    auc = (pos[:, None] > neg[None, :]).mean() + 0.5 * (pos[:, None] == neg[None, :]).mean()
    thr = np.quantile(neg, 0.99)
    tpr = (pos > thr).mean()
    per = {}
    for t in sorted({k for k, l in zip(kinds, y) if l == 1}):
        m = np.array([k == t for k in kinds]) & (y == 1); per[t] = (s[m] > thr).mean()
    return auc, tpr, thr, per, pos, neg

results = {}
cache = {}
for mname, (setup, emb) in MODELS.items():
    t0 = time.time(); st = setup(); E = {n: emb(st, load(n)) for n in names}
    results[mname] = report(mname, lambda a, b: float(E[a] @ E[b])) + (time.time() - t0,)
H = {n: imagehash.phash(load(n)) for n in names}
results['phash'] = report('phash', lambda a, b: 1 - (H[a] - H[b]) / 64) + (0,)

ts = sorted({p[3] for p in pairs if p[2] == 1})
print(f"{'model':10s} {'AUC':>6s} {'TPR@1%':>7s} {'thr':>6s} {'neg99':>6s} " + ' '.join(f'{t[:7]:>7s}' for t in ts))
for m, (auc, tpr, thr, per, pos, neg, secs) in results.items():
    print(f"{m:10s} {auc:6.3f} {tpr:7.3f} {thr:6.3f} {np.quantile(neg,0.99):6.3f} " + ' '.join(f'{per[t]:7.2f}' for t in ts))
json.dump({m: {'auc': r[0], 'tpr1': r[1], 'thr': r[2], 'per': r[3], 'pos': r[4].tolist(), 'neg': r[5].tolist()} for m, r in results.items()}, open(os.path.join(B, 'results.json'), 'w'))
