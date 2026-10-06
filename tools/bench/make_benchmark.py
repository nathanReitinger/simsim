"""simsim mini-benchmark: copy edits of the scikit-image sample images (see evaluate.py for usage)."""
import json, io, os, random
import numpy as np, cv2
from PIL import Image, ImageDraw, ImageFont, ImageFilter, ImageEnhance, ImageOps
import skimage.data as d

random.seed(7); np.random.seed(7)
OUT = os.path.join(os.path.dirname(__file__), 'img')
def rgb(a):
    a = np.asarray(a)
    if a.ndim == 2: a = np.stack([a] * 3, -1)
    if a.shape[2] == 4: a = a[:, :, :3]
    if a.dtype != np.uint8: a = (255 * (a - a.min()) / max(1e-9, a.max() - a.min())).astype(np.uint8)
    im = Image.fromarray(a)
    s = 640 / max(im.size)
    return im.resize((round(im.width * s), round(im.height * s)), Image.LANCZOS) if s < 1 else im
names = ['astronaut', 'camera', 'chelsea', 'coffee', 'rocket', 'hubble_deep_field', 'coins', 'moon', 'page', 'clock',
         'immunohistochemistry', 'retina', 'colorwheel', 'logo', 'brick', 'cell', 'grass', 'gravel']
bases = {n: rgb(getattr(d, n)()) for n in names}
bases['motorcycle'] = rgb(d.stereo_motorcycle()[0])
font = lambda size: ImageFont.load_default(size=size)

def caption(im, top, bottom):
    w, h = im.size; bar = int(h * 0.16)
    c = Image.new('RGB', (w, h + 2 * bar), 'white'); c.paste(im, (0, bar))
    dr = ImageDraw.Draw(c); f = font(max(12, bar // 2))
    for txt, y in ((top, bar // 4), (bottom, h + bar + bar // 4)):
        tw = dr.textlength(txt, font=f); dr.text(((w - tw) / 2, y), txt, fill='black', font=f)
    return c
def overlay(im):
    c = im.copy(); dr = ImageDraw.Draw(c); w, h = c.size; r = int(min(w, h) * 0.25)
    x, y = int(w * 0.62), int(h * 0.58)
    dr.ellipse([x - r, y - r, x + r, y + r], fill=(250, 210, 0)); dr.ellipse([x - r // 3 - 6, y - r // 3, x - r // 3 + 6, y - r // 3 + 12], fill='black')
    dr.ellipse([x + r // 3 - 6, y - r // 3, x + r // 3 + 6, y - r // 3 + 12], fill='black'); dr.arc([x - r // 2, y - r // 4, x + r // 2, y + r // 2], 20, 160, fill='black', width=6)
    return c
def frame(im):
    w, h = im.size; pad = int(0.12 * w); top = int(0.12 * h)
    c = Image.new('RGB', (w + 2 * pad, h + pad + top), (235, 236, 240)); c.paste(im, (pad, top))
    dr = ImageDraw.Draw(c); dr.rectangle([0, 0, c.width, top // 2], fill=(60, 64, 72)); dr.text((10, 4), 'Photos - Gallery', fill='white', font=font(max(10, top // 4)))
    return c.resize(im.size, Image.BILINEAR)
def persp(im):
    a = np.asarray(im); h, w = a.shape[:2]; j = 0.1
    src = np.float32([[0, 0], [w, 0], [w, h], [0, h]]); dst = np.float32([[w * j, h * j * 0.5], [w * (1 - j * 0.3), 0], [w, h * (1 - j)], [0, h]])
    return Image.fromarray(cv2.warpPerspective(a, cv2.getPerspectiveTransform(src, dst), (w, h), borderValue=(255, 255, 255)))
def crop_area(im, area, where='center'):
    w, h = im.size; k = area ** 0.5; cw, ch = int(w * k), int(h * k)
    x0, y0 = ((w - cw) // 2, (h - ch) // 2) if where == 'center' else (0, 0)
    return im.crop((x0, y0, x0 + cw, y0 + ch))
def jpeg(im, q):
    b = io.BytesIO(); im.save(b, 'JPEG', quality=q); return Image.open(io.BytesIO(b.getvalue())).convert('RGB')
def hue(im, shift=0.3, sat=1.4):
    hsv = np.asarray(im.convert('HSV')).astype(np.int32); hsv[..., 0] = (hsv[..., 0] + int(255 * shift)) % 256; hsv[..., 1] = np.clip(hsv[..., 1] * sat, 0, 255)
    return Image.fromarray(hsv.astype(np.uint8), 'HSV').convert('RGB')
order = list(bases)
def collage(name):
    bg = bases[order[(order.index(name) + 1) % len(order)]].copy(); fg = bases[name]
    s = 0.55; f = fg.resize((int(fg.width * s), int(fg.height * s))); bg = bg.resize((max(bg.width, f.width + 20), max(bg.height, f.height + 20)))
    bg.paste(f, (bg.width - f.width - 10, 10)); return bg, order[(order.index(name) + 1) % len(order)]
T = {
    'jpeg15': lambda im: jpeg(im, 15),
    'small40': lambda im: im.resize((max(8, int(im.width * 0.4)), max(8, int(im.height * 0.4))), Image.BILINEAR),
    'crop50': lambda im: crop_area(im, 0.5).resize(im.size, Image.BICUBIC),
    'corner30': lambda im: crop_area(im, 0.3, 'corner'),
    'flip': lambda im: ImageOps.mirror(im),
    'rot15': lambda im: im.rotate(15, expand=True, fillcolor='white'),
    'rot90': lambda im: im.rotate(90, expand=True),
    'hue': hue,
    'gray': lambda im: ImageOps.grayscale(im).convert('RGB'),
    'blur3': lambda im: im.filter(ImageFilter.GaussianBlur(3)),
    'noise': lambda im: Image.fromarray(np.clip(np.asarray(im).astype(np.float32) + np.random.normal(0, 20, np.asarray(im).shape), 0, 255).astype(np.uint8)),
    'meme': lambda im: caption(im, 'WHEN YOU COPY IT', 'BUT CALL IT ART'),
    'overlay': overlay,
    'frame': frame,
    'persp': persp,
    'pixelate': lambda im: im.resize((max(4, im.width // 8), max(4, im.height // 8)), Image.BILINEAR).resize(im.size, Image.NEAREST),
    'contrast': lambda im: ImageEnhance.Brightness(ImageEnhance.Contrast(im).enhance(1.6)).enhance(0.8),
    'combo': lambda im: caption(hue(ImageOps.mirror(crop_area(im, 0.6)), 0.15, 1.3), 'NOT A COPY', ''),
}
pairs = []
os.makedirs(OUT, exist_ok=True)
for n, im in bases.items():
    im.save(f'{OUT}/{n}.png')
    for t, f in T.items():
        f(im).convert('RGB').save(f'{OUT}/{n}__{t}.png'); pairs.append([n, f'{n}__{t}', 1, t])
    c, bgname = collage(n); c.save(f'{OUT}/{n}__collage.png'); pairs.append([n, f'{n}__collage', 1, 'collage'])
# negatives: other bases and their common edits (skip collage backgrounds, which do contain the base)
for n in bases:
    for m in bases:
        if m == n: continue
        for t in ['', '__jpeg15', '__crop50', '__flip', '__meme']:
            pairs.append([n, m + t, 0, 'neg' + (t or '__orig')])
        if collage(m)[1] != n: pairs.append([n, m + '__collage', 0, 'neg__collage'])
json.dump(pairs, open(os.path.join(os.path.dirname(__file__), 'pairs.json'), 'w'))
print(len(bases), 'bases;', sum(p[2] for p in pairs), 'positives;', sum(1 - p[2] for p in pairs), 'negatives')
