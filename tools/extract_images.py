"""Pull the embedded pictures out of teaching files (.ppt, .pptx, .pdf) or copy
plain images, so case pairs can be picked from them.

    pip install olefile pypdf pillow
    python extract_images.py harney.ppt out/harney

Legacy .ppt files keep their pictures in the OLE "Pictures" stream as raw
JPEG/PNG data; .pptx files keep them under ppt/media/; PDFs embed them as
image XObjects.
"""

import os
import re
import sys
import zipfile

from PIL import Image

JPEG = re.compile(rb"\xff\xd8\xff")
PNG_START = b"\x89PNG\r\n\x1a\n"
PNG_END = b"IEND\xaeB`\x82"


def carve(data):
    """Find complete JPEG and PNG files inside a byte string."""
    found = []
    for m in JPEG.finditer(data):
        end = data.find(b"\xff\xd9", m.start())
        while end != -1:
            blob = data[m.start() : end + 2]
            try:
                Image.open(__import__("io").BytesIO(blob)).load()
                found.append((".jpg", blob))
                break
            except Exception:
                end = data.find(b"\xff\xd9", end + 2)
    pos = 0
    while (start := data.find(PNG_START, pos)) != -1:
        end = data.find(PNG_END, start)
        if end == -1:
            break
        found.append((".png", data[start : end + len(PNG_END)]))
        pos = end
    return found


def extract(path, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    ext = os.path.splitext(path)[1].lower()
    blobs = []
    if ext == ".pptx":
        with zipfile.ZipFile(path) as z:
            for name in sorted(n for n in z.namelist() if n.startswith("ppt/media/")):
                blobs.append((os.path.splitext(name)[1].lower(), z.read(name)))
    elif ext == ".ppt":
        import olefile

        with olefile.OleFileIO(path) as ole:
            data = ole.openstream("Pictures").read() if ole.exists("Pictures") else open(path, "rb").read()
        blobs = carve(data)
    elif ext == ".pdf":
        from pypdf import PdfReader

        for page in PdfReader(path).pages:
            for im in page.images:
                blobs.append((os.path.splitext(im.name)[1].lower() or ".png", im.data))
    else:
        blobs = [(ext, open(path, "rb").read())]

    written = []
    for i, (e, blob) in enumerate(blobs, 1):
        target = os.path.join(out_dir, f"{i:02d}{e}")
        with open(target, "wb") as f:
            f.write(blob)
        try:
            with Image.open(target) as im:
                written.append((target, im.size))
        except Exception:
            written.append((target, None))
    return written


if __name__ == "__main__":
    for target, size in extract(sys.argv[1], sys.argv[2]):
        print(target, size)
