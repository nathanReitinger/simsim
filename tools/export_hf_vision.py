"""Build the compressed CLIP and DINOv2 image encoders used by the site.

Starts from the full-precision ONNX exports published on Hugging Face
(pinned revisions below), then applies weight-only int8 compression while
keeping the few quantization-sensitive layers in fp32. The int8 files on the
Hub use dynamic activation quantization, which moved cosine similarities by
up to 0.06 in our tests; this recipe stays within ~0.005 of fp32.

    python export_hf_vision.py ../models
"""

import os
import sys
import urllib.request

from quantize_weights import quantize_weight_only

MODELS = {
    # openai/clip-vit-base-patch32 (MIT), exported by Xenova
    "clip_vit_b32_vision.onnx": (
        "https://huggingface.co/Xenova/clip-vit-base-patch32/resolve/"
        "d15189d7028b43f1d3e65039190477f6af591c2a/onnx/vision_model.onnx",
        r"patch_embedding",
    ),
    # facebook/dinov2-small (Apache-2.0), exported by onnx-community
    "dinov2_small.onnx": (
        "https://huggingface.co/onnx-community/dinov2-small/resolve/"
        "8b1f705a3a7f6f062f6bdd21986c1583d3ef105d/onnx/model.onnx",
        r"patch_embeddings|/layer\.0/",
    ),
}


def main(out_dir):
    for name, (url, keep) in MODELS.items():
        src = os.path.join(out_dir, name + ".fp32")
        if not os.path.exists(src):
            print("downloading", url)
            urllib.request.urlretrieve(url, src)
        quantize_weight_only(src, os.path.join(out_dir, name), keep=keep)
        os.remove(src)


if __name__ == "__main__":
    main(sys.argv[1])
