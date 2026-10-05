"""Convert Meta's SSCD TorchScript models to ONNX for onnxruntime-web.

Download the official weights first (MIT license):
    https://dl.fbaipublicfiles.com/sscd-copy-detection/sscd_disc_mixup.torchscript.pt
    https://dl.fbaipublicfiles.com/sscd-copy-detection/sscd_disc_large.torchscript.pt

    python export_sscd.py sscd_disc_mixup.torchscript.pt ../models/sscd_disc_mixup.onnx
"""

import os
import sys
import tempfile

import torch

from quantize_weights import quantize_weight_only


def export(torchscript_path, onnx_path):
    model = torch.jit.load(torchscript_path, map_location="cpu").eval()
    dummy = torch.randn(1, 3, 288, 352)
    with torch.no_grad():
        out = model(dummy)
    print("output", tuple(out.shape), "norm", float(out.norm(dim=1)[0]))

    with tempfile.TemporaryDirectory() as tmp:
        fp32_path = os.path.join(tmp, "fp32.onnx")
        torch.onnx.export(
            model,
            (dummy,),
            fp32_path,
            dynamo=False,
            opset_version=17,
            input_names=["input"],
            output_names=["embedding"],
            dynamic_axes={
                "input": {0: "batch", 2: "height", 3: "width"},
                "embedding": {0: "batch"},
            },
            do_constant_folding=True,
        )
        print("fp32 onnx", os.path.getsize(fp32_path) / 1e6, "MB")
        # The stem and first two stages are small but the most sensitive to
        # int8 rounding, so they stay fp32 (keeps cosine error below ~0.005).
        quantize_weight_only(fp32_path, onnx_path, keep=r"initial_block|blocks/0/|blocks/1/")


if __name__ == "__main__":
    export(sys.argv[1], sys.argv[2])
