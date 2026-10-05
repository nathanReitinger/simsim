"""Export LPIPS (AlexNet, v0.1) to ONNX for onnxruntime-web.

Requires `pip install lpips` (BSD-2-Clause). The model takes two RGB images of
the same size scaled to [-1, 1] and returns one distance per pair.

    python export_lpips.py ../models/lpips_alex.onnx
"""

import os
import sys
import tempfile

import lpips
import torch

from quantize_weights import quantize_weight_only


class LPIPSDistance(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.lpips = lpips.LPIPS(net="alex", version="0.1", verbose=False).eval()

    def forward(self, a, b):
        return self.lpips(a, b).reshape(-1)


def export(onnx_path):
    model = LPIPSDistance().eval()
    a = torch.rand(1, 3, 256, 256) * 2 - 1
    b = torch.rand(1, 3, 256, 256) * 2 - 1
    with tempfile.TemporaryDirectory() as tmp:
        fp32_path = os.path.join(tmp, "fp32.onnx")
        torch.onnx.export(
            model,
            (a, b),
            fp32_path,
            dynamo=False,
            opset_version=17,
            input_names=["a", "b"],
            output_names=["distance"],
            dynamic_axes={
                "a": {0: "batch", 2: "height", 3: "width"},
                "b": {0: "batch", 2: "height", 3: "width"},
                "distance": {0: "batch"},
            },
        )
        quantize_weight_only(fp32_path, onnx_path)


if __name__ == "__main__":
    export(sys.argv[1])
