"""Convert Meta's SSCD TorchScript models to ONNX for onnxruntime-web.

Download the official weights first (MIT license):
    https://dl.fbaipublicfiles.com/sscd-copy-detection/sscd_disc_mixup.torchscript.pt
    https://dl.fbaipublicfiles.com/sscd-copy-detection/sscd_disc_large.torchscript.pt

    python export_sscd.py sscd_disc_mixup.torchscript.pt ../models/sscd_disc_mixup_explain.onnx --explain
    python export_sscd.py sscd_disc_large.torchscript.pt ../models/sscd_disc_large.onnx

With --explain the model also returns, for every location of the backbone's
feature map, that location's share of the (unnormalised) descriptor. SSCD
pools with GeM (p = 3): g_c = (mean_l x_cl^3)^(1/3), which can be written
exactly as a sum over locations, g_c = sum_l x_cl^3 / (N g_c^2). Pushing each
term through the final linear layer gives P_l with z = sum_l P_l + bias, so
the cosine similarity with another image's descriptor e_B splits into
    s = sum_l (e_B . P_l) / |z| + (e_B . bias) / |z|,
an additive map of where the copy evidence is (cf. Stylianou et al.,
"Visualizing Deep Similarity Networks", WACV 2019).
"""

import argparse
import os
import tempfile

import torch

from quantize_weights import quantize_weight_only


class SSCDExplain(torch.nn.Module):
    def __init__(self, ts):
        super().__init__()
        self.backbone = ts.backbone
        fc = getattr(ts.embeddings, "1")
        self.weight = torch.nn.Parameter(fc.weight.detach().clone())
        self.bias = torch.nn.Parameter(fc.bias.detach().clone())

    def forward(self, x):
        f = self.backbone(x).clamp(min=1e-6)  # [B, 2048, h, w]
        n = f.shape[2] * f.shape[3]
        f3 = f.pow(3)
        g = f3.mean(dim=[2, 3]).pow(1.0 / 3)  # GeM pooling
        z = torch.nn.functional.linear(g, self.weight, self.bias)
        znorm = torch.linalg.vector_norm(z, ord=2, dim=[1], keepdim=True).clamp(min=1e-12)
        embedding = z / znorm
        h = f3 / (n * g.pow(2))[:, :, None, None]  # sum over locations == g
        contrib = torch.einsum("oc,bchw->bohw", self.weight, h)  # [B, 512, h, w]
        return embedding, znorm, contrib


def export(torchscript_path, onnx_path, explain=False):
    ts = torch.jit.load(torchscript_path, map_location="cpu").eval()
    # Scripting (not tracing) keeps the TorchScript backbone inside the graph.
    model = torch.jit.script(SSCDExplain(ts).eval()) if explain else ts
    dummy = torch.randn(1, 3, 288, 352)
    with torch.no_grad():
        ref = ts(dummy)
        out = model(dummy)
    if explain:
        diff = (out[0] - ref).abs().max().item()
        print("embedding matches TorchScript model to", diff)
    else:
        print("output", tuple(ref.shape), "norm", float(ref.norm(dim=1)[0]))

    outputs = ["embedding", "znorm", "contrib"] if explain else ["embedding"]
    axes = {"input": {0: "batch", 2: "height", 3: "width"}, "embedding": {0: "batch"}}
    if explain:
        axes.update({"znorm": {0: "batch"}, "contrib": {0: "batch", 2: "fh", 3: "fw"}})
    with tempfile.TemporaryDirectory() as tmp:
        fp32_path = os.path.join(tmp, "fp32.onnx")
        torch.onnx.export(
            model,
            (dummy,),
            fp32_path,
            dynamo=False,
            opset_version=17,
            input_names=["input"],
            output_names=outputs,
            dynamic_axes=axes,
            do_constant_folding=True,
        )
        print("fp32 onnx", os.path.getsize(fp32_path) / 1e6, "MB")
        # The stem and first two stages are small but the most sensitive to
        # int8 rounding, so they stay fp32 (keeps cosine error below ~0.005).
        quantize_weight_only(fp32_path, onnx_path, keep=r"initial_block|blocks/0/|blocks/1/")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("torchscript")
    p.add_argument("onnx")
    p.add_argument("--explain", action="store_true", help="also output per-location contributions")
    a = p.parse_args()
    export(a.torchscript, a.onnx, a.explain)
