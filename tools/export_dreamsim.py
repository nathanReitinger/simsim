"""Export DreamSim (Fu et al., NeurIPS 2023) to ONNX for onnxruntime-web.

DreamSim fine-tunes vision backbones with LoRA on NIGHTS, a dataset of human
judgments about which of two images looks more like a reference. We use the
single-branch OpenCLIP ViT-B/32 variant (95.3% agreement with humans on the
NIGHTS test set, against 96.2% for the 3x larger ensemble): the LoRA weights
are merged into the backbone, and the graph returns DreamSim's normalised
512-d embedding, so distance = 1 - cosine.

    pip install dreamsim  # or: git clone https://github.com/ssundaram21/dreamsim
    python export_dreamsim.py ../models/dreamsim_ocb32.onnx

Input: pixel_values, float32 [1, 3, 224, 224], RGB in [0, 1] (bicubic resize
to 224x224, no crop — DreamSim's own preprocessing). Code and weights: MIT.
"""

import os
import sys
import tempfile

import torch

from quantize_weights import quantize_weight_only


class Embed(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        return self.model.embed(x)


def main(out_path, cache_dir='./dreamsim_models'):
    from dreamsim import dreamsim

    model, _ = dreamsim(pretrained=True, dreamsim_type='open_clip_vitb32', device='cpu', cache_dir=cache_dir)
    model = model.merge_and_unload().eval()
    wrapped = Embed(model).eval()
    dummy = torch.rand(1, 3, 224, 224)
    with torch.no_grad():
        ref = wrapped(dummy)
    with tempfile.TemporaryDirectory() as tmp:
        fp32 = os.path.join(tmp, 'fp32.onnx')
        torch.onnx.export(wrapped, (dummy,), fp32, dynamo=False, opset_version=17, input_names=['pixel_values'], output_names=['embedding'], do_constant_folding=True)
        quantize_weight_only(fp32, out_path, keep=r'conv1')
    import onnxruntime as ort

    out = ort.InferenceSession(out_path).run(None, {'pixel_values': dummy.numpy()})[0]
    cos = torch.nn.functional.cosine_similarity(torch.tensor(out), ref).item()
    print(f'{out_path}: {os.path.getsize(out_path) / 1e6:.1f} MB, cosine to PyTorch {cos:.5f}')


if __name__ == '__main__':
    main(sys.argv[1], *(sys.argv[2:3]))
