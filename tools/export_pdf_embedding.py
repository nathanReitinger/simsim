"""Export ICDiff's PDF-Embedding (Wang et al., NeurIPS 2024) to ONNX.

PDF-Embedding is a DeiT-B/16 with one class token and five extra "query"
tokens. Each of the six outputs corresponds to a replication level, from 5
(the generated image reproduces the original) down to 0 (no replication);
two images are compared token by token, and the level whose token agrees
best is the predicted replication level. Trained on D-Rep, 40,000 pairs of
real and Stable-Diffusion-generated images labelled by people.

    # weights (CC BY-NC 4.0): https://huggingface.co/datasets/WenhaoWang/D-Rep/resolve/main/vit_exp_563.pth.tar
    python export_pdf_embedding.py vit_exp_563.pth.tar ../models/pdf_embedding.onnx

Input: pixel_values, float32 [1, 3, 224, 224]: bicubic resize to 224x224
(no crop), ImageNet mean/std. Output: tokens [1, 6, 768].
"""

import os
import sys
import tempfile

from functools import partial

import torch
import torch.nn as nn
from timm.models.vision_transformer import VisionTransformer

from quantize_weights import quantize_weight_only


class QueryViT(VisionTransformer):
    def __init__(self):
        super().__init__(patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), num_classes=1000)
        self.dist_token = nn.Parameter(torch.zeros(1, 5, 768))
        self.pos_embed = nn.Parameter(torch.zeros(1, 196 + 6, 768))

    def forward(self, x):
        b = x.shape[0]
        x = self.patch_embed(x)
        x = torch.cat((self.cls_token.expand(b, -1, -1), self.dist_token.expand(b, -1, -1), x), dim=1)
        x = x + self.pos_embed
        for blk in self.blocks:
            x = blk(x)
        return self.norm(x)[:, :6]


def load(checkpoint):
    model = QueryViT().eval()
    state = torch.load(checkpoint, map_location='cpu')
    state = {k.replace('module.base.0.', ''): v for k, v in state.items() if k.startswith('module.base.0.')}
    missing, unexpected = model.load_state_dict(state, strict=False)
    missing = [k for k in missing if not k.startswith('head')]
    assert not missing and not [k for k in unexpected if not k.startswith('head')], (missing, unexpected)
    return model


def main(checkpoint, out_path):
    model = load(checkpoint)
    dummy = torch.randn(1, 3, 224, 224)
    with torch.no_grad():
        ref = model(dummy)
    with tempfile.TemporaryDirectory() as tmp:
        fp32 = os.path.join(tmp, 'fp32.onnx')
        torch.onnx.export(model, (dummy,), fp32, dynamo=False, opset_version=17, input_names=['pixel_values'], output_names=['tokens'], do_constant_folding=True)
        quantize_weight_only(fp32, out_path, keep=r'patch_embed')
    import onnxruntime as ort

    out = torch.tensor(ort.InferenceSession(out_path).run(None, {'pixel_values': dummy.numpy()})[0])
    cos = torch.nn.functional.cosine_similarity(out, ref, dim=2).min().item()
    print(f'{out_path}: {os.path.getsize(out_path) / 1e6:.1f} MB, worst token cosine to PyTorch {cos:.5f}')


if __name__ == '__main__':
    main(*sys.argv[1:3])
