"""Export the XFeat backbone (Potje et al., CVPR 2024) to ONNX for onnxruntime-web.

    git clone https://github.com/verlab/accelerated_features   (Apache-2.0)
    python export_xfeat.py path/to/accelerated_features ../models/xfeat.onnx

Input: image, float32 [1, 3, H, W] with H and W multiples of 32 (any value
range: the network averages the channels and applies instance norm).
Outputs at 1/8 resolution: feats [1, 64, H/8, W/8] (descriptors before L2
normalisation), keypoints [1, 65, H/8, W/8] (logits: 64 positions in each
8x8 cell + "no keypoint"), heatmap [1, 1, H/8, W/8] (reliability). The
keypoint selection, NMS and descriptor sampling run in js/lib/xfeat.js.
"""

import os
import sys

import torch


def main(repo, out_path):
    sys.path.insert(0, repo)
    from modules.model import XFeatModel

    def unfold8(self, x, ws=8):
        # same as XFeatModel._unfold2d (channel = dy * ws + dx), but with
        # reshapes that export with dynamic image sizes
        b, c, h, w = x.shape
        x = x.reshape(b, c, h // ws, ws, w // ws, ws).permute(0, 1, 3, 5, 2, 4)
        return x.reshape(b, c * ws * ws, h // ws, w // ws)

    XFeatModel._unfold2d = unfold8
    net = XFeatModel().eval()
    net.load_state_dict(torch.load(os.path.join(repo, 'weights', 'xfeat.pt'), map_location='cpu'))
    dummy = torch.rand(1, 3, 480, 640) * 255
    torch.onnx.export(
        net,
        (dummy,),
        out_path,
        dynamo=False,
        opset_version=17,
        input_names=['image'],
        output_names=['feats', 'keypoints', 'heatmap'],
        dynamic_axes={'image': {2: 'height', 3: 'width'}, 'feats': {2: 'h8', 3: 'w8'}, 'keypoints': {2: 'h8', 3: 'w8'}, 'heatmap': {2: 'h8', 3: 'w8'}},
        do_constant_folding=True,
    )
    import numpy as np
    import onnxruntime as ort

    x = torch.rand(1, 3, 352, 512) * 255
    with torch.no_grad():
        ref = net(x)
    out = ort.InferenceSession(out_path).run(None, {'image': x.numpy()})
    err = max(float(np.abs(o - r.numpy()).max()) for o, r in zip(out, ref))
    print(f'{out_path}: {os.path.getsize(out_path) / 1e6:.2f} MB, max abs diff vs PyTorch at 352x512: {err:.2e}')


if __name__ == '__main__':
    main(*sys.argv[1:3])
