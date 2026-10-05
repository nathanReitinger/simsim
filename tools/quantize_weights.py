"""Weight-only int8 compression for ONNX models.

Every large Conv / Gemm / MatMul weight is stored as int8 with one fp32 scale
per output channel. A Cast + Mul pair rebuilds the fp32 tensor inside the
graph; onnxruntime constant-folds that pair when the session loads, so
inference runs in plain fp32. Accuracy stays very close to the original model
while the file is ~4x smaller. Works with any opset.

    python quantize_weights.py in.onnx out.onnx [--keep REGEX]

--keep is matched against the names of the nodes that consume each weight;
matching weights stay fp32 (useful for the few layers that are sensitive to
quantization, typically the first ones).
"""

import argparse
import re

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper


def quantize_weight_only(src, dst, min_elements=4096, keep=None):
    model = onnx.load(src)
    graph = model.graph
    inits = {init.name: init for init in graph.initializer}

    # weight name -> axis that indexes output channels
    targets = {}
    for node in graph.node:
        if len(node.input) < 2 or node.input[1] not in inits:
            continue
        name = node.input[1]
        if keep and re.search(keep, node.name):
            targets[name] = None
        elif node.op_type == "Conv":
            targets.setdefault(name, 0)
        elif node.op_type == "Gemm":
            trans_b = next((a.i for a in node.attribute if a.name == "transB"), 0)
            targets.setdefault(name, 0 if trans_b else 1)
        elif node.op_type == "MatMul":
            targets.setdefault(name, 1)

    new_nodes, new_inits, removed = [], [], set()
    saved = 0
    for name, axis in targets.items():
        if axis is None:
            continue
        w = numpy_helper.to_array(inits[name]).astype(np.float32)
        if w.size < min_elements or w.ndim < 2:
            continue
        reduce_axes = tuple(i for i in range(w.ndim) if i != axis)
        shape = [1] * w.ndim
        shape[axis] = -1
        scale = np.abs(w).max(axis=reduce_axes) / 127.0
        scale[scale == 0] = 1.0
        scale = scale.astype(np.float32).reshape(shape)
        q = np.clip(np.round(w / scale), -127, 127).astype(np.int8)

        new_inits += [
            numpy_helper.from_array(q, name + "__q"),
            numpy_helper.from_array(scale, name + "__scale"),
        ]
        new_nodes += [
            helper.make_node("Cast", [name + "__q"], [name + "__f"], to=TensorProto.FLOAT, name=name + "__cast"),
            helper.make_node("Mul", [name + "__f", name + "__scale"], [name], name=name + "__mul"),
        ]
        removed.add(name)
        saved += w.nbytes - q.nbytes

    kept = [i for i in graph.initializer if i.name not in removed]
    del graph.initializer[:]
    graph.initializer.extend(kept + new_inits)
    # Graph inputs that merely mirror initializers (old exporters) must go too.
    inputs = [i for i in graph.input if i.name not in removed]
    del graph.input[:]
    graph.input.extend(inputs)
    nodes = list(graph.node)
    del graph.node[:]
    graph.node.extend(new_nodes + nodes)

    onnx.checker.check_model(model)
    onnx.save(model, dst)
    print(f"{src} -> {dst}: {len(removed)} weights quantized, saved {saved / 1e6:.1f} MB")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("src")
    p.add_argument("dst")
    p.add_argument("--keep", default=None, help="regex of consumer node names whose weights stay fp32")
    a = p.parse_args()
    quantize_weight_only(a.src, a.dst, keep=a.keep)
