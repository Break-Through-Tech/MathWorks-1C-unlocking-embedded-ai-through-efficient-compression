"""
src/cnn_1D/cnn_sparse.py

Sparse storage for pruned BearingCNN checkpoints.

A pruned model saved as a normal .pth keeps every zero as a 4-byte
float, so pruning never shrinks the file. This format stores only what
is needed to rebuild the weights:

    - the nonzero values of each weight tensor (float32)
    - a 1-bit-per-weight mask saying which positions were kept
    - the tensor's shape

and writes everything with np.savez_compressed. This is the file a
pruned model would actually be shipped as, so it is the one measured
for model size (measurement-guide.md: "Size is the file you would
actually ship").

Loading unpacks back to a normal dense BearingCNN, so this saves
storage and transfer size, not compute.

Pruning + quantization
----------------------
A plain INT8 file stores every weight, zeros included, so it loses the
pruning saving. save_sparse_int8 / load_sparse_int8 do the same trick for
a quantized model: only the nonzero int8 weights are stored, plus a
1-bit mask, the quantization scales/zero-points and the float biases.
The quantized model is rebuilt exactly (same int8 weights, same scales).

Usage:

    from src.cnn_1D.cnn_sparse import save_sparse, load_sparse

    save_sparse(model, checkpoint, "pruned.npz")
    model, meta = load_sparse("pruned.npz")

    from src.cnn_1D.cnn_sparse import save_sparse_int8, load_sparse_int8

    save_sparse_int8(qmodel, checkpoint, "pruned_int8.npz")
    qmodel, meta = load_sparse_int8("pruned_int8.npz")   # runs on qnnpack
"""

import json

import numpy as np
import torch

from src.cnn_1D.cnn_model import BearingCNN

FORMAT_VERSION = 1

# Checkpoint fields needed to run the model (normalization + labels).
META_KEYS = ("train_mean", "train_std", "class_names")


def save_sparse(model, checkpoint, path):
    """Save a pruned model's weights as nonzero values + bit mask."""
    values, masks, layout, dense = [], [], [], {}

    for name, tensor in model.state_dict().items():
        array = tensor.detach().cpu().numpy()

        if tensor.is_floating_point() and tensor.dim() > 0:
            flat = array.ravel()
            kept = flat != 0
            values.append(flat[kept].astype(np.float32))
            masks.append(kept)
            layout.append([name, list(array.shape)])
        else:
            # Integer buffers such as BatchNorm's num_batches_tracked.
            dense[name] = array.tolist()

    meta = {key: checkpoint[key] for key in META_KEYS}
    meta["class_names"] = list(meta["class_names"])
    meta["format_version"] = FORMAT_VERSION
    meta["layout"] = layout
    meta["dense"] = dense

    # Everything goes into three arrays: one zip entry per tensor would
    # cost more in archive headers than the pruned weights themselves.
    np.savez_compressed(
        path,
        values=np.concatenate(values),
        mask=np.packbits(np.concatenate(masks)),
        meta=np.array(json.dumps(meta)),
    )


def load_sparse(path):
    """Rebuild a dense BearingCNN from a file written by save_sparse."""
    with np.load(path, allow_pickle=False) as data:
        meta = json.loads(str(data["meta"]))
        if meta["format_version"] != FORMAT_VERSION:
            raise ValueError(f"Unsupported sparse format: {meta['format_version']}")

        sizes = [int(np.prod(shape)) for _, shape in meta["layout"]]
        kept = np.unpackbits(data["mask"], count=sum(sizes)).astype(bool)
        flat = np.zeros(sum(sizes), dtype=np.float32)
        flat[kept] = data["values"]

    state_dict = {}
    offset = 0
    for (name, shape), size in zip(meta["layout"], sizes):
        state_dict[name] = torch.from_numpy(flat[offset:offset + size].reshape(shape).copy())
        offset += size
    for name, value in meta["dense"].items():
        state_dict[name] = torch.tensor(value)

    model = BearingCNN()
    model.load_state_dict(state_dict)
    model.eval()

    meta["class_names"] = tuple(meta["class_names"])
    return model, meta


# ---------------------------------------------------------------------------
# Pruned + INT8 (quantized) models
# ---------------------------------------------------------------------------

INT8_FORMAT_VERSION = 1


def _int8_entries(qmodel):
    """The quantized model's state_dict with Linear's packed weights unpacked."""
    entries = {}
    for name, value in qmodel.state_dict().items():
        if name.endswith("_packed_params.dtype"):
            continue
        if name.endswith("_packed_params._packed_params"):
            prefix = name[: -len("_packed_params._packed_params")]
            entries[prefix + "weight"], entries[prefix + "bias"] = value
        else:
            entries[name] = value
    return entries


def save_sparse_int8(qmodel, checkpoint, path):
    """Save a quantized (pruned) model as nonzero int8 values + bit mask."""
    values, masks, layout, floats, float_layout, small = [], [], [], [], [], {}

    for name, tensor in _int8_entries(qmodel).items():
        if tensor.is_quantized:
            if tensor.qscheme() != torch.per_tensor_affine or tensor.q_zero_point() != 0:
                raise ValueError(f"{name}: only symmetric per-tensor int8 weights are supported")
            flat = tensor.int_repr().numpy().ravel().astype(np.int8)
            kept = flat != 0
            values.append(flat[kept])
            masks.append(kept)
            layout.append([name, list(tensor.shape), tensor.q_scale()])
        elif tensor.dim() > 0 and tensor.is_floating_point():
            floats.append(tensor.detach().numpy().ravel().astype(np.float32))
            float_layout.append([name, list(tensor.shape)])
        else:
            # Quantization scale / zero-point entries (tiny).
            small[name] = {"value": tensor.tolist(), "dtype": str(tensor.dtype).split(".")[-1]}

    meta = {key: checkpoint[key] for key in META_KEYS}
    meta["class_names"] = list(meta["class_names"])
    meta["format_version"] = INT8_FORMAT_VERSION
    meta["layout"] = layout
    meta["float_layout"] = float_layout
    meta["small"] = small

    np.savez_compressed(
        path,
        values=np.concatenate(values),
        mask=np.packbits(np.concatenate(masks)),
        floats=np.concatenate(floats),
        meta=np.array(json.dumps(meta)),
    )


def load_sparse_int8(path, backend="qnnpack"):
    """Rebuild the quantized BearingCNN from a file written by save_sparse_int8."""
    # Imported here: cnn_quantize pulls in data loading and pandas.
    from src.cnn_1D.cnn_quantize import QuantizableBearingCNN, make_qconfig
    from torch.ao import quantization as tq

    with np.load(path, allow_pickle=False) as data:
        meta = json.loads(str(data["meta"]))
        if meta["format_version"] != INT8_FORMAT_VERSION:
            raise ValueError(f"Unsupported int8 sparse format: {meta['format_version']}")
        sizes = [int(np.prod(shape)) for _, shape, _ in meta["layout"]]
        kept = np.unpackbits(data["mask"], count=sum(sizes)).astype(bool)
        flat = np.zeros(sum(sizes), dtype=np.int8)
        flat[kept] = data["values"]
        floats = data["floats"]

    torch.backends.quantized.engine = backend

    state = {}
    offset = 0
    for (name, shape, scale), size in zip(meta["layout"], sizes):
        ints = torch.from_numpy(flat[offset:offset + size].reshape(shape).copy())
        state[name] = torch._make_per_tensor_quantized_tensor(ints, scale, 0)
        offset += size
    offset = 0
    for name, shape in meta["float_layout"]:
        size = int(np.prod(shape))
        state[name] = torch.from_numpy(floats[offset:offset + size].reshape(shape).copy())
        offset += size
    for name, item in meta["small"].items():
        state[name] = torch.tensor(item["value"], dtype=getattr(torch, item["dtype"]))

    # Same module structure the quantizer produced (no calibration needed:
    # every scale is overwritten from the file).
    model = QuantizableBearingCNN(BearingCNN()).eval().fuse()
    model.qconfig = make_qconfig(backend, "histogram")
    tq.prepare(model, inplace=True)
    model(torch.zeros(1, 1, 5000))
    tq.convert(model, inplace=True)

    model.load_state_dict(state)
    model.eval()

    meta["class_names"] = tuple(meta["class_names"])
    return model, meta
