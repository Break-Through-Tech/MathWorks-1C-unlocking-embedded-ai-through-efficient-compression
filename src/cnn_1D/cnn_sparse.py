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

Usage:

    from src.cnn_1D.cnn_sparse import save_sparse, load_sparse

    save_sparse(model, checkpoint, "pruned.npz")
    model, meta = load_sparse("pruned.npz")
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
