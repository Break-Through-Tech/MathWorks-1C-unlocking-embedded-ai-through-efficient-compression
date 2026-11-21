"""
src/cnn_1D/combined_robustness.py

Issue #30: Gurbaj's validation robustness suite (20 perturbed versions of
the 27 validation windows: 4 time shifts + 5 SNR levels x 3 noise draws +
clean) on the pruned, quantized, combined and micro models.

Reuses build_conditions() / evaluate_conditions() from projection_robustness.py,
so the perturbed windows are identical to Gurbaj's. The dense reference is
run first and must reproduce his recorded numbers. Validation only.

Writes results/combined_robustness.csv (same columns as
results/projection_robustness.csv, plus `compression`).

Usage (from the repository root):

    python -m src.cnn_1D.combined_robustness
"""

import csv
from pathlib import Path

import numpy as np
import torch

from src.cnn_1D.cnn_model import CLASS_NAMES, BearingCNN
from src.cnn_1D.cnn_sparse import load_sparse
from src.cnn_1D.combine_projection_quantize import load_projected_int8
from src.cnn_1D.projection_robustness import (
    SNR_LEVELS, build_conditions, check_against_reference, describe, evaluate_conditions,
)
from src.cnn_1D.cnn_quantize import pick_backend
from src.metrics import measure_model_size_mb

OUTPUT_CSV = Path("results/combined_robustness.csv")

REFERENCE = Path("cnn_outputs/final_models/original_reference_seed42.pth")
MICRO = Path("cnn_outputs/final_models/micro_selected_seed1337.pth")
INT8_DIR = Path("cnn_outputs/quantized/static_int8_histogram_qnnpack")
PRUNED_DIR = Path("cnn_outputs/pruned")


def load_dense(path):
    ckpt = torch.load(path, map_location="cpu", weights_only=True)
    model = BearingCNN(channels=tuple(ckpt.get("channels", (16, 32, 64, 96))),
                       dropout=ckpt.get("dropout", 0.20))
    model.load_state_dict(ckpt["model_state_dict"])
    return model.eval(), ckpt


def load_int8_scripted(path):
    torch.backends.quantized.engine = "qnnpack"
    ref = torch.load(REFERENCE, map_location="cpu", weights_only=True)
    return torch.jit.load(str(path)).eval(), ref   # same train mean/std as the reference


def models():
    """(label, compression, file measured for size, loader)."""
    yield "Dense reference", "None", REFERENCE, lambda: load_dense(REFERENCE)
    for pct in (90, 95, 97):
        path = PRUNED_DIR / f"unstructured_{pct}pct" / "model_sparse.npz"
        yield f"Pruned {pct}%", "Pruning", path, lambda p=path: load_sparse(p)
    path = INT8_DIR / "int8_scripted.pt"
    yield "INT8 (Venya)", "Quantization", INT8_DIR / "int8_state_dict.pth", lambda: load_int8_scripted(path)
    path = Path("cnn_outputs/combined/pruned97_int8/model_sparse_int8.npz")
    from src.cnn_1D.cnn_sparse import load_sparse_int8
    yield "Pruned 97% + INT8", "Pruning + Quantization", path, lambda: load_sparse_int8(path)
    path = Path("cnn_outputs/combined/projected10_int8/model_int8.pth")
    yield "Projected 10% + INT8", "Projection + Quantization", path, lambda: load_projected_int8(path)
    yield "Micro", "None", MICRO, lambda: load_dense(MICRO)
    path = Path("cnn_outputs/combined/micro_int8/model_int8.pth")
    yield "Micro + INT8", "Quantization", path, lambda: load_projected_int8(path)


def main():
    torch.set_num_threads(1)
    pick_backend()
    conditions, y = build_conditions()
    label_map = {name: i for i, name in enumerate(CLASS_NAMES)}
    targets = torch.tensor([label_map[label] for label in y], dtype=torch.long)

    rows = []
    for label, compression, path, loader in models():
        model, ckpt = loader()
        results = evaluate_conditions(model, ckpt, conditions, targets)
        if label == "Dense reference":
            check_against_reference("none", results)
            print("Dense reference reproduces Gurbaj's recorded results: PASSED")
        for name, (correct, loss) in sorted(results.items()):
            kind, snr, realization, shift = describe(name)
            rows.append({
                "model": label, "compression": compression, "artifact": path.as_posix(),
                "size_mb": measure_model_size_mb(path=path),
                "condition": name, "kind": kind, "snr_db": snr, "realization": realization,
                "shift": shift, "samples": len(y), "correct": correct,
                "accuracy": correct / len(y), "cross_entropy": loss,
            })

    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_CSV.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    print("\nMean accuracy over 3 noise draws (27 validation windows each)")
    print(f"{'model':<22}{'size MB':>9}{'clean':>7} " + " ".join(f"{s:>2}dB " for s in SNR_LEVELS) + "  worst shift")
    for label, _, path, _ in models():
        sub = [r for r in rows if r["model"] == label]
        clean = next(r["accuracy"] for r in sub if r["kind"] == "clean")
        noise = [np.mean([r["accuracy"] for r in sub if r["snr_db"] == s]) for s in SNR_LEVELS]
        shifts = min(r["accuracy"] for r in sub if r["kind"] == "shift")
        print(f"{label:<22}{sub[0]['size_mb']:>9.4f}{clean:>7.0%} " + " ".join(f"{a:>5.0%}" for a in noise) + f"  {shifts:>8.0%}")
    print(f"\nWrote {len(rows)} rows to {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
