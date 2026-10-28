
import tempfile
from pathlib import Path

import torch
from torch import nn

from src.data import load_split
from src.cnn_1D.cnn_model import BearingCNN
from src.cnn_1D.cnn_project import (
    DEFAULT_CHECKPOINT, DEFAULT_LAYERS,
    build_projected_model, conv_svd, factorize_conv, layer_report,
    load_model, project_model, select_rank,
)
from src.cnn_1D.cnn_train import prepare_data


def val_inputs(checkpoint):
    X_val, y_val = load_split("val")
    dataset = prepare_data(X_val, y_val, checkpoint["train_mean"], checkpoint["train_std"])
    return dataset.tensors[0]


def all_convs(model):
    return [name for name, m in model.named_modules() if isinstance(m, nn.Conv1d)]


def main():
    torch.manual_seed(42)

    model, checkpoint = load_model(DEFAULT_CHECKPOINT)
    inputs = val_inputs(checkpoint)

    with torch.inference_mode():
        original_logits = model(inputs)

    # 1. The default layers are the two largest ones.
    rows, total = layer_report(model)
    assert total == 50819
    assert {rows[0]["name"], rows[1]["name"]} == set(DEFAULT_LAYERS)
    assert sum(row["params"] for row in rows[:2]) == 45216
    print("Largest layers:", [row["name"] for row in rows[:2]])

    # 2. At full rank every conv layer is reproduced exactly.
    full, ranks, _ = project_model(model, all_convs(model), keep=1.0, mode="rank")
    assert ranks == {"features.0": 15, "features.4": 32, "features.8": 64, "features.12": 96}

    with torch.inference_mode():
        full_logits = full(inputs)

    torch.testing.assert_close(full_logits, original_logits, rtol=1e-4, atol=1e-4)
    assert torch.equal(full_logits.argmax(1), original_logits.argmax(1))
    print("Full-rank projection reproduces logits: PASSED")

    # 3. Truncation error matches the discarded singular values
    #    (Eckart-Young: the SVD is the best possible low-rank fit).
    conv = model.get_submodule("features.12")
    _, S, _ = conv_svd(conv)
    for rank in (1, 10, 48, 95):
        split = factorize_conv(conv, rank)
        rebuilt = torch.einsum(
            "or,rik->oik", split[1].weight[:, :, 0].double(), split[0].weight.double()
        )
        error = (rebuilt - conv.weight.double()).norm()
        expected = S[rank:].pow(2).sum().sqrt()
        assert torch.isclose(error, expected, rtol=1e-4, atol=1e-5), (rank, error, expected)
    print("Truncation error matches discarded singular values: PASSED")

    # 4. Parameter counts match the hand calculation for conv3 + conv4.
    for keep, expected_params in ((0.5, 34947), (0.25, 20355), (0.1, 11651)):
        projected, _, summary = project_model(model, DEFAULT_LAYERS, keep=keep, mode="rank")
        params = sum(p.numel() for p in projected.parameters())
        assert params == expected_params, (keep, params)
        assert all(row["rank"] <= row["break_even_rank"] for row in summary)
    print("Parameter counts (50% / 25% / 10% rank): PASSED")

    # 5. Rank selection behaves sensibly in both modes.
    assert select_rank(S, 0.5, "rank") == 48
    assert select_rank(S, 0.001, "rank") == 1
    assert select_rank(S, 1.0, "energy") == len(S)
    energy = S.pow(2).cumsum(0) / S.pow(2).sum()
    previous = 0
    for keep in (0.5, 0.75, 0.9, 0.99):
        rank = select_rank(S, keep, "energy")
        assert energy[rank - 1] >= keep and (rank == 1 or energy[rank - 2] < keep)
        assert rank >= previous
        previous = rank
    print("Rank selection: PASSED")

    # 6. The projected model still trains: gradients reach the thin layers.
    projected, ranks, _ = project_model(model, DEFAULT_LAYERS, keep=0.25, mode="rank")
    projected.train()
    loss = nn.CrossEntropyLoss()(projected(inputs[:9]), torch.zeros(9, dtype=torch.long))
    loss.backward()
    for name in DEFAULT_LAYERS:
        for layer in projected.get_submodule(name):
            assert layer.weight.grad is not None
            assert torch.isfinite(layer.weight.grad).all()
            assert layer.weight.grad.abs().sum() > 0
    print("Backward pass through projected layers: PASSED")

    # 7. Saving and reloading rebuilds the same projected model.
    projected.eval()
    with torch.inference_mode():
        projected_logits = projected(inputs)

    with tempfile.TemporaryDirectory() as folder:
        path = Path(folder) / "projected.pth"
        torch.save({
            "model_state_dict": projected.state_dict(),
            "projection_ranks": ranks,
            "train_mean": checkpoint["train_mean"],
            "train_std": checkpoint["train_std"],
        }, path)

        restored, restored_checkpoint = load_model(path)
        assert restored_checkpoint["projection_ranks"] == ranks

    with torch.inference_mode():
        restored_logits = restored(inputs)

    torch.testing.assert_close(restored_logits, projected_logits, rtol=1e-6, atol=1e-6)
    print("Projected checkpoint reload: PASSED")

    # 8. Empty ranks gives the plain baseline architecture.
    assert repr(build_projected_model({})) == repr(BearingCNN())

    print("ALL PROJECTION TESTS PASSED")


if __name__ == "__main__":
    main()
