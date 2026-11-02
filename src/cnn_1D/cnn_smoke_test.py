
import tempfile
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from torch import nn

from src.data import load_all_splits
from src.cnn_1D.cnn_model import BearingCNN, CLASS_NAMES


def main():
    torch.manual_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    # 1. Load and verify the provided dataset.
    splits = load_all_splits()

    expected = {
        "train": (393, 131),
        "val": (27, 9),
        "test": (102, 34),
    }

    for name, (X, y) in splits.items():
        total, per_class = expected[name]
        assert X.shape == (total, 5000)
        assert np.isfinite(X).all()
        assert Counter(map(str, y)) == {
            cls: per_class for cls in CLASS_NAMES
        }

    X_train, y_train = splits["train"]

    # 2. Compute clean-training normalization statistics.
    mean = float(X_train.mean())
    std = float(X_train.std(ddof=0))

    assert np.isclose(mean, -0.1537873379955809, rtol=0, atol=1e-10)
    assert np.isclose(std, 1.2870536466273186, rtol=0, atol=1e-10)

    print("Mean:", mean)
    print("Population std:", std)

    # 3. Select three training examples from each class.
    indices = np.concatenate([
        np.flatnonzero(y_train == cls)[:3]
        for cls in CLASS_NAMES
    ])

    X_small = (X_train[indices] - mean) / std

    # PyTorch Conv1d expects: (batch, channels, time).
    X_small = np.ascontiguousarray(
        X_small[:, None, :], dtype=np.float32
    )

    label_map = {name: i for i, name in enumerate(CLASS_NAMES)}
    y_small = np.array(
        [label_map[str(y_train[i])] for i in indices],
        dtype=np.int64,
    )

    inputs = torch.from_numpy(X_small).to(device)
    targets = torch.from_numpy(y_small).to(device)

    assert inputs.shape == (9, 1, 5000)
    assert targets.shape == (9,)
    print("Input shape:", tuple(inputs.shape))

    # 4. Create the CNN and verify its output.
    model = BearingCNN().to(device)

    parameters = sum(p.numel() for p in model.parameters())
    assert parameters == 50819

    logits = model(inputs)

    assert logits.shape == (9, 3)
    assert torch.isfinite(logits).all()

    print("Parameters:", parameters)
    print("Output shape:", tuple(logits.shape))

    # 5. Verify loss, gradients, and one optimizer update.
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    loss = criterion(logits, targets)
    assert torch.isfinite(loss)

    optimizer.zero_grad()
    loss.backward()

    gradients = [
        p.grad for p in model.parameters() if p.requires_grad
    ]

    assert all(g is not None for g in gradients)
    assert all(torch.isfinite(g).all() for g in gradients)
    assert any(torch.count_nonzero(g).item() > 0 for g in gradients)

    optimizer.step()
    print("Initial loss:", round(loss.item(), 4))
    print("Backward pass: PASSED")

    # 6. Verify that saving/reloading preserves predictions.
    model.eval()

    with tempfile.TemporaryDirectory() as folder:
        path = Path(folder) / "smoke_test.pth"
        torch.save(model.state_dict(), path)

        restored = BearingCNN().to(device)
        restored.load_state_dict(
            torch.load(path, map_location=device, weights_only=True)
        )
        restored.eval()

        with torch.inference_mode():
            original_output = model(inputs)
            restored_output = restored(inputs)

        torch.testing.assert_close(
            original_output, restored_output,
            rtol=1e-6, atol=1e-6
        )

    print("Checkpoint reload: PASSED")
    print("ALL SMOKE TESTS PASSED")


if __name__ == "__main__":
    main()
