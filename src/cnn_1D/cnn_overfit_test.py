
from collections import Counter

import numpy as np
import torch
from torch import nn

from src.data import load_split
from src.cnn_1D.cnn_model import BearingCNN, CLASS_NAMES


def main():
    # Make the test reproducible.
    torch.manual_seed(42)
    np.random.seed(42)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    print("Device:", device)

    # Load ONLY training data.
    X_train, y_train = load_split("train")

    # Fit global normalization using clean training data.
    mean = X_train.mean()
    std = X_train.std(ddof=0)

    # Select three windows from each bearing class.
    indices = np.concatenate([
        np.flatnonzero(y_train == cls)[:3]
        for cls in CLASS_NAMES
    ])

    X_small = (X_train[indices] - mean) / std

    # Conv1D requires (batch, channels, time).
    X_small = np.ascontiguousarray(
        X_small[:, None, :], dtype=np.float32
    )

    label_map = {
        name: index for index, name in enumerate(CLASS_NAMES)
    }

    y_small = np.array(
        [label_map[str(y_train[i])] for i in indices],
        dtype=np.int64
    )

    inputs = torch.from_numpy(X_small).to(device)
    targets = torch.from_numpy(y_small).to(device)

    assert Counter(y_small.tolist()) == {0: 3, 1: 3, 2: 3}

    model = BearingCNN().to(device)

    # Disable dropout for this memorization diagnostic.
    for layer in model.classifier:
        if isinstance(layer, nn.Dropout):
            layer.p = 0.0

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=0.003,
        weight_decay=0.0
    )

    max_steps = 500
    passed = False

    for step in range(1, max_steps + 1):
        model.train()

        # Forward pass.
        optimizer.zero_grad()
        logits = model(inputs)
        loss = criterion(logits, targets)

        # Backpropagation and weight update.
        loss.backward()
        optimizer.step()

        # Evaluate using inference mode.
        if step == 1 or step % 25 == 0:
            model.eval()

            with torch.inference_mode():
                outputs = model(inputs)
                eval_loss = criterion(outputs, targets)
                predictions = outputs.argmax(dim=1)

                correct = (
                    predictions == targets
                ).sum().item()

                accuracy = correct / len(targets)

            print(
                f"Step {step:3d} | "
                f"Loss: {eval_loss.item():.4f} | "
                f"Accuracy: {accuracy:.2%} "
                f"({correct}/9)"
            )

            # We want near-perfect memorization.
            if correct == 9 and eval_loss.item() < 0.05:
                passed = True
                break

    if passed:
        print("\nTINY-SUBSET OVERFIT TEST PASSED")
    else:
        print("\nTEST NOT YET PASSED — inspect training output")

    print("Actual labels:", targets.cpu().tolist())
    print("Predictions:", predictions.cpu().tolist())


if __name__ == "__main__":
    main()
