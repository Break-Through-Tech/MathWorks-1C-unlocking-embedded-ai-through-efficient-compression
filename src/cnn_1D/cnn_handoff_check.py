
from pathlib import Path

import numpy as np
import torch
from torch import nn
from sklearn.metrics import confusion_matrix, classification_report

from src.data import load_split
from src.cnn_1D.cnn_model import BearingCNN, CLASS_NAMES


def check_checkpoint(path, X_val, y_val):
    checkpoint = torch.load(
        path, map_location="cpu", weights_only=True
    )

    assert tuple(checkpoint["class_names"]) == CLASS_NAMES

    # Reconstruct the network from its saved weights.
    model = BearingCNN()
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    # Apply the exact normalization saved during training.
    mean = checkpoint["train_mean"]
    std = checkpoint["train_std"]

    X = np.ascontiguousarray(
        ((X_val - mean) / std)[:, None, :],
        dtype=np.float32
    )

    label_map = {name: i for i, name in enumerate(CLASS_NAMES)}
    targets = np.array(
        [label_map[str(label)] for label in y_val],
        dtype=np.int64
    )

    with torch.inference_mode():
        logits = model(torch.from_numpy(X))
        reference_logits = model(torch.from_numpy(X[:3]))
        validation_loss = nn.CrossEntropyLoss()(
            logits, torch.from_numpy(targets)
        ).item()

    assert logits.shape == (len(targets), 3)
    assert torch.isfinite(logits).all()

    # Stored reference was generated on GPU using exactly the first three
    # validation examples in a batch of three. Keep that batch boundary.
    saved_reference = checkpoint["reference_logits"].detach().cpu()
    assert saved_reference.shape == reference_logits.shape == (3, 3)
    max_logit_difference = (reference_logits - saved_reference).abs().max().item()
    torch.testing.assert_close(
        reference_logits, saved_reference, rtol=1e-4, atol=5e-4,
        msg=f"Reference-logit drift in {path.name}"
    )

    saved_val_loss = float(checkpoint["val_loss"])
    assert np.isclose(
        validation_loss, saved_val_loss, rtol=1e-3, atol=1e-4
    ), (
        f"Validation loss drift in {path.name}: "
        f"saved={saved_val_loss:.8g}, reloaded={validation_loss:.8g}"
    )

    predictions = logits.argmax(dim=1).numpy()
    accuracy = float(np.mean(predictions == targets))

    print(f"\nCheckpoint: {path.name}")
    print("Augmentation:", checkpoint["augmentation"])
    print("Selected epoch:", checkpoint["epoch"])
    print(f"Saved validation accuracy: {checkpoint['val_accuracy']:.2%}")
    print(f"Reloaded validation accuracy: {accuracy:.2%}")
    print(f"Reference logits: PASSED (max abs diff {max_logit_difference:.8g})")
    print(f"Saved validation loss: {saved_val_loss:.8g}")
    print(f"Reloaded validation loss: {validation_loss:.8g}")
    print("Validation loss comparison: PASSED")

    print("\nConfusion matrix:")
    print(confusion_matrix(targets, predictions, labels=[0, 1, 2]))

    print("\nPer-class metrics:")
    print(classification_report(
        targets, predictions,
        labels=[0, 1, 2],
        target_names=CLASS_NAMES,
        zero_division=0
    ))

    assert abs(accuracy - checkpoint["val_accuracy"]) < 1e-9, (
        f"Validation accuracy drift in {path.name}"
    )
    print("VALIDATION CHECK PASSED")


def main():
    X_val, y_val = load_split("val")

    for filename in (
        "cnn_rough_seed42.pth",
        "cnn_rough_aug_seed42.pth",
    ):
        check_checkpoint(
            Path("cnn_outputs") / filename,
            X_val, y_val
        )


if __name__ == "__main__":
    main()
