
import csv
import random
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from src.data import load_split
from src.cnn_1D.cnn_model import BearingCNN, CLASS_NAMES
from src.cnn_1D.cnn_run_paths import reserve_run_paths
from src.cnn_1D.cnn_augmentation import AugmentedTrainingDataset

SEED = 42
EPOCHS = 60
BATCH_SIZE = 32
LEARNING_RATE = 0.001

OUTPUT_DIR = Path("cnn_outputs")


def set_seed(seed):
    """Initialize random generators for reproducible runs."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def prepare_data(X, y, mean, std):
    """
    Apply clean-training global standardization and convert
    the supplied arrays into PyTorch-compatible tensors.
    """
    X = np.ascontiguousarray(
        ((X - mean) / std)[:, None, :],
        dtype=np.float32
    )

    label_map = {
        name: index for index, name in enumerate(CLASS_NAMES)
    }

    y = np.array(
        [label_map[str(label)] for label in y],
        dtype=np.int64
    )

    assert X.ndim == 3 and X.shape[1:] == (1, 5000)
    assert y.shape == (len(X),)
    assert np.isfinite(X).all()

    return TensorDataset(
        torch.from_numpy(X),
        torch.from_numpy(y)
    )


def evaluate(model, loader, device):
    """
    Calculate average cross-entropy loss and accuracy.

    Evaluation mode disables dropout and uses BatchNorm's
    stored running statistics rather than updating them.
    """
    model.eval()

    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    criterion = nn.CrossEntropyLoss(reduction="sum")

    with torch.inference_mode():
        for inputs, targets in loader:
            inputs = inputs.to(device)
            targets = targets.to(device)

            logits = model(inputs)
            loss = criterion(logits, targets)

            # argmax returns flat integer class predictions.
            predictions = logits.argmax(dim=1)

            assert predictions.shape == targets.shape

            total_loss += loss.item()
            total_correct += (
                predictions == targets
            ).sum().item()
            total_samples += targets.size(0)

    return (
        total_loss / total_samples,
        total_correct / total_samples
    )


def main():
    set_seed(SEED)
    OUTPUT_DIR.mkdir(exist_ok=True)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print("Device:", device)

    # Load the original splits. Do not use test data here.
    X_train, y_train = load_split("train")
    X_val, y_val = load_split("val")

    # Fit normalization ONLY using clean training data.
    mean = float(X_train.mean())
    std = float(X_train.std(ddof=0))

    assert np.isclose(mean, -0.1537873379955809, rtol=0, atol=1e-10)
    assert np.isclose(std, 1.2870536466273186, rtol=0, atol=1e-10)
    assert std > 0

    train_data = AugmentedTrainingDataset(
    X_train, y_train, mean, std, seed=SEED
    )
    val_data = prepare_data(X_val, y_val, mean, std)

    generator = torch.Generator().manual_seed(SEED)

    train_loader = DataLoader(
        train_data,
        batch_size=BATCH_SIZE,
        shuffle=True,
        generator=generator,
        num_workers=0
    )

    clean_train_data = prepare_data(
    X_train, y_train, mean, std
    )

    train_eval_loader = DataLoader(
    clean_train_data,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0
    )

    val_loader = DataLoader(
    val_data,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0
    )
    model = BearingCNN().to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE
    )

    criterion = nn.CrossEntropyLoss()

    # Fail before any training if this exact experiment already exists.
    checkpoint_path, history_path = reserve_run_paths(
        OUTPUT_DIR, model, optimizer,
        seed=SEED, augmentation="shift_p0.5_noise_p0.5_20dB",
        batch_size=BATCH_SIZE, epochs=EPOCHS
    )

    best_val_loss = float("inf")
    best_epoch = None
    history = []

    print("Training samples:", len(train_data))
    print("Validation samples:", len(val_data))
    print("Train mean:", mean)
    print("Train std:", std)
    print("Parameters:", sum(
        p.numel() for p in model.parameters()
    ))

    for epoch in range(1, EPOCHS + 1):
        model.train()

        # Training: forward pass, gradient calculation,
        # and optimizer update for every minibatch.
        for inputs, targets in train_loader:
            inputs = inputs.to(device)
            targets = targets.to(device)

            optimizer.zero_grad()
            logits = model(inputs)
            loss = criterion(logits, targets)
            loss.backward()
            optimizer.step()

        # Measure both splits in evaluation mode.
        train_loss, train_acc = evaluate(
            model, train_eval_loader, device
        )

        val_loss, val_acc = evaluate(
            model, val_loader, device
        )

        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "train_accuracy": train_acc,
            "val_loss": val_loss,
            "val_accuracy": val_acc
        })

        # Select the checkpoint using validation loss,
        # never training accuracy or test accuracy.
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch

            model.eval()

            with torch.inference_mode():
                reference_logits = model(
                    val_data.tensors[0][:3].to(device)
                ).cpu()

            torch.save({
                "model_state_dict": model.state_dict(),
                "epoch": epoch,
                "seed": SEED,
                "train_mean": mean,
                "train_std": std,
                "class_names": CLASS_NAMES,
                "learning_rate": LEARNING_RATE,
                "batch_size": BATCH_SIZE,
                "epoch_budget": EPOCHS,
                "augmentation": "shift_p0.5_noise_p0.5_20dB",
                "val_loss": val_loss,
                "val_accuracy": val_acc,
                "torch_version": str(torch.__version__),
                "reference_logits": reference_logits
            }, checkpoint_path)

        if epoch == 1 or epoch % 5 == 0:
            print(
                f"Epoch {epoch:02d}/{EPOCHS} | "
                f"Train Loss: {train_loss:.4f} | "
                f"Train Acc: {train_acc:.2%} | "
                f"Val Loss: {val_loss:.4f} | "
                f"Val Acc: {val_acc:.2%}"
            )

    # Save complete learning history, including epochs
    # that were not chosen as the best checkpoint.
    with history_path.open("w", newline="") as file:
        writer = csv.DictWriter(
            file, fieldnames=history[0].keys()
        )
        writer.writeheader()
        writer.writerows(history)

    # Reload the best checkpoint into a fresh model.
    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=True
    )

    restored = BearingCNN().to(device)
    restored.load_state_dict(checkpoint["model_state_dict"])
    restored.eval()

    with torch.inference_mode():
        actual_logits = restored(
            val_data.tensors[0][:3].to(device)
        ).cpu()

    torch.testing.assert_close(
        actual_logits.cpu(),
        checkpoint["reference_logits"].cpu(),
        rtol=1e-6, atol=1e-6
    )

    print("\nTraining completed.")
    print("Best validation epoch:", best_epoch)
    print("Best validation loss:", round(best_val_loss, 4))
    print("Checkpoint:", checkpoint_path)
    print("History:", history_path)
    print("Checkpoint reload: PASSED")


if __name__ == "__main__":
    main()
