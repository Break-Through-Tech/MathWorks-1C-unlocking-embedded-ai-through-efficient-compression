# load the model
import numpy as np
import torch
from src.metrics import accuracy, measure_model_size_mb, measure_inference_latency_ms

from src.data import load_split
from src.cnn_1D.cnn_model import BearingCNN
from pathlib import Path

# Define prediction function.
def predict_fn(X):
    with torch.inference_mode():
        logits = model(torch.from_numpy(X))
        y_pred = logits.argmax(dim=1).tolist()
        class_names = checkpoint["class_names"]
        return [class_names[i] for i in y_pred[:]]

# 1. Load the checkpoint and reconstruct the model.
folder = Path(".")
pth_files = list(folder.rglob("*.pth"))
for elem in pth_files:

    checkpoint = torch.load(
        elem,
        map_location="cpu",
        weights_only=True
    )

    model = BearingCNN()
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    # 2. Prepare raw, unstandardized windows (N, 5000).
    X_raw, y_true = load_split("test")  # Replace with your own raw windows.
    X = np.ascontiguousarray(
        ((X_raw - checkpoint["train_mean"]) /
        checkpoint["train_std"])[:, None, :],
        dtype=np.float32
    )
    # 3. Evaluate Metrics
    y_pred = predict_fn(X)
    print(f"Path to file: {elem}")
    print("Model Accuracy:", accuracy(y_true, y_pred))
    print("Model Size: (MB)", measure_model_size_mb(model, kind="pytorch"))
    results = measure_inference_latency_ms(predict_fn, X)
    print("Inference mean_ms: ", results['mean_ms'])
    print("Inference std_ms: ", results['std_ms'])
    print("Inference min_ms: ", results['min_ms'])
    print("Inference max_ms: ", results['max_ms'])
    print("Inference n_repeats: ", results['n_repeats'])




