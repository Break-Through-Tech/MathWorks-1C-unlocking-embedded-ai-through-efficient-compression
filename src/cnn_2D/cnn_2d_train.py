import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

from src.data import load_all_splits
from src.cnn_2D.cnn_2d_model import TinySpecCNN, signals_to_spectrograms

# 1. Load data splits
splits = load_all_splits(data_dir='data')
X_train, y_train = splits["train"]
X_val,   y_val   = splits["val"]

# 2. Global standardization using ONLY training set statistics
train_mean = X_train.mean()
train_std = X_train.std()

X_train_std = (X_train - train_mean) / train_std
X_val_std   = (X_val - train_mean) / train_std

# 3. Transform 1D arrays into 2D spectrogram images
X_train_spec = signals_to_spectrograms(X_train_std)
X_val_spec   = signals_to_spectrograms(X_val_std)

# PyTorch's CrossEntropyLoss requires 0-indexed integer targets, not strings.
label_map = {'Normal': 0, 'InnerRaceFault': 1, 'OuterRaceFault': 2}
y_train_idx = np.array([label_map[y] for y in y_train])
y_val_idx   = np.array([label_map[y] for y in y_val])

# Wrap the pre-processed spectrograms and integer labels into PyTorch Datasets.
# X_train_spec shape must be (N, Channels, Height, Width) -> (393, 1, 129, 22).
train_dataset = TensorDataset(torch.tensor(X_train_spec, dtype=torch.float32),
                              torch.tensor(y_train_idx, dtype=torch.long))
val_dataset   = TensorDataset(torch.tensor(X_val_spec, dtype=torch.float32),
                              torch.tensor(y_val_idx, dtype=torch.long))

# DataLoaders handle batching. Shuffle training data to prevent cyclical learning biases,
# but leave validation unshuffled since order does not impact evaluation.
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader   = DataLoader(val_dataset, batch_size=16, shuffle=False)

model = TinySpecCNN()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

os.makedirs('cnn_outputs', exist_ok=True)
epochs = 20
best_val_acc = 0.0
history = [] # Initialize list to track metrics

for epoch in range(epochs):
    # --- Training Phase ---
    model.train() # Enables dropout/batchnorm if added later
    running_loss = 0.0

    for inputs, labels in train_loader:
        optimizer.zero_grad()               # Clear old gradients
        outputs = model(inputs)             # Forward pass
        loss = criterion(outputs, labels)   # Calculate divergence from truth
        loss.backward()                     # Backpropagation
        optimizer.step()                    # Update weights

        # Accumulate total loss for the epoch
        running_loss += loss.item() * inputs.size(0)

    epoch_loss = running_loss / len(train_loader.dataset)

    # --- Validation Phase ---
    model.eval() # Freezes model weights
    correct = 0
    total = 0

    # torch.no_grad() disables gradient tracking, saving memory and speeding up inference
    with torch.no_grad():
        for inputs, labels in val_loader:
            outputs = model(inputs)
            # Find the class index with the highest logit score
            _, predicted = torch.max(outputs, 1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    val_acc = correct / total

    # Log metrics for this epoch
    history.append({
        'epoch': epoch + 1,
        'train_loss': epoch_loss,
        'val_accuracy': val_acc
    })

    # Checkpointing: Save only the model weights that perform best on unseen validation data
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), 'cnn_outputs/cnn_2d_spec.pth')

    print(f"Epoch {epoch+1:02d}/{epochs} | Train Loss: {epoch_loss:.4f} | Val Acc: {val_acc:.4f}")

# Save the training history to a CSV to match the team's workflow
history_df = pd.DataFrame(history)
history_df.to_csv('cnn_outputs/cnn_2d_spec_history.csv', index=False)

print(f"\n Training complete. Best Validation Accuracy: {best_val_acc:.4f}")
print("Saved model weights to 'cnn_outputs/cnn_2d_spec.pth' and metrics to 'cnn_outputs/cnn_2d_spec_history.csv'.")
