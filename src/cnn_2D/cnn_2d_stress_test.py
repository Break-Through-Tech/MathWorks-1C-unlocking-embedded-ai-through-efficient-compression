import numpy as np
import torch
import matplotlib.pyplot as plt
from torch.utils.data import TensorDataset, DataLoader

# setup for standalone script
from src.data import load_all_splits
from src.cnn_2D.cnn_2d_model import TinySpecCNN, signals_to_spectrograms

# Load data to get the exact training mean/std for the standardization step
splits = load_all_splits(data_dir='data')
X_train, _ = splits["train"]
train_mean = X_train.mean()
train_std = X_train.std()
label_map = {'Normal': 0, 'InnerRaceFault': 1, 'OuterRaceFault': 2}

# Initialize the blank model architecture
model = TinySpecCNN()
# -----------------------------------

# 1. Reload best saved model weights
# Load the weights saved at the lowest validation loss during training
model.load_state_dict(torch.load('cnn_outputs/cnn_2d_spec.pth', map_location='cpu'))
# Put the model in evaluation mode so it doesn't change weights during testing
model.eval()


# 2. Define helper functions (adding noise & running predictions)
def add_awgn(signal, snr_db, rng):
    """
    Adds Additive White Gaussian Noise (AWGN) to raw 1D vibration signals.
    - signal: The clean test data of shape (102, 5000)
    - snr_db: Signal-to-Noise Ratio in decibels (lower dB = louder noise)
    - rng: Random number generator with a fixed seed for reproducible results
    """
    # Step A: Measure the power (variance) of each 5,000-sample vibration wave.
    # keepdims=True keeps the shape as (102, 1) so each sample scales by its own power.
    signal_power = np.var(signal, axis=1, keepdims=True)

    # Step B: Convert the target SNR (in dB) to a linear ratio and calculate how
    # strong the static noise needs to be to match that SNR level.
    # Formula: SNR_linear = 10^(SNR_dB / 10)  -->  Noise_Power = Signal_Power / SNR_linear
    noise_power = signal_power / (10 ** (snr_db / 10.0))

    # Step C: Generate random bell-curve (Gaussian) static noise centered at 0.
    # Standard deviation is the square root of noise_power.
    noise = rng.normal(0, np.sqrt(noise_power), signal.shape)

    # Step D: Add the random static noise directly onto the raw vibration wave.
    return signal + noise


def evaluate_signals(X_raw_input, y_idx):
    """
    Runs the full 2D CNN pipeline on a set of raw 1D signals:
    Raw 1D wave -> Standardize -> Convert to 2D spectrogram -> Predict class
    """
    # Step A: Standardize using ONLY the clean training mean and std (no data leakage)
    X_std = (X_raw_input - train_mean) / train_std

    # Step B: Convert standardized 1D waves (102, 5000) into 2D spectrograms (102, 1, 129, 22)
    X_spec = signals_to_spectrograms(X_std)

    # Step C: Wrap spectrograms and true labels into a PyTorch DataLoader for batching
    dataset = TensorDataset(torch.tensor(X_spec, dtype=torch.float32),
                            torch.tensor(y_idx, dtype=torch.long))
    loader = DataLoader(dataset, batch_size=16, shuffle=False)

    # Step D: Count how many test samples the 2D CNN classifies correctly
    correct, total = 0, 0
    with torch.no_grad(): # Disable gradient calculation to speed up testing
        for inputs, labels in loader:
            outputs = model(inputs)              # Get the 3 class logit scores
            _, predicted = torch.max(outputs, 1) # Pick the class with the highest score
            total += labels.size(0)              # Add batch size to total count (102 total)
            correct += (predicted == labels).sum().item() # Add correct guesses

    # Return accuracy as a decimal (e.g., 98 / 102 = 0.9608)
    return correct / total


# 3. Run the noise stress test (clean + 10 trials per noise level)
# Get the 102 raw test signals and convert text labels ('Normal', etc.) to numbers (0, 1, 2)
X_test, y_test = splits["test"]
y_test_idx = np.array([label_map[y] for y in y_test])

# Noise levels from mild noise (30 dB) down to extreme noise (0 dB, where noise = signal)
snr_levels = [30, 20, 10, 5, 0]
num_trials = 10 # Run 10 random noise draws per level so one "lucky" draw doesn't skew results

# First, get the baseline accuracy on the clean, uncorrupted test set
clean_acc = evaluate_signals(X_test, y_test_idx)
mean_accuracies = [clean_acc]
std_accuracies = [0.0] # Clean data has 0 random noise variation

print(f"Clean Baseline | Accuracy: {clean_acc:.4f}")

# Loop through each noise level (30, 20, 10, 5, 0 dB)
for snr in snr_levels:
    # Create a seeded random generator so running the cell again gives the exact same numbers
    rng = np.random.default_rng(seed=42 + snr)
    trial_accs = []

    # Run 10 separate trials with different random static noise draws
    for _ in range(num_trials):
        X_noisy = add_awgn(X_test, snr, rng)
        trial_accs.append(evaluate_signals(X_noisy, y_test_idx))

    # Calculate the average accuracy and standard deviation (spread) across the 10 trials
    mean_acc = np.mean(trial_accs)
    std_acc = np.std(trial_accs)

    mean_accuracies.append(mean_acc)
    std_accuracies.append(std_acc)
    print(f"SNR: {snr:2d} dB       | Mean Accuracy: {mean_acc:.4f} ± {std_acc:.4f}")


# 4. Plot the results (zoomed y-axis to show the degradation curve clearly)
plot_labels = ["Clean", "30 dB", "20 dB", "10 dB", "5 dB", "0 dB"]
mean_arr = np.array(mean_accuracies)
std_arr = np.array(std_accuracies)

plt.figure(figsize=(8, 5))

# Plot the main accuracy line
plt.plot(plot_labels, mean_arr, marker='o', linestyle='-', color='b', linewidth=2, label='2D CNN (Baseline)')

# Plot the light blue shaded band showing ±1 Standard Deviation across the 10 trials
plt.fill_between(range(len(plot_labels)), mean_arr - std_arr, np.minimum(1.0, mean_arr + std_arr), color='b', alpha=0.15, label='±1 Std Dev (10 trials)')

# Add percentage text labels directly above each dot so the drop is easy to read
for i, acc in enumerate(mean_arr):
    plt.text(i, acc + 0.012, f"{acc*100:.1f}%", ha='center', fontsize=9, fontweight='bold', color='navy')

plt.title("2D CNN (Spectrogram) Robustness to Noise (10-Trial Mean ± SD)")
plt.xlabel("Signal-to-Noise Ratio (SNR) — Lower dB = More Noise")
plt.ylabel("Test Accuracy")

# Zoom the y-axis between 75% and 103% so the drop from 97% to 85% isn't squished at the top
plt.ylim(0.75, 1.03)
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend(loc='lower left')

plt.savefig("cnn_outputs/noise_stress_test_chart.png", bbox_inches='tight')
plt.show()
