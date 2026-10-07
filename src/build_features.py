"""Creates 12 condition-monitoring features (RMS, kurtosis, crest factor, etc.) per window."""
import pandas as pd
import data
import numpy as np
from scipy.fft import rfft
from scipy.signal import spectrogram, welch
from scipy.stats import kurtosis

FS = 48858
N = 5000

def extract_features(X_train, X_val, X_test):
    """
    Extracts the 12 features. More Specifically:
    Args:
        X: Input data, as Numpy array.

    Returns: List of lists. Each row is another example.
        Return example:
        |     mean       |      std      |  ... (10 more)
        [[mean(example_1), std(example_1),  ...],
        [ mean(example_2), std(example_2),  ...],
        [ mean...        , std...        ,  ...],
        [ mean(example_n), std(example_n),  ...]]
    """
    
    extracted = []
    X_train_std, X_val_std, X_test_std = data.standardize(X_train, X_val, X_test)
    X = [X_train_std, X_val_std, X_test_std]
    for dataset in X:
        #Features:
        #time-domain signal, fourier-domain signal, spectrogram, PSD
        #obtain RMS, kurtosis, stdev for each plot.
        ret = []
        for example in dataset:
            #spectrogram
            _, _, Sxx = spectrogram(example, fs=FS, nperseg=256)
            # FFT:
            fd_signal = np.abs(rfft(example))
            #PSD
            _, Pxx = welch(example, fs=FS)

            rms_time = np.sqrt(np.mean(example**2))
            kurtosis_time = kurtosis(example)
            stdev_time = np.std(example)

            rms_four = np.sqrt(np.mean(fd_signal**2))
            kurtosis_four = kurtosis(fd_signal)
            stdev_four = np.std(fd_signal)

            rms_spec = np.sqrt(np.mean(Sxx**2))
            kurtosis_spec = kurtosis(Sxx.ravel())
            stdev_spec = np.std(Sxx)

            rms_psd = np.sqrt(np.mean(Pxx**2))
            kurtosis_psd = kurtosis(Pxx.ravel())
            stdev_psd = np.std(Pxx)
            ret.append([rms_time, kurtosis_time, stdev_time, rms_four, kurtosis_four, stdev_four, rms_spec, kurtosis_spec, 
                        stdev_spec, rms_psd, kurtosis_psd, stdev_psd])
        extracted.append(ret)

    return extracted
    
def condense_window(splits):
    """
    Condenses window to 12 monitoring features.

    Args:
        splits: dict like {"train": (X_train, y_train), "val": (X_val, y_val), "test": (X_test, y_test)}
    Returns: 
        DataFrame of split data (for all files)

    For single file, n examples: 
        Turns (n, 5000) -> (n, 12). 
        Then appends (DataFrame, Series) to final output.
    Repeat for each file. Consequently returns [df_train, df_val, df_test]  
    """
    ret = []

    X_train, y_train = splits['train']
    X_val, y_val = splits['val']
    X_test, y_test = splits['test']

    X = extract_features(X_train, X_val, X_test) 
    ret.append((X[0], y_train, 'train.csv'))
    ret.append((X[1], y_val, 'val.csv'))
    ret.append((X[2], y_test, 'test.csv'))
    return ret

folder = '../data'

def augment_and_print(dataset):
    X_features = ['rms_time', 'kurtosis_time', 'stdev_time', 'rms_four','kurtosis_four', 'stdev_four',
                   'rms_spec', 'kurtosis_spec','stdev_spec','rms_psd', 'kurtosis_psd','stdev_psd' ]
    df_X = pd.DataFrame(dataset[0], columns=X_features)
    df_y = pd.DataFrame(dataset[1], columns=['Fault'])
    df = pd.concat([df_X, df_y], axis=1)
    df.to_csv(f"{folder}/{dataset[2]}", index=False)

def main():
    splits = data.load_all_splits()
    d = condense_window(splits)
    for dataset in d:    
        augment_and_print(dataset)

if __name__ == '__main__':
    main()