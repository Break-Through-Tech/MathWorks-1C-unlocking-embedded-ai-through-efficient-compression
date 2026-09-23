"""Creates the baseline model."""

import data
from sklearn import SVC
import itertools
import metrics
import subprocess
import pandas as pd

def compute_acc(model, X, y) -> float:
    """
    Computes the accuracy of the model using the given features and label.

    Args:
        model: Machine learning model used for prediction. 
        X: Input feature data. (X_train OR X_val OR X_test)
        y: Input Label data. (y_train OR y_val OR y_test)

    Returns: 
        Accuracy between predictions, and actual values.
    """
    y_pred = model.predict(X)
    return metrics.accuracy(y, y_pred)

def optimize(X_train_std, y_train, X_val, y_val):
    """
    Finds the best configuration of hyperparameters.

    Args:
        X_val: Validation input feature data.
        y_val: Validation input label data.
    
    Returns:
        Best hyper parameters for SVM in the format:
            [C, kernel, gamma]

        Currently supported optimizations: C, kernel, gamma.
        I was struggling to understand where GridSearch might come into play.
        Maybe actual for the Machine Learning Algorithm. Please let me know.
    """
    C_temp = [10 ** n for n in range(-5, 6)]
    kernel_temp = ['linear', 'poly', 'rbf', 'sigmoid', 'precomputed']
    gamma_temp = ['scale', 'auto']

    best_hyper = []
    best_score = 0
    n = 0
    
    for c, k, g in itertools.product(C_temp, kernel_temp, gamma_temp):
        model = SVC(C=c, kernel=k, gamma=g)
        model.fit(X_train_std, y_train)
        acc_score = compute_acc(model, X_val, y_val)
        print(f"Score for current model: {acc_score}")
        if not best_hyper or acc_score > best_score:
            best_score = acc_score
            best_hyper = [c, k, g]
        n+=1
        print(f"Best current model score: {best_score}.\n"
                f"{n} models trained. {100 - n} to go.")
    print(f"Best model found: C, kernel, gamma = {best_hyper[0], best_hyper[1], best_hyper[2]}")

    return best_hyper


def print_metrics(acc_score, mb_size, latency_dict, f):
    """
    Helper function to print evaluation metrics.

    Args:
        acc_score: Accuracy score of model.
        mb_size: Size of model, in MB
        latency_dict:
            "mean_ms"   : average ms per sample
            "std_ms"    : standard deviation across repeats
            "min_ms"    : fastest single-sample call
            "max_ms"    : slowest single-sample call
            "n_repeats" : how many timed calls this is based on.

    Returns:
        None
    """
    print(f"Accuracy Score:{acc_score}", file=f)
    print(f"File size (MB): {mb_size}", file=f)
    print(f"Mean Latency: {latency_dict['mean_ms']}", file=f)
    print(f"Latency stdev: {latency_dict['std_ms']}", file=f)
    print(f"Min Latency: {latency_dict['min_ms']}", file=f)
    print(f"Max Latency: {latency_dict['max_ms']}", file=f)
    print(f"Sampled {latency_dict['n_repeats']} times.",file=f)

def save_metrics(model, X, y):
    """
    Saves evaluation metrics to output/log.txt.

    Args:
        model: Machine learning model used for prediction. 
        X: Input feature data. (X_train OR X_val OR X_test)
        y: Input Label data. (y_train OR y_val OR y_test)
    
    Returns:
        None

    Creates:
        log.txt. Currently contains:
            - Classification Accuracy
            - Model Size (MB)
            - Inference Speed (ms/sample)
    """
    acc_score = compute_acc(model, X, y)
    mb_size = metrics.measure_model_size_mb(model=model, kind="sklearn")
    latency_dict = metrics.measure_inference_latency_ms()
    with open("log.txt", 'w') as f:
        print_metrics(acc_score, mb_size, latency_dict, file=f)

def save_model(model):
    """
    Saves model as a pickle file to output/model.pkl.

    Args:
        model: Machine learning model used for prediction. 

    Returns:
        None

    Creates:
        Pickle file. Stored as model.pkl. 
    """
    import pickle
    pkl_model_filename = "model.pkl"
    pickle.dump(model, open(pkl_model_filename, 'wb'))

def main():
    """
    Performs the following, in order.
    Reads 12 condition monitoring features dataset.
    Preprocesses Data.
    Optimizes hyperparameters
    Evaluates best hyperparametered model. Saved in output/log.txt.
    Saves model. Saved in output/model.pkl.
    """
    df_train = pd.read_csv("../data/train.csv")
    df_val = pd.read_csv("../data/val.csv")
    df_test = pd.read_csv("../data/test.csv")
    features = [] # TODO
    X_train, y_train = df_train(columns=features), df_train['Fault']
    X_val, y_val = df_val(columns=features), df_train['Fault']
    X_test, y_test = df_test(columns=features), df_train['Fault']

    X_train_std, X_val_std, X_test_std = data.standardize(X_train, X_val, X_test)
    print("Preprocessing complete. Optimizing hyperparameters...")
    optimize(X_train_std, y_train, X_val_std, y_val)
    best = optimize(model, X_val_std, y_val)

    print("Hyper parameters optimized. Creating testing model...")
    model = SVC(C=best[0], kernel=best[1], gamma=best[2])

    print("Model Created. Saving Evaluation Metrics...")
    save_metrics(model, X_test_std, y_test)

    print("Saved. Saving Model...")
    save_model()

    print("Creating output directory...")
    subprocess.run("mkdir output", shell=True)
    subprocess.run("mv model.pkl output", shell=True)
    subprocess.run("mv log.txt output", shell=True)

    print("Done. Check output folder for model and log.txt")
    
if __name__ == '__main__':
    main()


