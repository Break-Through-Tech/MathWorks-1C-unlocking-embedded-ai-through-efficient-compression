"""Creates the baseline model."""

import sys
import os
from sklearn.svm import SVC
import itertools
import metrics
import pandas as pd
import pickle

class InvalidMethodError(Exception):
    pass
def compute_acc(model, X, y):
    """
    Computes the accuracy of the model using the given features and label.

    Args:
        model: Machine learning model used for prediction. 
        X: Input feature data. (X_train OR X_val OR X_test)
        y: Input Label data. (y_train OR y_val OR y_test)

    Returns: 
        Accuracy between predictions, and actual values.
    """
    if method == 'Baseline':
        y_pred = model.predict(X)
        return metrics.accuracy(y, y_pred)
    else:
        print("Define custom pipeline here...")


def train_model(X_train, y_train, hyperparams):
    """
    Trains a model using Command Line Arguments specified when running the program. 
    Baseline:
        Args: 
        X_train: Training feature input data.
        y_train: Training label input data.

        Returns: The model to be used.
    """
    if method == 'Baseline':
        C, K, G = hyperparams
        model = SVC(C=C, kernel=K, gamma=G)
        model.fit(X_train, y_train)
        return model
    else:
        print("Define custom pipeline here...")


def read_from_csv(paths: list[str]):
    """

    Extracts training, validation, and test data.
    Baseline:
        Args: 
            paths: The list paths leading to the .csv's, relative to the cwd.
        Returns:
            X_train, y_train, X_val, y_val, X_test, y_test
    """
    if method == 'Baseline':
        train_path, val_path, test_path = paths
        df_train = pd.read_csv(train_path)
        df_val = pd.read_csv(val_path)
        df_test = pd.read_csv(test_path)
        features = ['rms_time', 'kurtosis_time', 'stdev_time', 'rms_four','kurtosis_four', 'stdev_four',
                    'rms_spec', 'kurtosis_spec','stdev_spec','rms_psd', 'kurtosis_psd','stdev_psd' ]
        X_train, y_train = df_train[features], df_train['Fault']
        X_val, y_val = df_val[features], df_val['Fault']
        X_test, y_test = df_test[features], df_test['Fault']
        return [X_train, y_train, X_val, y_val, X_test, y_test]
    else:
        print("Define custom pipeline here...")





def optimize(X_train, y_train, X_val, y_val):
    """
    Finds the best configuration of hyperparameters.
    Baseline:
        Args:
            X_train: Training input feature data.
            y_train: Training input label data.
            X_val: Validation input feature data.
            y_val: Validation input label data.
        
        Returns:
            Best hyper parameters for SVM in the format:
                [C, kernel, gamma]
    """
    if method == 'Baseline':
        C_temp = [10 ** n for n in range(-5, 6)]
        kernel_temp = ['linear', 'poly', 'rbf', 'sigmoid']
        gamma_temp = ['scale', 'auto']
        total = len(C_temp) * len(kernel_temp) * len(gamma_temp)

        best_hyper = []
        best_score = 0
        n = 0
        
        for c, k, g in itertools.product(C_temp, kernel_temp, gamma_temp):
            model = train_model(X_train, y_train, [c, k, g])
            acc_score = compute_acc(model, X_val, y_val)
            print(f"Score for current model: {acc_score}")
            if not best_hyper or acc_score > best_score:
                best_score = acc_score
                best_hyper = [c, k, g]
            n+=1
            print(f"Best current model score: {best_score}.\n"
                    f"{n} models trained. {total - n} to go.")
        print(f"Best model found: C, kernel, gamma = {best_hyper[0], best_hyper[1], best_hyper[2]}")

        return best_hyper
    else:
        print("Define custom pipeline here...")



def print_metrics(acc_score, mb_size, latency_dict, best, hyperparams, f):
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
    def print_hyperparams(best, hyperparams, f):
        if method == 'Baseline':
            for i in range (len(hyperparams)):
                print(f"Optimal hyperparameter {hyperparams[i]}: {best[i]}", file=f)
        else:
            print('define cutsom method here...')
        
    if method == 'Baseline':
        print(f"Accuracy Score:{acc_score}", file=f)
        print(f"File size (MB): {mb_size}", file=f)
        print(f"Mean Latency: {latency_dict['mean_ms']}", file=f)
        print(f"Latency stdev: {latency_dict['std_ms']}", file=f)
        print(f"Min Latency: {latency_dict['min_ms']}", file=f)
        print(f"Max Latency: {latency_dict['max_ms']}", file=f)
        print(f"Sampled {latency_dict['n_repeats']} times.",file=f)
        print_hyperparams(best, hyperparams, f)
        

    else:
        print("Define custom pipeline here...")


def save_metrics(model, X, y, best, hyperparams):
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
    if method == 'Baseline':
        acc_score = compute_acc(model, X, y)
        mb_size = metrics.measure_model_size_mb(model=model, kind="sklearn")
        latency_dict = metrics.measure_inference_latency_ms(model.predict, X)
        with open("log.txt", 'w') as f:
            print_metrics(acc_score, mb_size, latency_dict, best, hyperparams, f)
    else:
        print("Define custom pipeline here...")


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
    pkl_model_filename = "model.pkl"
    with open(pkl_model_filename, "wb") as f:
        pickle.dump(model, f)

def main():
    """
    The driver for program.
    Note that you can use Command Line Arguments to specify custom versions of each function.
    Example invocation: python3 build_model.py (argument)
    Then, you can define specific behavior for that custom argument instead.
    """
    global method
    method = sys.argv[1] if len(sys.argv) > 1 else 'Baseline'
  
    try:
        valid_methods = ['Baseline'] #include your methods here. 
        if method not in valid_methods:
            raise InvalidMethodError
    except InvalidMethodError as e:
        print(f'Error: {e}. Please include your specified program method, or ensure your spelling is correct.')
        exit(1)


    if method == 'Baseline': 
        X_train, y_train, X_val, y_val, X_test, y_test = read_from_csv(
            ['../data/train.csv', '../data/val.csv', '../data/test.csv' ],
        )
        print("Preprocessing complete. Optimizing hyperparameters...")
        C, K, G =  optimize(X_train, y_train, X_val, y_val)
        print("Hyper parameters optimized. Creating testing model...")
        model = train_model(X_train, y_train, [C, K, G])
        print("Model Created. Saving Evaluation Metrics...")
        save_metrics(model, X_test, y_test, [C, K, G], ['C', 'Kernel', 'Gamma'] )

        print("Saved. Saving Model...")
        save_model(model=model)
        print("Creating output directory...")
        os.makedirs("output", exist_ok=True)
        os.replace("model.pkl", "output/model.pkl")
        os.replace("log.txt", "output/log.txt")
        print("Done. Check output folder for model and log.txt")
    else:
        print("Define custom pipeline here...")
    
if __name__ == '__main__':
    main()