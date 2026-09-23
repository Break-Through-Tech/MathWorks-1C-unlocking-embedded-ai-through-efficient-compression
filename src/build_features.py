"""Creates 12 condition-monitoring features (RMS, kurtosis, crest factor, etc.) per window."""
import pandas as pd
import data
def extract_features(X):
    """
    Extracts the 12 features. More Specifically: TODO
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
    ret = []
    n = len(X)
    for i in range(n):
        example = []
        # TODO determine which features will be used, compute them in this loop, per example.
        ret.append(example)
    return ret
    
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
    
    Unsure of what features we want to use.
    I presume that they all related to statistical measures of the data. 

    This function begins from correct extraction 
    of features stored in 'features' and feature values per example.    
    """
    ret = []
    #this will vary by train, val, test
    for elem in splits.keys():

        # Obtain 12 condition-monitoring features
        X = pd.DataFrame(extract_features(splits[elem][0]))       
        # Getting y
        y = pd.Series(data._clean_labels(splits[elem][1]))
        
        # Append (DataFrame, Series) to final output.
        ret.append((X, y))
        
    return ret

folder = '../data'

def augment_and_print(dataset):
    #TODO: Features
    X_features = [str(n) for n in range(11)]
    X_label = ['something']
    df_X = pd.DataFrame(dataset[0], columns=X_features)
    df_y = pd.DataFrame(dataset[1], columns=X_label)
    df = pd.concat([df_X, df_y], ignore_index=True)
    df.to_csv(f"{folder}/{dataset[2]}", index=False)

def main():
    splits = data.load_all_splits()
    d = condense_window(splits)
    d[0].append('train.csv')
    d[1].append('val.csv')
    d[2].append('test.csv')
    for dataset in d:    
        augment_and_print(dataset)

if __name__ == '__main__':
    main()