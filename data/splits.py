import numpy as np
from sklearn.model_selection import KFold

""" Create K-Fold splits for cross-validation
    We use sklearn's KFold to create the splits.
    The function takes the number of samples, number of splits, and a random seed for reproducibility.
    It returns a list of tuples, each containing the training and validation indices for each fold.
"""

def make_kfold_splits(n_samples, n_splits, seed):
    indices = np.arange(n_samples)
    kf = KFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=seed
    )
    return list(kf.split(indices))

