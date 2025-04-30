import numpy as np
import pickle
import inf_vec
from sklearn.model_selection import train_test_split
from typing import Tuple, List

def load_data(cache_rir: str, cache_emb: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[str], List[str], np.ndarray, np.ndarray]:
    """
    Load preprocessed RIRs and standardized embeddings from pickle files,
    compute distances, and split everything into training and test sets.

    Args:
        cache_rir (str): Path to pickled file containing RIRs and corresponding filenames.
        
        cache_emb (str): Path to pickled file containing embeddings and their normalization stats.

    Returns:
        Tuple containing:

            - **emb_train (np.ndarray)**: 2D array of shape (N_train, F), training embeddings, where:

                - **N_train**: Number of training samples.
                - **F**: Feature dimensionality (typically 10).

            - **emb_test (np.ndarray)**: 2D array of shape (N_test, F), test embeddings, where:

                - **N_test**: Number of test samples.
                - **F**: Feature dimensionality (typically 10).

            - **rirs_train (np.ndarray)**: 2D array of shape (N_train, L), training RIRs, where:

                - **L**: RIR length in samples.

            - **rirs_test (np.ndarray)**: 2D array of shape (N_test, L), test RIRs.

            - **distance_train (np.ndarray)**: 1D array of shape (N_train,), Euclidean distances for training set.

            - **distance_test (np.ndarray)**: 1D array of shape (N_test,), Euclidean distances for test set.

            - **file_names_train (List[str])**: List of N_train filenames for training.

            - **file_names_test (List[str])**: List of N_test filenames for testing.

            - **feature_means (np.ndarray)**: 1D array of shape (F,), feature-wise means used for normalization.

            - **feature_stds (np.ndarray)**: 1D array of shape (F,), feature-wise standard deviations used for normalization.
    """
    with open(cache_rir, 'rb') as f:
        rirs, file_names = pickle.load(f)

    with open(cache_emb, 'rb') as f:
        data = pickle.load(f)
        embeddings = data['embeddings']
        feature_means = data['mean']
        feature_stds = data['std']

    distance, _ = inf_vec.room_inf(file_names)

    emb_train, emb_test, rirs_train, rirs_test, distance_train, distance_test, file_names_train, file_names_test = train_test_split(
        embeddings, rirs, distance, file_names, test_size=0.1, random_state=42)

    return emb_train, emb_test, rirs_train, rirs_test, distance_train, distance_test, file_names_train, file_names_test, feature_means, feature_stds




