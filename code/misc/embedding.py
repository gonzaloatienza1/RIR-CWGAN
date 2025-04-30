"""
This module handles the extraction, normalization, and serialization of room configuration embeddings
used for Room Impulse Response (RIR) generation.

Functionality:
- Loads raw room metadata from audio files.
- Converts metadata into numerical feature vectors.
- Normalizes features and stores mean/std for reuse.
- Exports standardized embeddings in a consistent pickle format.

Note:
The output embeddings should be used directly in all training and evaluation scripts,
which expect pre-standardized embeddings.
"""
import os
import pickle
import numpy as np
import inf_vec
from misc import load_audios_and_filenames
import argparse
from typing import Tuple, List

def embedding(data_folder: str, cache_file: str, output_embedding_file: str) -> None:
    """
    Full embedding processing pipeline: load room info → extract features → normalize → save to disk.

    Args:
        data_folder (str): Directory containing the raw audio data.
        cache_file (str): Path to the cache file containing or storing audio metadata.
        output_embedding_file (str): Destination path for the output pickle file with normalized embeddings.
    """
    _, room_info = load_and_extract_room_info(cache_file, data_folder)
    features = prepare_and_convert_room_configs(room_info)
    normalized_features, means, stds = normalize_features(features)
    save_normalized_embeddings(normalized_features, means, stds, output_embedding_file)

def load_data(cache_file: str, data_folder: str) -> Tuple[np.ndarray, List[str]]:
    """
    Load audio data and file names from cache or regenerate them from the audio folder.

    Args:
        cache_file (str): Path to cache file.

        data_folder (str): Directory containing the audio data.

    Returns:
        Tuple[np.ndarray, List[str]]: 

            - **audios (np.ndarray)**: 1D array of shape (n_samples,) containing audio signals, where:

                - **n_samples**: Number of audio recordings.

            - **files (List[str])**: List of n_samples file names associated with each audio.
    """
    if os.path.exists(cache_file):
        with open(cache_file, 'rb') as f:
            return pickle.load(f)
    else:
        audios, files = load_audios_and_filenames(data_folder)
        with open(cache_file, 'wb') as f:
            pickle.dump((audios, files), f)
        return audios, files

def load_and_extract_room_info(cache_file: str, data_folder: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load audio filenames and extract room information from their metadata.

    Args:
        cache_file (str): Path to the audio data cache.
        
        data_folder (str): Audio data directory.

    Returns:
        Tuple[np.ndarray, np.ndarray]: 

            - **np.ndarray**: Distance metadata of shape (N,), where:

                - **N**: Number of audio files (1 distance per file).

            - **np.ndarray**: Room configuration metadata of shape (N, M), where:

                - **N**: Number of audio files.
                - **M**: Number of room-related features extracted from each filename.
    """
    _, files = load_data(cache_file, data_folder)
    return inf_vec.room_inf(files)


def prepare_and_convert_room_configs(room_info: np.ndarray) -> np.ndarray:
    """
    Convert raw room metadata to a numerical feature matrix.

    Args:
        room_info (np.ndarray): Parsed room data per audio file. Expected shape: (N, M), where:

            - **N**: Number of audio files.
            - **M**: Number of extracted metadata fields.

    Returns:
        np.ndarray: Feature matrix of shape (N, F), where:

            - **N**: Number of audio files.
            - **F**: Number of extracted features per sample.
    """
    features = []
    for room in room_info:
        lx, ly, lz = round(room[8] / 100, 2), round(room[9] / 100, 2), round(room[10] / 100, 2)
        sx, sy, sz = round(room[5] / 100, 2), round(room[6] / 100, 2), round(room[7] / 100, 2)
        rx, ry, rz = round(room[0] / 100, 2), round(room[1] / 100, 2), round(room[4] / 100, 2)
        t60 = round(room[11] / 100, 2)
        features.append([lx, ly, lz, sx, sy, sz, rx, ry, rz, t60])
    return np.array(features)

def normalize_features(features: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Standardize features using their mean and standard deviation.

    Args:
        features (np.ndarray): Raw feature matrix. Shape: (N, F), where:

            - **N**: Number of samples.
            - **F**: Number of features per sample.

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray]:

            - **normalized_features (np.ndarray)**: Standardized feature 2D array of shape (N, F), where:

                - **N**: Number of samples.
                - **F**: Number of features per sample.

            - **means (np.ndarray)**: 1D array of shape (F,), containing feature-wise means, where:

                - **F**: Number of features per sample.

            - **stds (np.ndarray)**: 1D array of shape (F,), containing feature-wise standard deviations, where:

                - **F**: Number of features per sample.
    """
    means = np.mean(features, axis=0)
    stds = np.maximum(np.std(features, axis=0), 1e-10)
    normalized = (features - means) / stds
    return normalized, means, stds

def destandardize_data(features: np.ndarray, means: np.ndarray, stds: np.ndarray) -> np.ndarray:
    """
    Revert standardized features back to their original scale.

    Args:
        features (np.ndarray): Normalized feature 2D array of shape (N, F), where:

            - **N**: Number of samples.
            - **F**: Number of features per sample.

        means (np.ndarray): Feature-wise means used during standardization. 1D array of shape: (F,), where:

            - **F**: Number of features per sample.

        stds (np.ndarray): Feature-wise standard deviations used during standardization. 1D array of shape: (F,), where:

            - **F**: Number of features per sample.

    Returns:
        np.ndarray: De-standardized feature 2D array of shape (N, F), same as the input.
    """
    return np.round(features * stds + means, 2)

def save_normalized_embeddings(embeddings: np.ndarray, means: np.ndarray, stds: np.ndarray, filename: str) -> None:
    """
    Save standardized embeddings and their associated statistics (mean and std) to a pickle file.

    Args:
        embeddings (np.ndarray): Normalized embeddings 2D array of shape (N, F), where:

            - **N**: Number of samples.
            - **F**: Number of features per sample.

        means (np.ndarray): Feature-wise means. 1D array of shape: (F,), where:

            - **F**: Number of features per sample.

        stds (np.ndarray): Feature-wise standard deviations. 1D array of shape: (F,), where:

            - **F**: Number of features per sample.

        filename (str): Full path where the pickle file will be saved.
    """
    data = {
        'embeddings': embeddings,
        'mean': means,
        'std': stds
    }
    with open(filename, 'wb') as f:
        pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process and normalize embeddings for room configurations.")
    parser.add_argument('--data_folder', type=str, default='../../../../data', help='Path to the folder containing audio data.')
    parser.add_argument('--cache_file', type=str, required=True, help='Path to the cache file.')
    parser.add_argument('--output_embedding_file', type=str, required=True, help='Path to save the normalized embeddings.')
    args = parser.parse_args()
    embedding(args.data_folder, args.cache_file, args.output_embedding_file)
