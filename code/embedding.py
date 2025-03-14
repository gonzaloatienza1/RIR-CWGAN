"""
Script for processing and normalizing embeddings based on room configurations derived from audio file metadata.

This script processes audio file metadata to extract room configurations (dimensions, listener and speaker positions, 
T60 reverberation times, and distances). It then converts these configurations into feature matrices, normalizes them, 
and saves the normalized embeddings in a pickle file.

Main functionalities:
- Load audio files and extract corresponding room information (dimensions, positions, distances, etc.).
- Convert the room information into numerical feature matrices.
- Normalize the features using their mean and standard deviation.
- Save the normalized embeddings for further use in Room Impulse Response generation models.

When to use this script:
- When you need to preprocess room data before training models.
- When generating embeddings for models that rely on standardized input data.

Recommendation:
- For consistency, it is advisable to use embeddings derived from non-standardized audio features 
  (the audio processing should be done in raw form before normalization).
"""

import os
import pickle
import numpy as np
import inf_vec
from utils import load_audios_and_filenames
import argparse

def load_data(cache_file, data_folder):
    """
    Load audio data and corresponding file paths from a cache file, or generate them if the cache is missing.

    Parameters:
    - cache_file (str): Path to the cache file.
    - data_folder (str): Directory containing the audio data.

    Returns:
    - tuple: (array_audios, files) containing audio signals and corresponding file paths.
    """
    if os.path.exists(cache_file):
        # Load cached data if it exists
        with open(cache_file, 'rb') as f:
            return pickle.load(f)
    else:
        # Generate audio data and file paths, then cache them
        audios, files = load_audios_and_filenames(data_folder)
        with open(cache_file, 'wb') as f:
            pickle.dump((audios, files), f)
        return audios, files

def load_and_extract_room_info(cache_file, data_folder):
    """
    Load audio files and extract room information using their file names.

    Parameters:
    - cache_file (str): Path to the cache file.
    - data_folder (str): Directory containing the audio data.

    Returns:
    - tuple: Distances and room configuration information extracted from the file names.
    """
    _, files = load_data(cache_file, data_folder)
    return inf_vec.room_inf(files)

def prepare_and_convert_room_configs(room_info):
    """
    Prepare room configuration data and convert it to a feature matrix.

    Parameters:
    - room_info (array): Array containing room information for each audio file.

    Returns:
    - np.ndarray: Feature matrix where each row contains processed room data.
    """
    features = []
    for room in room_info:
        # Convert dimensions from centimeters to meters
        lx, ly, lz = round(room[8] / 100, 2), round(room[9] / 100, 2), round(room[10] / 100, 2)  # Listener coordinates
        sx, sy, sz = round(room[5] / 100, 2), round(room[6] / 100, 2), round(room[7] / 100, 2)  # Speaker coordinates
        rx, ry, rz = round(room[0] / 100, 2), round(room[1] / 100, 2), round(room[4] / 100, 2)  # Room dimensions

        # Extract and adjust T60 values based on predefined conditions
        t60 = round(room[11] / 100, 2)

        # Create a feature row and add it to the features list
        features.append([lx, ly, lz, sx, sy, sz, rx, ry, rz, t60])

    return np.array(features)  # Convert to NumPy array for further processing

def normalize_features(features):
    """
    Normalize features using their mean and standard deviation.

    Parameters:
    - features (np.ndarray): Feature matrix to normalize.

    Returns:
    - tuple: (normalized_features, means, std_devs)
    """
    means = np.mean(features, axis=0)  # Calculate mean of each feature
    std_devs = np.maximum(np.std(features, axis=0), 1e-10)  # Avoid division by zero
    normalized_features = (features - means) / std_devs  # Normalize the features
    return normalized_features, means, std_devs

def destandardize_data(features, means, stds):
    """
    Revert normalized features to their original values.

    Parameters:
    - features (np.ndarray): Normalized feature matrix.
    - means (np.ndarray): Mean values used during normalization.
    - stds (np.ndarray): Standard deviation values used during normalization.

    Returns:
    - np.ndarray: De-normalized (original) features.
    """
    return np.round(features * stds + means, 2)

def save_normalized_embeddings(embeddings, filename):
    """
    Save the normalized embeddings to a pickle file.

    Parameters:
    - embeddings (np.ndarray): Normalized feature matrix.
    - filename (str): Path to the output pickle file.
    """
    with open(filename, 'wb') as f:
        pickle.dump(embeddings, f, protocol=pickle.HIGHEST_PROTOCOL)

def main(data_folder, cache_file, output_embedding_file):
    """
    Main workflow for processing room information, normalizing embeddings, and saving them to a file.

    Parameters:
    - data_folder (str): Directory containing the audio data.
    - cache_file (str): Path to the cache file for loading/storing audio data.
    - output_embedding_file (str): Path to save the normalized embeddings.
    """
    # Load data and extract room information
    dists, room_info = load_and_extract_room_info(cache_file, data_folder)

    # Prepare room configurations and convert them to a feature matrix
    features = prepare_and_convert_room_configs(room_info)

    # Display the first row of raw features for verification
    print("Features (first row):", features[0])

    # Normalize the features and display the first row of normalized features
    normalized_features, feature_means, feature_stds = normalize_features(features)
    print("Normalized features (first row):", normalized_features[0])

    # Save the normalized embeddings to a pickle file
    save_normalized_embeddings(normalized_features, output_embedding_file)

if __name__ == "__main__":
    # Argument parser for command-line execution
    parser = argparse.ArgumentParser(description="Process and normalize embeddings for room configurations.")
    parser.add_argument('--data_folder', type=str, default='../../../../data', help='Path to the folder containing audio data.')
    parser.add_argument('--cache_file', type=str, required=True, help='Path to the cache file.')
    parser.add_argument('--output_embedding_file', type=str, required=True, help='Path to save the normalized embeddings.')
    args = parser.parse_args()

    # Execute the main workflow
    main(args.data_folder, args.cache_file, args.output_embedding_file)
