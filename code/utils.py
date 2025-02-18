import numpy as np
import os
import re
import tqdm
import librosa
import pickle
import embedding
import inf_vec
from scipy.signal import correlate, correlation_lags
from sklearn.model_selection import train_test_split

def load_data(cache_rir, cache_emb, standardize=False):
    """
    Load RIRs and embeddings from cached pickle files and split them into training and test sets.
    Optionally standardize the embeddings and return their means and standard deviations.
    """
    # Load cached RIR data and corresponding filenames
    with open(cache_rir, 'rb') as f:
        rirs, file_names = pickle.load(f)
    
    # Load cached embedding data
    with open(cache_emb, 'rb') as f:
        embeddings = pickle.load(f)

    # If standardization is requested, normalize the features
    feature_means, feature_stds = None, None
    if standardize:
        embeddings, feature_means, feature_stds = embedding.normalize_features(embeddings)
    
    # Compute distances based on room information
    distance, _ = inf_vec.room_inf(file_names)

    # Split the data into training and test sets
    emb_train, emb_test, rirs_train, rirs_test, distance_train, distance_test, file_names_train, file_names_test = train_test_split(
        embeddings, rirs, distance, file_names, test_size=0.1, random_state=42)

    # Return feature means and stds only if standardization was applied
    if standardize:
        return emb_train, emb_test, rirs_train, rirs_test, distance_train, distance_test, file_names_train, file_names_test, feature_means, feature_stds
    else:
        return emb_train, emb_test, rirs_train, rirs_test, distance_train, distance_test, file_names_train, file_names_test


def align_rirs(rir1, rir2, target_samples=6000, lag_threshold_ratio=0.2):
    """
    Align two RIRs by cross-correlation and pad/truncate them to the target sample size.
    rir1, rir2: Input RIR signals to align.
    target_samples: Number of samples to pad/truncate the aligned signals.
    lag_threshold_ratio: Maximum allowable lag as a fraction of the RIR length.
    """
    # Compute cross-correlation between the two RIRs
    corr = correlate(rir1, rir2, mode='full')
    # Find lags associated with the correlation
    lags = correlation_lags(len(rir1), len(rir2), mode='full')
    # Determine the lag with the maximum correlation
    lag_max = lags[np.argmax(corr)]
    lag_threshold = int(lag_threshold_ratio * len(rir1))
    
    # If lag exceeds the threshold, do not adjust
    if abs(lag_max) > lag_threshold:
        aligned_rir1 = rir1
        aligned_rir2 = rir2
    else:
        # Adjust signals according to the maximum lag
        if lag_max > 0:
            aligned_rir1 = rir1[lag_max:]
            aligned_rir2 = rir2[:len(aligned_rir1)]
        elif lag_max < 0:
            aligned_rir2 = rir2[-lag_max:]
            aligned_rir1 = rir1[:len(aligned_rir2)]
        else:
            aligned_rir1 = rir1
            aligned_rir2 = rir2

        # Ensure both signals have the same length
        min_len = min(len(aligned_rir1), len(aligned_rir2))
        aligned_rir1 = aligned_rir1[:min_len]
        aligned_rir2 = aligned_rir2[:min_len]

    # Pad or truncate to the target number of samples
    if len(aligned_rir1) < target_samples:
        zeros_to_add = target_samples - len(aligned_rir1)
        aligned_rir1 = np.pad(aligned_rir1, (0, zeros_to_add), 'constant')
        aligned_rir2 = np.pad(aligned_rir2, (0, zeros_to_add), 'constant')
    else:
        aligned_rir1 = aligned_rir1[:target_samples]
        aligned_rir2 = aligned_rir2[:target_samples]

    return aligned_rir1, aligned_rir2, lag_max

def trim_audio(audio_array):
    """
    Parameters:
    - audio_array: List of NumPy arrays representing audio signals.
    
    Returns:
    - trimmed_audio_array: NumPy array of trimmed audio signals.
    """
    trimmed_audio_array = []

    # Loop through each audio and trim to the desired length
    for audio in audio_array:
        start_index = 0
        end_index = 6144
        trimmed_audio = audio[start_index:end_index]
        trimmed_audio_array.append(trimmed_audio)

    return np.array(trimmed_audio_array)

def load_audios_and_filenames(data_directory, max_audios_per_folder=None):
    """
    Load audio files and their corresponding file paths from the specified directory.
    Only .wav files that match the criteria are loaded.
    
    Parameters:
    - data_directory: Directory containing room subdirectories with audio files.
    - max_audios_per_folder: Maximum number of audios to load per folder. If None, load all audios.
    
    Returns:
    - all_audios: NumPy array containing the trimmed audio signals.
    - all_filenames: List of filenames corresponding to the audio signals.
    """
    all_file_paths = []
    all_audios = []

    # Iterate through specified room folders
    for room in ['Small Meeting Room', 'Medium Meeting Room', 'Large Meeting Room', 'Anechoic Room', 'Shoe-box Room']:
        room_file_paths = []
        room_audios = []

        # Loop through zones and microphone array types
        for zone in ['ZoneA', 'ZoneB', 'ZoneC', 'ZoneD', 'ZoneE']:
            for array_type in ["PlanarMicrophoneArray", "CircularMicrophoneArray"]:
                zone_folder = os.path.join(data_directory, room, zone, array_type)
                print(zone_folder)
                
                # Sort files based on speaker and microphone indices
                file_list = sorted(
                    os.listdir(zone_folder), 
                    key=lambda x: [int(re.search(r'\d+', x.split('_')[3]).group()), int(re.search(r'\d+', x.split('_')[4]).group())]
                )

                # Limit the number of files if specified
                if max_audios_per_folder is not None:
                    file_list = file_list[:max_audios_per_folder]

                # Load and process each audio file
                for file_name in tqdm.tqdm(file_list):
                    file_path = os.path.join(zone_folder, file_name)
                    if file_name.endswith('.wav'):
                        speaker = int(re.search(r'\d+', file_name.split('_')[3]).group())
                        if speaker in range(1, 61):
                            audio, sr = librosa.load(file_path, sr=48000)
                            peak_index = np.argmax(np.abs(audio))  # Find peak sample index
                            audio = audio[peak_index - 10:]  # Trim starting near the peak
                            room_audios.append(audio)
                            room_file_paths.append(file_path)

        # Trim and store the audio signals
        trimmed_audios = trim_audio(room_audios)
        all_audios.extend(trimmed_audios)
        all_file_paths.extend(room_file_paths)

    # Convert lists to NumPy arrays
    all_audios = np.array(all_audios)
    all_file_paths = np.array(all_file_paths)

    # Extract filenames from paths
    all_filenames = [os.path.basename(path) for path in all_file_paths]

    return all_audios, all_filenames
