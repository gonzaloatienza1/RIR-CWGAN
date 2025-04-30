import numpy as np
import os
import re
import tqdm
import librosa
from typing import Tuple, List

def load_audios_and_filenames(data_directory: str, max_audios_per_folder: int = None) -> Tuple[np.ndarray, List[str]]:
    """
    Load .wav audio files from structured room folders, filter valid recordings, trim and return them.

    Args:
        data_directory (str): Base directory containing the structured room folders.
        
        max_audios_per_folder (int, optional): Maximum number of audio files to load per subfolder. Defaults to None.

    Returns:
        Tuple[np.ndarray, List[str]]: A tuple containing the following elements:

            - **np.ndarray**: 2D array of shape (N, L) containing the trimmed audio signals, where:

                - **N**: Total number of loaded audio files.
                - **L**: Number of samples per trimmed audio signal (fixed at 6144).

            - **List[str]**: List of N filenames corresponding to the loaded audio files, where:

                - **N**: Same as above, one filename per audio sample.
    """
    all_file_paths = []
    all_audios = []

    for room in ['Small Meeting Room', 'Medium Meeting Room', 'Large Meeting Room', 'Anechoic Room', 'Shoe-box Room']:
        for zone in ['ZoneA', 'ZoneB', 'ZoneC', 'ZoneD', 'ZoneE']:
            for array_type in ["PlanarMicrophoneArray", "CircularMicrophoneArray"]:
                zone_folder = os.path.join(data_directory, room, zone, array_type)
                print(zone_folder)

                file_list = sorted(
                    os.listdir(zone_folder),
                    key=lambda x: [int(re.search(r'\d+', x.split('_')[3]).group()), int(re.search(r'\d+', x.split('_')[4]).group())]
                )

                if max_audios_per_folder is not None:
                    file_list = file_list[:max_audios_per_folder]

                for file_name in tqdm.tqdm(file_list):
                    file_path = os.path.join(zone_folder, file_name)
                    if file_name.endswith('.wav'):
                        speaker = int(re.search(r'\d+', file_name.split('_')[3]).group())
                        if speaker in range(1, 61):
                            audio, sr = librosa.load(file_path, sr=48000)
                            peak_index = np.argmax(np.abs(audio))
                            audio = audio[peak_index - 10:]
                            all_audios.append(audio[:6144])
                            all_file_paths.append(file_path)

    return np.array(all_audios), [os.path.basename(p) for p in all_file_paths]
