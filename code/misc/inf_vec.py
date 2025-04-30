"""
This script extracts room and microphone-related features from filenames in the database.
It builds room embeddings and calculates speaker-to-microphone distances using predefined room
configurations from the `rooms.py` module.

Typical usage:
- Load a list of RIR filenames.
- Use `room_inf()` to extract room embeddings and distances.

Returns:
- Embeddings encoding room geometry, source/receiver locations, and T60.
- 2D Euclidean distance from source to microphone.
"""

from rooms import UTSRoom
import numpy as np
from typing import List, Tuple

def room_inf(file_paths: List[str]) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract room and microphone-related features from a list of RIR file names.

    Args:
        file_paths (List[str]): List of file paths or filenames.

    Returns:
        Tuple[np.ndarray, np.ndarray]: A tuple containing the following elements:

            - **dists (np.ndarray)**: 1D array of shape (N,) with the Euclidean distances between speaker and microphone, where:

                - **N**: Number of processed RIR files.

            - **room_info (np.ndarray)**: 2D array of shape (N, F) containing room embedding features per file, where:

                - **N**: Number of valid RIR files.
                - **F**: Number of embedding features extracted from each file.
    """
    Small_Room = UTSRoom(355, 410, 401, 378, 300, [175.5, 205], 497)
    Medium_Room = UTSRoom(736, 520, 650, 434.5, 300, [368, 217.5], 659)
    Large_Room = UTSRoom(994, 923, 1087, 1022, 300, [497, 486.25], 1281)
    Anechoic_Room = UTSRoom(490, 722, 490, 722, 529, [245, 361], 45)
    Box_Room = UTSRoom(600, 1175, 600, 1175, 300, [300, 881.25], 667)

    room_info_list = []

    for file in file_paths:
        file = str(file).rsplit("\\", 1)[-1]
        characteristics = file.split('_')

        characteristics[1] = characteristics[1].replace('Zone', '')
        characteristics[2] = characteristics[2].replace('MicrophoneArray', '')
        characteristics[3] = int(characteristics[3].replace('L', ''))
        characteristics[4] = int(characteristics[4].replace('M', '').replace('.wav', ''))

        if characteristics[0] == 'SmallMeetingRoom':
            room_info = Small_Room.return_embedding(characteristics)
        elif characteristics[0] == 'MediumMeetingRoom':
            room_info = Medium_Room.return_embedding(characteristics)
        elif characteristics[0] == 'LargeMeetingRoom':
            room_info = Large_Room.return_embedding(characteristics)
        elif characteristics[0] == 'AnechoicRoom':
            room_info = Anechoic_Room.return_embedding(characteristics)
        elif characteristics[0] == 'ShoeBoxRoom':
            room_info = Box_Room.return_embedding(characteristics)
        else:
            continue

        room_info_list.append(room_info)

    room_info = np.array(room_info_list)
    coords = room_info[:, 5:10]

    dists = np.round(np.abs(np.sqrt((coords[:, 3] - coords[:, 0])**2 + (coords[:, 4] - coords[:, 1])**2)), 2)

    return dists, room_info



