""" 
This script extracts room and microphone-related features from file names in the database.
It uses room information to generate embeddings and distances.
The script relies on the predefined UTSRoom configurations defined in rooms.py. 
"""

from rooms import UTSRoom
import numpy as np

def room_inf(array_file_path):
    """
    Extract room and microphone-related features from file names.
    array_file_path: List of file paths containing room-related data.

    Returns:
    - dists: Array of distances between microphone and speaker positions.
    - room_info: Array of room embeddings for each file.
    """
    # Define UTSRoom configurations based on room sizes and properties
    Small_Room = UTSRoom(355, 410, 401, 378, 300, [175.5, 205], 497)
    Medium_Room = UTSRoom(736, 520, 650, 434.5, 300, [368, 217.5], 659)
    Large_Room = UTSRoom(994, 923, 1087, 1022, 300, [497, 486.25], 1281)
    Anechoic_Room = UTSRoom(490, 722, 490, 722, 529, [245, 361], 45)
    Box_Room = UTSRoom(600, 1175, 600, 1175, 300, [300, 881.25], 667)

    room_info_list = []  # List to store room embeddings

    for file in array_file_path:
        # Extract the file name from the full path
        file = str(file).rsplit("\\", 1)[-1]

        # Split the file name to extract its characteristics
        characteristics = file.split('_')

        # Clean up and convert specific characteristics
        characteristics[1] = characteristics[1].replace('Zone', '')
        characteristics[2] = characteristics[2].replace('MicrophoneArray', '')
        characteristics[3] = int(characteristics[3].replace('L', ''))  # Speaker ID
        characteristics[4] = int(characteristics[4].replace('M', '').replace('.wav', ''))  # Microphone ID

        # Determine the room type based on the first characteristic
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
            continue  # Skip if the room type is unknown

        room_info_list.append(room_info)  # Store the room embedding

    room_info = np.array(room_info_list)  # Convert list to NumPy array

    # Extract microphone and speaker coordinates from the embeddings
    coords = room_info[:, 5:10]
    
    # Calculate distances between the microphone and speaker positions
    dists = np.round(np.abs(np.sqrt((coords[:, 3] - coords[:, 0])**2 + (coords[:, 4] - coords[:, 1])**2)), 2)

    return dists, room_info  

