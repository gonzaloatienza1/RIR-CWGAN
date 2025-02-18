""" 
This script defines classes to represent rooms and their geometrical configurations,
including quadrilaterals, general rooms, and specific room configurations (UTSRoom).
It also provides methods to compute embeddings based on microphone and speaker placements.
IMPORTANT: This script is specifically designed for use with the database described in the provided project documentation.
"""

import math

class Quadrilateral:
    """
    Quadrilateral class defines the main elements of a quadrilateral,
    characterized by its four sides: a, b, c, d.
    """
    def __init__(self, a, b, c, d):
        # Initialize side lengths of the quadrilateral
        self.a = a
        self.b = b
        self.c = c
        self.d = d

class Room(Quadrilateral):
    """
    Room class extends the Quadrilateral class and adds the height dimension,
    representing a 3D room. 
    """
    def __init__(self, a, b, c, d, height):
        # Initialize the base quadrilateral and room height
        super().__init__(a, b, c, d)
        self.height = height

        # Vector to store room dimensions
        self.vector = []
        self.set_vector()  # Initialize the room vector

    def set_vector(self):
        """ Set the room's vector using rounded dimensions. """
        self.vector = [round(self.a), round(self.b), round(self.c), round(self.d), round(self.height)]

    def return_vector(self):
        """ Return the room's vector representing its dimensions. """
        return self.vector

class UTSRoom(Room):
    """
    UTSRoom class models specific room configurations based on microphone and speaker placements.
    It extends the general Room class and includes additional parameters like grid center and RT60.
    This class is specifically tailored for the database's structure and naming conventions.
    """
    def __init__(self, a, b, c, d, height, grid_center, rt60):
        # Initialize base room and add specific attributes for the UTSRoom
        super().__init__(a, b, c, d, height)
        self.grid_center = grid_center  # Center of the microphone/speaker grid
        self.rt60 = rt60  # Reverberation time in milliseconds

    def get_m_l_position(self, characteristics):
        """
        Calculate microphone and speaker positions based on room configuration and input characteristics.
        characteristics: List containing [zone, array type, speaker ID, microphone ID].
        """
        zone = characteristics[1]
        array = characteristics[2]
        l = int(characteristics[3])  # Speaker ID
        m = int(characteristics[4])  # Microphone ID

        # Calculate speaker position (xl, yl, zl)
        xl = round(-150 * math.sin((2 * l - 1) * math.pi / 60)) + self.grid_center[0]
        yl = round(150 * math.cos((2 * l - 1) * math.pi / 60)) + self.grid_center[1]
        zl = 145  # Fixed speaker height

        # Initialize microphone position
        xm, ym, zm = 0, 0, 145  # Default microphone height is fixed at 145

        # Calculate microphone positions based on array type and zone
        if array == 'Planar':
            if zone == "A":
                xm = -14 + (4 * ((m - 1) % 8)) - 40 + self.grid_center[0]
                ym = 14 - (4 * math.floor(((m - 1) / 8))) + self.grid_center[1]
            elif zone == "B":
                xm = -14 + (4 * ((m - 1) % 8)) + 40 + self.grid_center[0]
                ym = 14 - (4 * math.floor(((m - 1) / 8))) + self.grid_center[1]
            elif zone == "C":
                xm = -14 + (4 * ((m - 1) % 8)) + self.grid_center[0]
                ym = 14 - (4 * math.floor(((m - 1) / 8))) + 40 + self.grid_center[1]
            elif zone == "D":
                xm = -14 + (4 * ((m - 1) % 8)) + self.grid_center[0]
                ym = 14 - (4 * math.floor(((m - 1) / 8))) - 40 + self.grid_center[1]
            elif zone == "E":
                xm = -14 + (4 * ((m - 1) % 8)) + self.grid_center[0]
                ym = 14 - (4 * math.floor(((m - 1) / 8))) + self.grid_center[1]

        elif array == 'Circular':
            # Determine microphone radius based on ID
            rm = 12 - (2 * math.floor((m - 1) / 30))
            angle = ((m - 1) % 30) * 2 * math.pi / 30  # Angle of the microphone
            if zone == "A":
                xm = -rm * math.sin(angle) - 40 + self.grid_center[0]
                ym = rm * math.cos(angle) + self.grid_center[1]
            elif zone == "B":
                xm = -rm * math.sin(angle) + 40 + self.grid_center[0]
                ym = rm * math.cos(angle) + self.grid_center[1]
            elif zone == "C":
                xm = -rm * math.sin(angle) + self.grid_center[0]
                ym = rm * math.cos(angle) + 40 + self.grid_center[1]
            elif zone == "D":
                xm = -rm * math.sin(angle) + self.grid_center[0]
                ym = rm * math.cos(angle) - 40 + self.grid_center[1]
            elif zone == "E":
                xm = -rm * math.sin(angle) + self.grid_center[0]
                ym = rm * math.cos(angle) + self.grid_center[1]

        # Return rounded positions for both speaker and microphone, along with RT60 value
        return [round(xl), round(yl), round(zl), round(xm), round(ym), round(zm), self.rt60]

    def return_embedding(self, characteristics):
        """
        Generate an embedding for the room based on its configuration and input characteristics.
        characteristics: List containing [zone, array type, speaker ID, microphone ID].
        """
        lis_mic_vector = self.get_m_l_position(characteristics)  # Get positions of mic and speaker
        room_vector = self.return_vector()  # Get room dimensions
        return room_vector + lis_mic_vector  # Combine room and position information

def return_room(emb):
    """
    Determine the room type based on the first element of the room embedding.
    emb: Room embedding vector.
    """
    name = None
    if emb[0] == 490:
        name = 'Anechoic'
    elif emb[0] == 355:
        name = 'Small'
    elif emb[0] == 736:
        name = 'Medium'
    elif emb[0] == 994:
        name = 'Large'
    elif emb[0] == 600:
        name = 'Box'

    return name  # Return the room name

if __name__ == "__main__":
    # Example initialization of predefined room configurations
    Anechoic_Room = UTSRoom(490, 722, 490, 722, 529, [245, 361], 45)
    Small_Room = UTSRoom(355, 410, 401, 378, 300, [175.5, 205], 497)
    Medium_Room = UTSRoom(736, 520, 650, 434.5, 300, [368, 217.5], 659)
    Large_Room = UTSRoom(994, 923, 1087, 1022, 300, [497, 486.25], 1281)
    Box_Room = UTSRoom(600, 1175, 600, 1175, 300, [300, 881.25], 667)

    zones = ['A', 'B', 'C', 'D', 'E']
    arrays = ['Planar', 'Circular']

    for m in range(1, 65):
        # Example vector generation for a specific room configuration
        vector = Small_Room.return_embedding(['LargeMeetingRoom', 'B', 'Circular', 22, m])
        # Print the generated embedding
        print(m, vector)





























