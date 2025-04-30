"""
This script defines classes for modeling the geometry and acoustic layout of rooms,
including general quadrilaterals and structured rooms used in the UTS database.

Features:
- Compute room embeddings combining geometry, speaker, and microphone positions.
- Encode reverberation time (RT60) into the feature.
- Support planar and circular microphone array configurations.

Note:
This implementation is tightly coupled to the naming structure and spatial assumptions
from the specific RIR dataset used in the project.
"""

import math
from typing import List

class Quadrilateral:
    """
    Represents a basic 2D quadrilateral using its side lengths.

    Attributes:
        a (float): Length of side a.
        b (float): Length of side b.
        c (float): Length of side c.
        d (float): Length of side d.
    """
    def __init__(self, a: float, b: float, c: float, d: float) -> None:
        self.a = a
        self.b = b
        self.c = c
        self.d = d

class Room(Quadrilateral):
    """
    Represents a 3D room based on a quadrilateral base and a height.

    Attributes:
        height (float): Height of the room.
        vector (List[float]): Rounded room dimensions.
    """
    def __init__(self, a: float, b: float, c: float, d: float, height: float) -> None:
        super().__init__(a, b, c, d)
        self.height = height
        self.vector: List[float] = []
        self.set_vector()

    def set_vector(self) -> None:
        """
        Compute and store the room's dimension vector.
        """
        self.vector = [round(self.a), round(self.b), round(self.c), round(self.d), round(self.height)]

    def return_vector(self) -> List[float]:
        """
        Return the stored room dimension vector.

        Returns:
            List[float]: Rounded dimensions [a, b, c, d, height].
        """
        return self.vector

class UTSRoom(Room):
    """
    Represents a room configuration used in the UTS RIR database.
    Extends Room with logic to compute embeddings based on microphone/speaker placement.

    Attributes:
        grid_center (List[float]): Central grid reference for position placement.
        rt60 (float): Reverberation time (ms).
    """
    def __init__(self, a: float, b: float, c: float, d: float, height: float, grid_center: List[float], rt60: float) -> None:
        super().__init__(a, b, c, d, height)
        self.grid_center = grid_center
        self.rt60 = rt60

    def get_m_l_position(self, characteristics: List) -> List[float]:
        """
        Compute microphone and speaker 3D positions from the given configuration.

        Args:
            characteristics (List): Format [room, zone, array type, speaker, mic].

        Returns:
            List[float]: [xl, yl, zl, xm, ym, zm, rt60] 3D positions and reverberation time.
        """
        zone = characteristics[1]
        array = characteristics[2]
        l = int(characteristics[3])
        m = int(characteristics[4])

        xl = round(-150 * math.sin((2 * l - 1) * math.pi / 60)) + self.grid_center[0]
        yl = round(150 * math.cos((2 * l - 1) * math.pi / 60)) + self.grid_center[1]
        zl = 145

        xm, ym, zm = 0, 0, 145

        if array == 'Planar':
            col_offset = 4 * ((m - 1) % 8)
            row_offset = 4 * ((m - 1) // 8)
            if zone == "A":
                xm = -14 + col_offset - 40 + self.grid_center[0]
                ym = 14 - row_offset + self.grid_center[1]
            elif zone == "B":
                xm = -14 + col_offset + 40 + self.grid_center[0]
                ym = 14 - row_offset + self.grid_center[1]
            elif zone == "C":
                xm = -14 + col_offset + self.grid_center[0]
                ym = 14 - row_offset + 40 + self.grid_center[1]
            elif zone == "D":
                xm = -14 + col_offset + self.grid_center[0]
                ym = 14 - row_offset - 40 + self.grid_center[1]
            elif zone == "E":
                xm = -14 + col_offset + self.grid_center[0]
                ym = 14 - row_offset + self.grid_center[1]

        elif array == 'Circular':
            rm = 12 - 2 * ((m - 1) // 30)
            angle = ((m - 1) % 30) * 2 * math.pi / 30
            x_offset = -rm * math.sin(angle)
            y_offset = rm * math.cos(angle)
            if zone == "A":
                xm = x_offset - 40 + self.grid_center[0]
                ym = y_offset + self.grid_center[1]
            elif zone == "B":
                xm = x_offset + 40 + self.grid_center[0]
                ym = y_offset + self.grid_center[1]
            elif zone == "C":
                xm = x_offset + self.grid_center[0]
                ym = y_offset + 40 + self.grid_center[1]
            elif zone == "D":
                xm = x_offset + self.grid_center[0]
                ym = y_offset - 40 + self.grid_center[1]
            elif zone == "E":
                xm = x_offset + self.grid_center[0]
                ym = y_offset + self.grid_center[1]

        return [round(xl), round(yl), round(zl), round(xm), round(ym), round(zm), self.rt60]

    def return_embedding(self, characteristics: List) -> List[float]:
        """
        Generate embedding vector for a room configuration.

        Args:
            characteristics (List): Format [room, zone, array type, speaker, mic].

        Returns:
            List[float]: Embedding combining room and mic/speaker layout.
        """
        lis_mic_vector = self.get_m_l_position(characteristics)
        room_vector = self.return_vector()
        return room_vector + lis_mic_vector




























