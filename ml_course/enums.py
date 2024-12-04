import enum
import numbers


class Confidence(enum.Enum):
    C50 = 0.67
    C68 = 1
    C80 = 1.28
    C90 = 1.64
    C95 = 1.96
    C98 = 2.33
    C99 = 2.58

    def __mul__(self, other):
        if isinstance(other, Confidence):
            return self.value * other.value
        if isinstance(other, numbers.Number):
            return self.value * other
        raise Exception("Invalid operation on __mul__ Confidence")

    def __str__(self):
        return f"{self.name.split("C")[1]}"

class Resolutions(enum.Enum):
    R8 = 8
    R16 = 16
    R32 = 32
    R64 = 64
    R128 = 128

    def __mul__(self, other):
        if isinstance(other, Resolutions):
            return self.value * other.value
        if isinstance(other, numbers.Number):
            return self.value * other
        raise Exception("Invalid operation on __mul__ Resolution")

    def __str__(self):
        return f"{self.value}x{self.value}"

class Colors(enum.Enum):
    NONE = (-1,-1,-1)
    # WHITE = (250, 250, 250)
    RED = (255, 0, 0)
    BLUE = (0, 0, 255)
    # GREEN = (0, 255, 0)
    YELLOW = (250, 250, 0)
    BLACK = (0, 0, 0)


class Concept(enum.Enum):
    NONE = -1
    STOP = 0
    VORFAHRT_RECHTS = 1
    VORFAHRTSSTRAßE = 2
    VORFAHRT_GEWAEHREN = 3
    FAHRTRICHTUNG_LINKS = 4
    FAHRTRICHTUNG_RECHTS = 5