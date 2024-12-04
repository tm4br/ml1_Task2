import json
from typing import List

from enums import Colors
from feature_vector import FeatureVector

# Color detection by Dana M�ller

class Helpers:

    @staticmethod
    def get_brightness(rgb):
        r, g, b = rgb
        # Calculate brightness using a weighted average (common formula for perceived brightness)
        return 0.299 * r + 0.587 * g + 0.114 * b

    @staticmethod
    def get_color_bright(rgb):
        r, g, b = rgb
        if abs(r - g) < 10 and abs(g - b) < 10 and abs(r - b) < 10 and r > 245:
            # return Colors.NONE
            pass
        if (r > g + 20 and r > b + 20) and abs(g - b) < 40:
            return Colors.RED
        elif (r >= g and r > b) and (g - b > 40):
            return Colors.YELLOW
        elif abs(r - g) < 30 and abs(g - b) < 30 and abs(r - b) < 30:
            return Colors.BLACK
        elif b > g > r > 170:
            return Colors.BLUE
        return Colors.NONE

    @staticmethod
    def get_color_dark(rgb):
        r, g, b = rgb
        if (r > g + 20 and r > b + 20) and abs(g - b) < 40:
            return Colors.RED
        elif (r >= g and r > b) and (g - b > 40):
            return Colors.YELLOW
        elif abs(r - g) < 15 and abs(g - b) < 15 and abs(r - b) < 15 and r < 30:
            return Colors.BLACK
        elif r < g < b and r < 75:
            return Colors.BLUE
        return Colors.NONE

    @staticmethod
    def get_color(rgb):
        brightness = Helpers.get_brightness(rgb)

        if brightness > 128:  # Adjust threshold as needed
            return Helpers.get_color_bright(rgb)
        else:
            return Helpers.get_color_dark(rgb)

    @staticmethod
    def chunk_list(lst, chunk_size):
        for i in range(0, len(lst), chunk_size):
            yield lst[i:i + chunk_size]

    @staticmethod
    def read_json_file(file_path: str) -> List[FeatureVector]:
        try:
            with open(file_path, 'r') as file:
                data = json.load(file)
                return [FeatureVector.from_json(item) for item in data]  # Convert to FeatureVector instances
        except FileNotFoundError:
            print(f"Error: The file {file_path} was not found.")
            return []
        except json.JSONDecodeError:
            print(f"Error: The file {file_path} contains invalid JSON.")
            return []

