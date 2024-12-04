import concurrent.futures
import json
import multiprocessing
import os
import threading
import numpy as np
from numpy import ndarray

from cluster import Cluster
from constants import STOCK_IMAGES_PATH
from feature_vector import FeatureVector
import PIL
from PIL import Image
from typing import Dict, Tuple, List
from enums import Colors, Concept
from helpers import Helpers
from preprocessor import Preprocessor


class Extractor:

    @staticmethod
    def proportions(absolut_colors: Dict[Colors, int]) -> Tuple[float, float, float, float]:
        total = sum(absolut_colors.values())
        return (
            absolut_colors[Colors.RED] / total,
            absolut_colors[Colors.BLUE] / total,
            absolut_colors[Colors.BLACK] / total,
            absolut_colors[Colors.YELLOW] / total
        )

    @staticmethod
    def extract_class(filename: str) -> Concept:
        if "Stop" in filename:
            return Concept.STOP
        elif "Vorfahrtsstraße" in filename:
            return Concept.VORFAHRTSSTRAßE
        elif "Vorfahrt gewähren" in filename:
            return Concept.VORFAHRT_GEWAEHREN
        elif "Fahrtrichtung links" in filename:
            return Concept.FAHRTRICHTUNG_LINKS
        elif "Fahrtrichtung rechts" in filename:
            return Concept.FAHRTRICHTUNG_RECHTS
        else:
            return Concept.VORFAHRT_RECHTS

    @staticmethod
    def connected_components(image_array: ndarray):
        height, width, _ = image_array.shape
        component_matrix = np.zeros((height, width))  # just numbers (encoding components)
        component_map = {0: []}
        for y in range(height):
            for x in range(width):
                component = 0
                pixel = image_array[y, x]
                if y > 0:
                    # check y
                    if np.array_equal(image_array[y - 1, x], pixel):
                        component = component_matrix[y - 1, x]
                    else:
                        component = max(component_map.keys()) + 1
                component_matrix[y, x] = component
                component_map.setdefault(component, []).append(
                    pixel.tolist() + [y, x])  # we add pixel and position for reverse lookup!

                if x > 0:
                    # check x
                    if np.array_equal(image_array[y, x - 1], pixel):
                        component = component_matrix[y, x - 1]
                    else:
                        component = max(component_map.keys()) + 1
                    if component_matrix[y, x] > 0 and component_matrix[y, x] != component:
                        # fix component matrix
                        mergable_component = component_matrix[y, x]
                        pixels_in_mergable_component = component_map[mergable_component]
                        for p in pixels_in_mergable_component:
                            y1, x1 = p[3:]  # coords
                            # update p in component matrix (should not be many)
                            component_matrix[y1, x1] = component

                        # fix component map
                        component_map.setdefault(component, []).extend(pixels_in_mergable_component)
                        component_map[mergable_component] = []
                    else:
                        component_matrix[y, x] = component
                        component_map.setdefault(component, []).append(pixel.tolist() + [y, x])

        removable = []
        for k, v in component_map.items():
            if len(v) == 0:
                removable.append(k)
        for r in removable:
            del component_map[r]
        return component_map

    # todo these really are columns instead of clusters - but its okay.
    @staticmethod
    def clustering(image: np.ndarray) -> List[Cluster]:
        num_clusters = 16
        clusters = [Cluster() for _ in range(num_clusters)]
        height, width, _ = image.shape
        for x in range(width):
            for y in range(height):
                idx = y + y*x
                col = Helpers.get_color(image[y, x].tolist())
                if Colors.BLACK == col:
                    clusters[idx % num_clusters].black_cluster += 1
                elif Colors.BLUE == col:
                    clusters[idx % num_clusters].blue_cluster += 1
                elif Colors.RED == col:
                    clusters[idx % num_clusters].red_cluster += 1
                elif Colors.YELLOW == col:
                    clusters[idx % num_clusters].yellow_cluster += 1

        return clusters

    @staticmethod
    def color_count(image: PIL.Image) -> Dict[Colors, int]:
        # quantize
        color_count = {color: 0 for color in Colors}
        for x in range(image.width):
            for y in range(image.height):
                color = image.getpixel((x, y))
                color_count[Helpers.get_color(color)] += 1
        return color_count

    @staticmethod
    def extract(image: PIL.Image, preprocessor: Preprocessor) -> FeatureVector:
        fv = FeatureVector()

        # classify
        c = Extractor.extract_class(image.filename)
        fv.concept = c

        fvm = preprocessor.extract_metadata(image, fv)
        if fvm is not None:
            fv = fvm # vector already calculated.
        else:
            preprocessed, _ = Preprocessor.preprocess(image)
            preprocessed.filename = image.filename

            image_array = np.array(preprocessed)

            colors = Extractor.color_count(preprocessed)

            # clusters
            fv.clusters = Extractor.clustering(image_array)

            # base colors
            fv.red_pixels = colors[Colors.RED]
            fv.blue_pixels = colors[Colors.BLUE]
            fv.black_pixels = colors[Colors.BLACK]
            fv.yellow_pixels = colors[Colors.YELLOW]

            # base color proportions
            fv.red_proportions, fv.blue_proportions, fv.black_proportions, fv.yellow_proportions = Extractor.proportions(
                colors)

            # connected components
            components = Extractor.connected_components(image_array)
            colors = {color: 1 for color in Colors}
            for key, value in components.items():
                color = Helpers.get_color(value[0][0:3])
                colors[color] += 1
            fv.red_connected_components = colors[Colors.RED]
            fv.blue_connected_components = colors[Colors.BLUE]
            fv.black_connected_components = colors[Colors.BLACK]
            fv.yellow_connected_components = colors[Colors.YELLOW]

            preprocessor.persist_preprocessed_image(preprocessed, fv)

        return fv


if __name__ == "__main__":
    lock = threading.Lock()
    jsons = []
    image_limit = 50 # per folder
    tasks = []
    preprocessor = Preprocessor()
    with concurrent.futures.ThreadPoolExecutor(max_workers=multiprocessing.cpu_count() - 2) as executor:
        for root, dirs, files in os.walk(STOCK_IMAGES_PATH):
            count = 0
            for filename in files:
                if filename.lower().endswith(('bmp', "jpg")):
                    if count >= image_limit:
                        break

                    image_path = os.path.join(root, filename)
                    image = Image.open(image_path)
                    tasks.append(executor.submit(Extractor.extract, image, preprocessor))

                    count += 1
                    print(f"Queued task for {filename}")
                    print(f"Count at {count}")
        # Wait for all tasks to complete
        for task in concurrent.futures.as_completed(tasks):
            jsons.append(task.result().to_json())
    with open("generated/jsons/test.16.fixed_extraction.bak.json", 'w') as file:
        json.dump(jsons, file, indent=4)
