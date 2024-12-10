import concurrent.futures
import multiprocessing
import os.path
import threading
from typing import List

import numpy as np
from PIL import Image

from ml_course.constants import STOCK_IMAGES_PATH


class ScikitExtractor:

    @staticmethod
    def preprocess(image: Image) -> np.ndarray:
        """Resize the image and return its raw RGB values as a flattened numpy array"""
        image = image.resize((128, 128))
        return np.array(image).flatten()

    @staticmethod
    def extract_images_from_directory(directory: str, image_limit: int = 50) -> List[np.ndarray]:
        lock = threading.Lock()
        images = []
        tasks = []

        with concurrent.futures.ThreadPoolExecutor(max_workers=multiprocessing.cpu_count() -2) as executor:
            for root, _, files in os.walk(directory):
                count = 0
                for filename in files:
                    if filename.lower().endswith(('bmp', 'jpg')):
                        if count >= image_limit:
                            break

                        image_path = os.path.join(root, filename)
                        image = Image.open(image_path)
                        tasks.append(executor.submit(ScikitExtractor.preprocess, image))

                        count += 1
                        print(f"Queued task for {filename}")
                        print(f"Count at {count}")

            #Wait for all tasks to complete
            for task in concurrent.futures.as_completed(tasks):
                images.append(task.result())

        return images


    if __name__ == "__main__":
        images = extract_images_from_directory(STOCK_IMAGES_PATH)
        print(f"Extracted {len(images)} images.")