import concurrent.futures
import json
import multiprocessing
import os
import threading
from constants import STOCK_IMAGES_PATH
from PIL import Image

from extractor import Extractor
from preprocessor import Preprocessor


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
                    if ("80x60" in image_path and "80x60+1" not in image_path and "80x60+2" not in image_path and
                            "80x60+3" not in image_path):
                        print(image_path)
                        image = Image.open(image_path)
                        tasks.append(executor.submit(Extractor.extract, image, preprocessor))

                        count += 1
                        print(f"Queued task for {filename}")
                        print(f"Count at {count}")
        # Wait for all tasks to complete
        for task in concurrent.futures.as_completed(tasks):
            jsons.append(task.result().to_json())
    with open("../generated/jsons/test.80x60.fixed_extraction.bak.json", 'w') as file:
        json.dump(jsons, file, indent=4)

    jsons = []
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
                    if "3500" in image_path:
                        image = Image.open(image_path)
                        tasks.append(executor.submit(Extractor.extract, image, preprocessor))

                        count += 1
                        print(f"Queued task for {filename}")
                        print(f"Count at {count}")
        # Wait for all tasks to complete
        for task in concurrent.futures.as_completed(tasks):
            jsons.append(task.result().to_json())
    with open("../generated/jsons/test.3500.fixed_extraction.bak.json", 'w') as file:
        json.dump(jsons, file, indent=4)
