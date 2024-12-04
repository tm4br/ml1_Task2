import hashlib
import os
from pathlib import Path

import PIL
import numpy as np
from PIL import Image, PngImagePlugin
from typing import Dict

import constants
from constants import PREPROCESSED_IMAGES_PATH
from enums import Colors
from feature_vector import FeatureVector
from helpers import Helpers


class Preprocessor:
    PREPROCESSED = Path(PREPROCESSED_IMAGES_PATH).resolve()
    VECTOR_VERSION = "FV_Version"
    PREPROCESSOR_VERSION = "Preprocessor_Version"
    METADATA_VECTOR = "FV"
    __SERIALIZATION_UID = ""

    def __init__(self):
        pass

    @staticmethod
    def preprocess(image: PIL.Image) -> (PIL.Image, Dict[Colors, int]):
        """ Crops the image and sets the color space returns image as numpy array """

        image = image.resize((128, 128))

        # quantize
        color_count = {color: 0 for color in Colors}
        new_img = Image.new("RGB", image.size)
        for x in range(image.width):
            for y in range(image.height):
                original_color = image.getpixel((x, y))
                new_color = Helpers.get_color(original_color)
                color_count[new_color] += 1
                new_img.putpixel((x, y), new_color.value)

        # crop
        non_black_mask = (np.array(new_img)[:, :, :3] > 20).any(axis=2)
        if np.any(non_black_mask):
            coords = np.argwhere(non_black_mask)
            y_min, x_min = coords.min(axis=0)
            y_max, x_max = coords.max(axis=0) + 1  # +1 because slicing is exclusive at the end
            new_img = new_img.crop((x_min, y_min, x_max, y_max))
        else:
            print(f"Non croppable image defaulting to default")

        new_img = new_img.resize((constants.RESOLUTION.value, constants.RESOLUTION.value))

        return new_img, color_count

    @staticmethod
    def preprocessed_filepath_helper(original: PIL.Image) -> Path:
        """
        For the requested image returns the respective filepath of the preprocessed version
        The filepath is relative and matches the 4 fast folgers of the original folder structure
        :param original: image the preprocessed filepath should be generated for
        """
        path_parts = str(original.filename).split("\\")

        # preprocessed image must be png
        path_parts[-1] = path_parts[-1].replace(".bmp", ".png")
        path_parts[-1] = path_parts[-1].replace(".jpg", ".png")

        # only the 4 last elements of the path are returned (like in ds structure)
        return Path("\\".join(path_parts[-4:]))

    def get_serialization_uid(self) -> str:
        """
        UID specific to this object version at runtime.
        Applied only on properties - not methods
        """
        if Preprocessor.__SERIALIZATION_UID == "":
            class_name = self.__class__.__name__
            attributes = [attr for attr in dir(self) if not callable(getattr(self, attr)) and not attr.startswith("__")]
            unique_string = class_name + "".join(sorted(attributes))
            hash_object = hashlib.sha256(unique_string.encode())
            Preprocessor.__SERIALIZATION_UID = hash_object.hexdigest()
        return Preprocessor.__SERIALIZATION_UID

    def persist_preprocessed_image(self, image: PIL.Image, vector: FeatureVector):
        """Save image to disk, attach serialization uid and feature vector metadata"""
        metadata = {
            Preprocessor.VECTOR_VERSION: vector.get_serialization_uid(),
            Preprocessor.PREPROCESSOR_VERSION: self.get_serialization_uid(),
            Preprocessor.METADATA_VECTOR: vector.to_json()
        }

        png_info = PngImagePlugin.PngInfo()
        for key, value in metadata.items():
            png_info.add_text(key, str(value))

        filename = Preprocessor.preprocessed_filepath_helper(image)  # todo change extension in all places
        filepath = Preprocessor.PREPROCESSED / filename  # new filepath - keep structure similar to datasets
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        image.save(filepath, "PNG", pnginfo=png_info)

    def extract_metadata(self, image: PIL.Image, vector: FeatureVector) -> FeatureVector | None:
        """ look inside the preprocessing folder for a similar image (determined by path)
        If image exists attempt metadata extraction. If metadata is valid for this processing config - extract it,
        else return None as indicator, that no valid metadata could be extracted.
        """
        filename = Preprocessor.preprocessed_filepath_helper(image)
        filepath = Preprocessor.PREPROCESSED / filename
        if os.path.exists(filepath):
            with Image.open(filepath) as preprocessed_image:
                if Preprocessor.VECTOR_VERSION in preprocessed_image.info and\
                        Preprocessor.PREPROCESSOR_VERSION in preprocessed_image.info and\
                        Preprocessor.METADATA_VECTOR in preprocessed_image.info:
                    if preprocessed_image.info[Preprocessor.VECTOR_VERSION] == vector.get_serialization_uid() and\
                            preprocessed_image.info[Preprocessor.PREPROCESSOR_VERSION] == self.get_serialization_uid():
                        print(f"Preprocessed info used for: {filename}")
                        return FeatureVector.from_json(preprocessed_image.info[Preprocessor.METADATA_VECTOR])
        return None
