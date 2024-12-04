import hashlib
import json
import numbers
from dataclasses import dataclass
from typing import Dict, List

import constants
from enums import Concept
from range import Range


@dataclass
class FeatureVector():

    __SERIALIZATION_UID = ""

    def __init__(self):
        self.concept = Concept.STOP
        self._red_pixels = 0
        self._black_pixels = 0
        self._blue_pixels = 0
        self._yellow_pixels = 0

        self._red_proportions = 0.0
        self._black_proportions = 0.0
        self._blue_proportions = 0.0
        self._yellow_proportions = 0.0

        self._black_connected_components = 0
        self._red_connected_components = 0
        self._blue_connected_components = 0
        self._yellow_connected_components = 0

        self.clusters = []

    @property
    def red_pixels(self):
        return self._red_pixels

    @red_pixels.setter
    @Range(0, constants.RESOLUTION*constants.RESOLUTION)
    def red_pixels(self, value):
        self._red_pixels = value

    @property
    def blue_pixels(self):
        return self._blue_pixels

    @blue_pixels.setter
    @Range(0, constants.RESOLUTION*constants.RESOLUTION)
    def blue_pixels(self, value):
        self._blue_pixels = value

    @property
    def black_pixels(self):
        return self._black_pixels

    @black_pixels.setter
    @Range(0, constants.RESOLUTION*constants.RESOLUTION)
    def black_pixels(self, value):
        self._black_pixels = value

    @property
    def yellow_pixels(self):
        return self._yellow_pixels

    @yellow_pixels.setter
    @Range(0, constants.RESOLUTION*constants.RESOLUTION)
    def yellow_pixels(self, value):
        self._yellow_pixels = value

    @property
    def red_proportions(self):
        return self._red_proportions

    @red_proportions.setter
    @Range(0, 1)
    def red_proportions(self, value):
        self._red_proportions = value

    @property
    def blue_proportions(self):
        return self._blue_proportions

    @blue_proportions.setter
    @Range(0, 1)
    def blue_proportions(self, value):
        self._blue_proportions = value

    @property
    def yellow_proportions(self):
        return self._yellow_proportions

    @yellow_proportions.setter
    @Range(0, 1)
    def yellow_proportions(self, value):
        self._yellow_proportions = value

    @property
    def black_proportions(self):
        return self._black_proportions

    @black_proportions.setter
    @Range(0, 1)
    def black_proportions(self, value):
        self._black_proportions = value

    @property
    def black_connected_components(self):
        return self._black_connected_components

    @black_connected_components.setter
    @Range(1, 25)
    def black_connected_components(self, value):
        self._black_connected_components = value

    @property
    def red_connected_components(self):
        return self._red_connected_components

    @red_connected_components.setter
    @Range(1, 25)
    def red_connected_components(self, value):
        self._red_connected_components = value

    @property
    def blue_connected_components(self):
        return self._blue_connected_components

    @blue_connected_components.setter
    @Range(1, 25)
    def blue_connected_components(self, value):
        self._blue_connected_components = value

    @property
    def yellow_connected_components(self):
        return self._yellow_connected_components

    @yellow_connected_components.setter
    @Range(1, 25)
    def yellow_connected_components(self, value):
        self._yellow_connected_components = value

    @staticmethod
    def generate_flat_uuid(key, length=5):
        hash_object = hashlib.sha256(key.encode())
        hex_dig = hash_object.hexdigest()
        return key+"_"+hex_dig[:length]

    def flatten(self) -> Dict[str, numbers.Number]:
        res = {}
        for k, v in self.__dict__.items():
            if not k.startswith("__"):
                self.__flatten_helper(res, k, v)
        return res

    def __flatten_helper(self, res, key, value):
        if isinstance(value, numbers.Number):
            res[key] = value
        elif isinstance(value, Concept) or isinstance(value, str): # only temp
            res[key] = value
        elif isinstance(value, Dict):
            for k, v in value.items():
                k = self.generate_flat_uuid(key + k)
                self.__flatten_helper(res, k, v)
        elif isinstance(value, List):
            for i, el in enumerate(value):
                k = self.generate_flat_uuid(key + str(i))
                self.__flatten_helper(res, k, el)

    def to_json(self) -> str:
        data = self.__dict__.copy()
        data['concept'] = self.concept.name  # must be handled explicitly
        try :
            data['clusters'] = [cl.to_dict() for cl in self.clusters]
        except:
            data['clusters'] = self.clusters # TODO RAUSFINDEN WARUM
        return json.dumps(data, indent=4)

    # should be similar to serialization uid in java
    def get_serialization_uid(self):
        if FeatureVector.__SERIALIZATION_UID == "":
            class_name = self.__class__.__name__
            attributes = [attr for attr in dir(self) if not callable(getattr(self, attr)) and not attr.startswith("__")]
            unique_string = class_name + "".join(sorted(attributes))
            hash_object = hashlib.sha256(unique_string.encode())
            FeatureVector.__SERIALIZATION_UID = hash_object.hexdigest()
        return FeatureVector.__SERIALIZATION_UID

    @classmethod
    def from_json(cls, json_data):
        dict_ = json.loads(json_data)
        concept = Concept[dict_["concept"]]
        del dict_["concept"]
        fv = FeatureVector()
        for k, v in dict_.items(): # this needs to be done explicitly to avoid overwriting annotation generated entries
            if hasattr(fv.__class__, k):
                setattr(fv, k, v)
            else:
                # Set other attributes directly if they aren't properties
                fv.__dict__[k] = v
        fv.concept = concept
        return fv

