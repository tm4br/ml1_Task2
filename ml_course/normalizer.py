from typing import Dict
from feature_vector import FeatureVector
from range import Range

class INormalizer:

    def normalize(self, vector: FeatureVector | Dict) -> Dict:
        pass


class FloatNormalizer(INormalizer):
    """ normalizes every value of the feature vector to lay in between 0 and 1"""

    def normalize(self, vector: FeatureVector | Dict) -> Dict:

        if isinstance(vector, FeatureVector):
            vector = vector.__dict__

        for key, value in vector.items():
            r = Range.get_range(vector, key)
            if r:
                min_, max_ = r
                if isinstance(value, float) and value > 1 or value < 0:
                    vector[key] = value / max_
                if isinstance(value, int):
                    vector[key] = value / max_
                else:
                    vector[key] = value

        return vector


class MinMaxNormalizer(INormalizer):
    """ normalizes every value of the feature vector to lay in between 0 and 1"""

    def normalize(self, vector: FeatureVector | Dict) -> Dict:

        if isinstance(vector, FeatureVector):
            vector = vector.__dict__

        for key, value in vector.items():
            r = Range.get_range(vector, key)
            if r:
                min_, max_ = r
                if isinstance(value, float) or isinstance(value, int):
                    vector[key] = (value - min_) / (max_ - min_)
                else:
                    vector[key] = value

        return vector


class DownsamplingNormalizer(INormalizer):
    """ Normalizes floats to fall in between int range 1 to 10. and all ints to fall within range 1 to 100"""
    def normalize(self, vector: FeatureVector | Dict) -> Dict:

        if isinstance(vector, FeatureVector):
            vector = vector.__dict__

        for key, value in vector.items():
            vector[key] = self._normalize_value(value)

        return vector

    def _normalize_value(self, value):
        if isinstance(value, float):
            # Map float values to an integer between 1 and 10
            return max(1, min(10, int(value * 10)))
        elif isinstance(value, int):
            # Map integer values to an integer between 1 and 100
            return max(1, min(100, value))
        elif isinstance(value, list):
            # Recursively normalize each item in the list
            return [self._normalize_value(item) for item in value]
        elif isinstance(value, dict):
            # Normalize numeric values in dictionary with string keys
            return {
                k: self._normalize_value(v) if isinstance(v, (float, int)) else v
                for k, v in value.items()
            }
        else:
            return value  # Leave non-numeric types unchanged
