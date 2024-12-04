from typing import Dict


class Range:

    __RANGE_KEY = "__ranges"

    def __init__(self, min_value, max_value):
        self.min = min_value
        self.max = max_value

    def __call__(self, func):
        def wrapper(instance, value):
            if Range.__RANGE_KEY not in instance.__dict__:
                instance.__dict__[Range.__RANGE_KEY] = {}
            instance.__dict__[Range.__RANGE_KEY][func.__name__.replace("_", "")] = (self.min, self.max)
            return func(instance, value)
        return wrapper

    @staticmethod
    def get_range(obj, key: str):

        # input is object
        if not isinstance(obj, Dict):
            obj = obj.__dict__

        return obj.get(Range.__RANGE_KEY, {}).get(key.replace("_", ""))

if __name__ == "__main__":

    class t:
        def __init__(self):
            self.someprop = 10

        @property
        def someprop(self):
            return self.someprop

        @someprop.setter
        @Range(0, 100)
        def someprop(self, value):
            self._someprop = value


    tt = t()
    tt.someprop = 23
    print(Range.get_range(tt, "someprop"))
