import json


class Cluster:
    def __init__(self):
        self.red_cluster = 0
        self.black_cluster = 0
        self.blue_cluster = 0
        self.yellow_cluster = 0

    @property
    def red_proportions(self):
        return self.red_cluster / sum(self.__dict__.values())

    @property
    def black_proportions(self):
        return self.black_cluster / sum(self.__dict__.values())

    @property
    def blue_proportions(self):
        return self.blue_cluster / sum(self.__dict__.values())

    @property
    def yellow_proportions(self):
        return self.yellow_cluster / sum(self.__dict__.values())

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=4)

    def to_dict(self) -> dict:
        return self.__dict__.copy()

    @classmethod
    def from_json(cls, json_data):
        dict_ = json.loads(json_data)
        cl = Cluster()
        for k, v in dict_.items():  # this needs to be done explicitly to avoid overwriting annotation generated entries
            if hasattr(cl.__class__, k):
                setattr(cl, k, v)
            else:
                cl.__dict__[k] = v
        return cl

if __name__ == "__main__":
    cl = Cluster()
    test = cl.to_json()
    print(test)