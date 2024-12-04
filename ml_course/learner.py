from enums import Concept
from feature_vector import FeatureVector
from typing import List, Dict

from normalizer import INormalizer


class ILearner:

    def learn(self, vectors: List[FeatureVector] | List[dict]):
        """ must guarantee that upon learning new vectors the old vectors are dropped
        """
        pass

    def classify(self, vector: FeatureVector | Dict) -> Concept:
        return Concept.NONE


class LearnerWrapper(ILearner):

    def __init__(self, learner: ILearner, normalizer: INormalizer = None):
        self.data = []
        self.normalizer = normalizer
        self.learner = learner

    def learn(self, vectors: List[FeatureVector] | List[dict]):
        for v in vectors:
            if self.normalizer:
                if isinstance(v, FeatureVector):
                    v = v.flatten()
                normalized = self.normalizer.normalize(v)
                self.data.append(normalized)
            else:
                if isinstance(v, FeatureVector):
                    self.data.append(v.flatten())
                else:
                    self.data.append(v)

        # pass to lower level
        self.learner.learn(self.data)

    def classify(self, vector: FeatureVector | Dict) -> Concept:

        if isinstance(vector, FeatureVector):
            vector = vector.flatten()

        if self.normalizer:
            vector = self.normalizer.normalize(vector)

        return self.learner.classify(vector)
