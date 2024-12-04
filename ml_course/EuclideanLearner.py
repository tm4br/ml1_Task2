import random
import time
from pathlib import Path
from typing import List, Dict
import math

from enums import Concept, Confidence
from evaluator import Evaluator
from feature_vector import FeatureVector
from helpers import Helpers
from learner import ILearner, LearnerWrapper
from normalizer import INormalizer, FloatNormalizer, DownsamplingNormalizer, MinMaxNormalizer
from plotter import Plotter


class EuclideanLearner(ILearner):


    def __init__(self, exclude, k=3, normalizer: INormalizer = None):
        self.data = []
        self.excluded = exclude
        self.normalizer = normalizer
        self.k = k
        if self.k == 0:
            raise ValueError("K can not be 0")

    def learn(self, vectors: List[FeatureVector] | List[Dict]):
        if len(vectors) < self.k:
            raise ValueError("K can not be greater than sample size")

        if len(vectors) > 0 and isinstance(vectors[0], FeatureVector):
            converted_vecs = []
            for vec in vectors:
                converted_vecs.append(vec.flatten())
            vectors = converted_vecs

        self.data = vectors

    def classify(self, vector: FeatureVector | Dict) -> Concept:
        d = []

        if isinstance(vector, FeatureVector):
            vector = vector.flatten()

        for v in self.data:
            dist = 0
            for key, value in v.items():
                if not key.startswith("__") and not isinstance(value, Concept) and key not in self.excluded:
                    try:
                        dist += (vector[key] - value) ** 2
                    except KeyError:
                        print(f"WARNING: Abbreviating vectors detected (key: {key}). This may be intentional, but was it?")
                    # dist += (vector.__dict__[key] - value) **2
            d.append((v, math.sqrt(dist)))

        best_k = sorted(d, key=lambda x: x[1])[0:self.k]
        counts = {concept: 0 for concept in Concept}
        for vec, _ in best_k:
            counts[vec["concept"]] += 1

        return max(counts, key=counts.get)

if __name__ == "__main__":
    vectors = Helpers.read_json_file("generated/jsons/test.16.fixed_extraction.bak.json") # needs to be retireved via extraction first

    # learner = EuclideanLearner(exclude=["red_pixels", "yellow_pixels", "blue_pixels", "black_pixels"])
    # learner = EuclideanLearner(exclude=["red_pixels", "yellow_pixels", "blue_pixels", "black_pixels"])
    # learner = EuclideanLearner(exclude=["red_pixels", "yellow_pixels", "blue_pixels", "black_pixels"], normalizer=FloatNormalizer())
    # learner = EuclideanLearner(exclude=["red_pixels", "yellow_pixels", "blue_pixels", "black_pixels"], normalizer=MinMaxNormalizer())
    # learner = EuclideanLearner(exclude=["red_pixels", "yellow_pixels", "blue_pixels", "black_pixels", "red_proportions", "blue_proportions", "yellow_proportions", "black_proportions"])

    with open("dump.txt", "a") as f:
        correctness = []
        confidences = []
        for i in range(5):
            learner = LearnerWrapper(EuclideanLearner(exclude=[]))
            print("#################")
            f.write("#################\r\n")
            seed = time.time_ns()
            random.seed(seed)
            print(f"seed: {seed}")
            f.write(f"seed: {seed}\n")
            evaluator = Evaluator(learner)
            t, t2 = Evaluator.split_sets(vectors, seed, (9, 1), 1000)
            evaluator.evaluate(t, t2)
            print(evaluator.sucess_confidence(Confidence.C99))
            correctness.append(evaluator.median_correctness())
            confidences.append(evaluator.sucess_confidence(Confidence.C99))
            f.write(f"correctness: {evaluator.median_correctness()}")

    plotter = Plotter()
    plotter.plot(list(range(0, 5)),
                 correctness,
                 "test",
                 "Konfiguration (Dummy)",
                 "Accuracy",
                 confidences,
                 Path("generated/plot.svg").resolve()
                 )
