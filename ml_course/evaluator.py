import concurrent.futures
import math
import multiprocessing
import threading
from dataclasses import dataclass
import random
from typing import List, Tuple

import enums
import feature_vector
from enums import Concept
from helpers import Helpers
from learner import ILearner


@dataclass
class Sample:
    correct_guesses: int = 0
    total_guesses: int = 0
    correct_guesses_per_concept = {concept: 0 for concept in Concept if concept != Concept.NONE}
    guesses_per_concept = {concept: 0 for concept in Concept if concept != Concept.NONE}

    @property
    def false_guesses(self):
        return self.total_guesses - self.correct_guesses

    @property
    def err_e(self):
        return self.false_guesses / self.total_guesses

    @property
    def suc_e(self):
        return self.correct_guesses / self.total_guesses

    def err_e_for_concept(self, concept: Concept):
        return self.correct_guesses_per_concept[concept]


# todo this calsses functionality and api will likely change a little,
# as it will also be responsible for splitting the datasets in the future
class Evaluator:

    def __init__(self, learner: ILearner):
        self.learner = learner
        self.samples = []
        self.lock = threading.Lock()

    @staticmethod
    def is_correct(sample, classified, expected):
        if classified != Concept.NONE:
            if classified == expected:
                sample.correct_guesses += 1
                # self.total_guesses += 1  # having it here means we do ommit None guesses and not count them at all
                sample.correct_guesses_per_concept[classified] += 1
            sample.guesses_per_concept[classified] += 1
        # self.total_guesses += 1  # having this here means we count the falsly as none classified as errors!
        sample.total_guesses += 1  # having this here means we count the falsly as none classified as errors!

    def print_stats(self):
        print(f"Median Correctness : {self.median_correctness()}")
        print(f"Median Error: {self.median_error()}")

    # todo could be cached with different datastructure
    def median_error(self):
        akku = 0
        for stat in self.samples:
            akku += stat.err_e
        return akku / len(self.samples)

    def median_correctness(self):
        return 1 - self.median_error()

    def _stddev(self):
        akku = 0
        median_error = self.median_error()
        for stat in self.samples:
            akku += (stat.err_e - median_error) ** 2

        return math.sqrt(akku / len(self.samples))

    def error_confidence(self, target: enums.Confidence = enums.Confidence.C50) -> (float, float):
        """ returns the confidence interval our ERROR lies in for the selected target
        as tuple of floats (lower, upper) """
        conf = target * (self._stddev() / math.sqrt(len(self.samples)))
        conf_upper = self.median_error() + conf
        conf_lower = self.median_error() - conf
        return conf_lower, conf_upper

    def sucess_confidence(self, target: enums.Confidence = enums.Confidence.C50) -> (float, float):
        conf_lower, conf_upper = self.error_confidence(target)
        return 1 - conf_upper, 1 - conf_lower

    # todo this can be made much easier and likely also faster...
    @staticmethod
    def split_sets(vectors: List[feature_vector.FeatureVector], seed=None, split=Tuple[int, int], sample_size=-1) -> (
    List[feature_vector.FeatureVector], List[feature_vector.FeatureVector]):
        a, b = split
        vec_len = len(vectors)
        indices = set(range(vec_len))

        if sample_size < 0:
            sample_size = vec_len
        total = a + b
        trainings_size = int(sample_size * (a / total))
        test_size = sample_size - trainings_size

        if seed is not None:
            random.seed(seed)
        training_indices = random.sample(sorted(indices), trainings_size)
        leftover_indices = indices.difference(set(training_indices))
        test_indices = random.sample(sorted(leftover_indices), test_size)

        # todo maybe this can be done directly in sample if it takes to much time.
        training_set = [vectors[i] for i in training_indices]
        test_set = [vectors[i] for i in test_indices]
        return training_set, test_set

    def _evaluate_helper(self, vectors: List[feature_vector.FeatureVector]):
        for vector in vectors:
            sample = Sample()
            res = self.learner.classify(vector)
            self.is_correct(sample, res, vector.concept)
            self.samples.append(sample)
            with self.lock:
                self.samples.append(sample)
        # self.print_stats()

    def evaluate(self, training_set: List[feature_vector.FeatureVector],
                 test_set: List[feature_vector.FeatureVector]) -> List[Sample]:

        self.learner.learn(training_set)

        tasks = []
        workers = multiprocessing.cpu_count() - 2
        with concurrent.futures.ThreadPoolExecutor(workers) as executor:
            for vectors in Helpers.chunk_list(test_set, len(training_set) // workers):
                tasks.append(executor.submit(self._evaluate_helper, vectors))

        concurrent.futures.wait(tasks)

        return self.samples


# todo the evaluator is likely flawed still. Recheck slides and formular implementations
if __name__ == "__main__":
    experiments = [
        [
            (Concept.STOP, Concept.VORFAHRT_GEWAEHREN),
            (Concept.FAHRTRICHTUNG_RECHTS, Concept.FAHRTRICHTUNG_RECHTS),
            (Concept.VORFAHRT_GEWAEHREN, Concept.VORFAHRT_GEWAEHREN)
        ],
        [
            (Concept.STOP, Concept.STOP),
            (Concept.VORFAHRT_RECHTS, Concept.VORFAHRT_RECHTS),
            (Concept.FAHRTRICHTUNG_LINKS, Concept.VORFAHRT_RECHTS)
        ]
    ]

    e = Evaluator(None)

    for experiment in experiments:
        for a, b in experiment:
            e.is_correct(a, b)
            e._new_sample()

    print(e.error_confidence())
