import os
import random
import sys
from pathlib import Path

from constants import PREPROCESSED_IMAGES_PATH, STOCK_IMAGES_PATH
from enums import Confidence
from EuclideanLearner import EuclideanLearner
from evaluator import Evaluator
from helpers import Helpers
from learner import LearnerWrapper
from plotter import Plotter

# 100 verschiedene aufteilungen split 1 zu 1, 1000 samples
# für jede mit variirendem K einmal alle testbeispiele evaluieren
# alles plotten

seed = random.randint(0, 2**32 - 1)
sample_size = 1000
n = 10
excluded_features = ["_red_pixels", "_yellow_pixels", "_blue_pixels", "_black_pixels"]
learners = [LearnerWrapper(EuclideanLearner(k=1, exclude=excluded_features)) for _ in range(5)]
evaluators = [Evaluator(learner) for learner in learners]

file_lst = []
for root, dirs, files in os.walk("C:\\Users\\user\\Desktop\\vectors new"):
    for filename in files:
        if filename.lower().endswith('json'):
            file_lst.append(os.path.join(root, filename))

for json_file, evaluator in zip(file_lst, evaluators):
    vectors = Helpers.read_json_file(json_file)
    for i in zip(range(0, n)):
        trainings_set, test_set = Evaluator.split_sets(vectors, seed, (9, 1), sample_size)

        evaluator.evaluate(trainings_set, test_set)

resolutions = ["8x8", "16x16", "32x32", "64x64", "128x128"]

correctness = []
confidences = []
for evaluator in evaluators:
    evaluator.sucess_confidence(Confidence.C99)
    evaluator.median_correctness()
    correctness.append(evaluator.median_correctness())
    confidences.append(evaluator.sucess_confidence(Confidence.C50))

plotter = Plotter()
plotter.plot(
    resolutions,
    correctness,
    f"sample size {sample_size} with 9:1 split repeated {n} times, seed: {seed}",
    "Resolution",
    "Accuracy",
    confidences,
    Path("../generated/k.svg").resolve(),
    "bar"
)