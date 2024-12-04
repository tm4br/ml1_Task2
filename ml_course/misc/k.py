from pathlib import Path
from enums import Confidence
from EuclideanLearner import EuclideanLearner
from evaluator import Evaluator
from helpers import Helpers
from learner import LearnerWrapper
from plotter import Plotter
import constants

seed = 2221154
sample_size = 200
max_k = 100
n = 10
confidence = Confidence.C99
excluded_features = []#["_red_pixels", "_yellow_pixels", "_blue_pixels", "_black_pixels"]
learners = [LearnerWrapper(EuclideanLearner(k=k, exclude=excluded_features)) for k in range(1, max_k)]
evaluators = [Evaluator(learner) for learner in learners]
vectors = Helpers.read_json_file("../generated/jsons/test.16.fixed_extraction.bak.json")  # needs to be retireved via extraction first

for i in range(1, n+1):
    print(f"at iter: {i}")
    trainings_set, test_set = Evaluator.split_sets(vectors, seed*i, (9, 1), sample_size)

    for k, evaluator in enumerate(evaluators):
        print(f"at k: {k+1}")
        evaluator.evaluate(trainings_set, test_set)

correctness = []
confidences = []
for evaluator in evaluators:
    evaluator.sucess_confidence(confidence)
    correctness.append(evaluator.median_correctness())
    confidences.append(evaluator.sucess_confidence(confidence))

plotter = Plotter()
plotter.plot(
    list(range(1, max_k)),
    correctness,
    f"K with {confidence}% confidence, sample size {sample_size} with 1:1 split repeated {n-1} times,\n seed: {seed}, all features, resolution {constants.RESOLUTION}, euclidean",
    "k",
    "Accuracy",
    confidences,
    Path("../generated/k.svg").resolve(),
    "line"
)