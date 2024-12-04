import pickle
from pathlib import Path

from matplotlib import pyplot as plt

from enums import Confidence, Concept
from EuclideanLearner import EuclideanLearner
from evaluator import Evaluator
from helpers import Helpers
from learner import LearnerWrapper
from plotter import Plotter
import constants

seed = 2000481812 #4356713
sample_size = 1000
n = 10
confidence = Confidence.C99
vectors = Helpers.read_json_file("../generated/jsons/test.16.fixed_extraction.bak.json")  # needs to be retireved via extraction first
dummy = LearnerWrapper(EuclideanLearner([], k=1))
dummy.learn([vectors[0]]) # just need one for this
excluded_features = ["_red_pixels", "_yellow_pixels", "_blue_pixels", "_black_pixels"]
splits = list(range(10, 100,10))
evaluators = [Evaluator(LearnerWrapper(EuclideanLearner(excluded_features, k=1))) for _ in splits]

for j, split in enumerate(splits):
    for i in range(1, n+1):
        print(f"at iter: {i}")
        if seed is None:
            s = None
        else:
            s = seed * i

        trainings_set, test_set = Evaluator.split_sets(vectors, s, (split, 100-split), sample_size)
        evaluators[j].evaluate(trainings_set, test_set)


correctness = []
confidences = []
for evaluator in evaluators:
    evaluator.sucess_confidence(confidence)
    correctness.append(evaluator.median_correctness())
    confidences.append(evaluator.sucess_confidence(confidence))


plotter = Plotter()
plotter.plot(
    [f"{s} : {100-s}" for s in splits],
    correctness,
    f"Training to Testdata Ratio with {confidence}% confidence, sample size {sample_size} with varying split repeated {n-1} times,\n seed: {seed}, filtered features, resolution {constants.RESOLUTION}, euclidean",
    "Training to Testdata Ratio",
    "Accuracy",
    confidences,
    Path("../generated/splits.svg").resolve(),
    "bar",
    45,
)

# inverted plot
#sorted_data = sorted(zip(excludable_feature, correctness, confidences), key=lambda x: x[1])
#sorted_excludable_feature, sorted_correctness, sorted_confidences = zip(*sorted_data)
#inverted_correctness = [1-y for y in sorted_correctness]

#plotter.plot(
#    sorted_excludable_feature,
#    inverted_correctness,
#    f"Feature Importance with {confidence}% confidence, sample size {sample_size} with 1:1 split repeated {n-1} times,\n seed: {seed}, filtered features, resolution {constants.RESOLUTION}, euclidean",
#    "Feature",
#    "Accuracy",
#    sorted_confidences,
#    Path("../generated/feature_importance_inverted.svg").resolve(),
#    "bar",  # Keep "bar" plot type
#    45  # Rotate x-axis labels by 45 degrees
#)