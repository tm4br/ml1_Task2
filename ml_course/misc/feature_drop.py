import pickle
from pathlib import Path

from matplotlib import pyplot as plt

from Cal2Learner import Cal2Learner
from enums import Confidence, Concept
from EuclideanLearner import EuclideanLearner
from evaluator import Evaluator
from helpers import Helpers
from learner import LearnerWrapper
from plotter import Plotter
import constants

seed = 7581902571
sample_size = 7500
n = 10
confidence = Confidence.C99
vectors = Helpers.read_json_file("../generated/jsons/test.16.fixed_extraction.bak.json")  # needs to be retireved via extraction first
dummy = LearnerWrapper(EuclideanLearner([], k=1))
dummy.learn([vectors[0]]) # just need one for this
excludable_feature = [f for f in dummy.data[0].keys() if not f.startswith("__") and not f == "concept" and not f.startswith("clusters") and not f.endswith("pixels")]
evaluators = [Evaluator(LearnerWrapper(EuclideanLearner([feature], k=1))) for feature in excludable_feature]

for i in range(1, n+1):
    print(f"at iter: {i}")
    if seed is None:
        s = None
    else:
        s = seed * i

    trainings_set, test_set = Evaluator.split_sets(vectors, s, (9, 1), sample_size)

    for j, feature in enumerate(excludable_feature):
        print(f"at i: {j+1}")
        evaluators[j].evaluate(trainings_set, test_set)


correctness = []
confidences = []
for evaluator in evaluators:
    evaluator.sucess_confidence(confidence)
    correctness.append(evaluator.median_correctness())
    confidences.append(evaluator.sucess_confidence(confidence))

with open("../generated/clusterless_baseline.pkl", "rb") as file:
    data = pickle.load(file)
    baseline_confidences, baseline_corectness = data["confidences"], data["correctness"]

fig = plt.figure(figsize=(10, 8))
plt.plot(excludable_feature, [baseline_corectness for _ in excludable_feature], color='red', marker='o', label='Baseline', linewidth=2)
plt.subplots_adjust(bottom=0.25)
plt.legend()
plt.ylim(0.75, 1)

plotter = Plotter()
plotter.plot(
    excludable_feature,
    correctness,
    f"Feature Importance with {confidence}% confidence, sample size {sample_size} with 9:1 split repeated {n-1} times,\n seed: {seed}, filtered generated and absolute features, resolution {constants.RESOLUTION}, euclidean",
    "Feature",
    "Accuracy",
    confidences,
    Path("../generated/feature_importance_absolutes_removed.svg").resolve(),
    "bar",
    45,
    fig
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