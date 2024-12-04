from pathlib import Path
import random

import numpy as np
from matplotlib import pyplot as plt

from enums import Confidence
from EuclideanLearner import EuclideanLearner
from evaluator import Evaluator
from helpers import Helpers
from learner import LearnerWrapper
from plotter import Plotter
import constants

seed = random.randint(0, 2**32 - 1)
sample_size = 2000
n = 10
confidence = Confidence.C99
excluded_features = ["_red_pixels", "_yellow_pixels", "_blue_pixels", "_black_pixels"]
old_ds_e = Evaluator(LearnerWrapper(EuclideanLearner(k=1, exclude=excluded_features)))
new_ds_e = Evaluator(LearnerWrapper(EuclideanLearner(k=1, exclude=excluded_features)))

old_ds = Helpers.read_json_file("../generated/jsons/test.16.fixed_extraction.bak.json")  # needs to be retireved via extraction first
new_ds = Helpers.read_json_file("../generated/jsons/test.16.fixed_extraction.all.bak.json")

for i in range(1, n+1):
    print(f"at iter: {i}")
    if seed is None:
        s = None
    else:
        s = seed * i
    training_old, test_old = Evaluator.split_sets(old_ds, s, (9, 1), sample_size)
    training_new, test_new = Evaluator.split_sets(new_ds, s, (9, 1), sample_size)

    old_ds_e.evaluate(training_old, test_old)
    new_ds_e.evaluate(training_new, test_new)

correctness_old = old_ds_e.median_correctness()
confidences_old = old_ds_e.sucess_confidence(confidence)

correctness_new = new_ds_e.median_correctness()
confidences_new = new_ds_e.sucess_confidence(confidence)

errors_old = Plotter.yerrs_from_confidences([confidences_old])
errors_new = Plotter.yerrs_from_confidences([confidences_new])


plt.bar([1], correctness_old, yerr=errors_old, capsize=5, color='skyblue', edgecolor='black')
plt.bar([2], correctness_new, yerr=errors_new, capsize=5, color='red', edgecolor='black')

plt.title(f"Dataset Comparison with {confidence}% confidence, sample size {sample_size} with 1:1 split repeated {n-1} times,\n seed: {seed}, filtered features, resolution {constants.RESOLUTION}, euclidean")
plt.xlabel("Datasets")
plt.ylabel("Accuracy")
plt.xticks([1, 2], ['Old Dataset', 'New Dataset'], rotation=45)

# plt.ylim(0, 1)

plt.savefig(Path("../generated/dataset_comparisons.svg").resolve(), format='svg', bbox_inches='tight')
plt.show()
