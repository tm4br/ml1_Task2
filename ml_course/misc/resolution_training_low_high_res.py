import random
from pathlib import Path

from enums import Confidence
from EuclideanLearner import EuclideanLearner
from evaluator import Evaluator
from helpers import Helpers
from plotter import Plotter

seed = random.randint(0, 2 ** 32 - 1)
sample_size = 1500
n = 40
excluded_features = ["_red_pixels", "_yellow_pixels", "_blue_pixels", "_black_pixels"]

hres_evaluator = Evaluator(EuclideanLearner(k=1, exclude=excluded_features))
lres_evaluator = Evaluator(EuclideanLearner(k=1, exclude=excluded_features))

high_res = Helpers.read_json_file("../generated/jsons/test.3500.fixed_extraction.bak.json")
low_res = Helpers.read_json_file("../generated/jsons/test.80x60.fixed_extraction.bak.json")

for i in range(n):
    print(f"at iter {i}")
    hres_trainings_set, hres_test_set = Evaluator.split_sets(high_res, seed, (9, 1), sample_size)
    lres_trainings_set, lres_test_set = Evaluator.split_sets(low_res, seed, (9, 1), sample_size)

    hres_evaluator.evaluate(hres_trainings_set, hres_test_set)
    lres_evaluator.evaluate(lres_trainings_set, lres_test_set)

hres_correctness = hres_evaluator.median_correctness()
hres_confidences = hres_evaluator.sucess_confidence(Confidence.C99)
lres_correctness = lres_evaluator.median_correctness()
lres_confidences = lres_evaluator.sucess_confidence(Confidence.C99)

plotter = Plotter()
plotter.plot(
    ["80x60", "3750x3750"],
    [lres_correctness, hres_correctness],
    f"High vs Low res sample size {sample_size} with 9:1 split repeated {n-1} times, seed: {seed}",
    "Resolution, downscaled",
    "Accuracy",
    [lres_confidences, hres_confidences],
    Path("../generated/high_vs_low_res.svg").resolve(),
    "bar"
)
