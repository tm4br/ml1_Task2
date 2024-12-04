import pickle

from enums import Confidence
from EuclideanLearner import EuclideanLearner
from evaluator import Evaluator
from helpers import Helpers
from learner import LearnerWrapper


seed = 7581902571
sample_size = 1000
n = 10
confidence = Confidence.C99

vectors = Helpers.read_json_file("../generated/jsons/test.16.fixed_extraction.bak.json")  # needs to be retireved via extraction first
dummy = LearnerWrapper(EuclideanLearner([], k=1))
dummy.learn([vectors[0]]) # just need one for this
exclude_clusters = [f for f in dummy.data[0].keys() if f.startswith("clusters")]
evaluator = Evaluator(LearnerWrapper(EuclideanLearner(exclude_clusters, k=1)))

for i in range(1, n+1):
    print(f"at iter: {i}")
    if seed is None:
        s = None
    else:
        s = seed * i

    trainings_set, test_set = Evaluator.split_sets(vectors, s, (9, 1), sample_size)
    evaluator.evaluate(trainings_set, test_set)


correctness = []
confidences = []
#evaluator.sucess_confidence(confidence)
correctness.append(evaluator.median_correctness())
confidences.append(evaluator.sucess_confidence(confidence))

with open("../generated/clusterless_baseline.pkl", "wb") as file:
    pickle.dump({"correctness": correctness, "confidences": confidences}, file)