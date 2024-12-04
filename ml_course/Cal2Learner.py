import numbers
import random
import time
from typing import List, Dict

from enums import Concept
from evaluator import Evaluator
from feature_vector import FeatureVector
from helpers import Helpers
from learner import LearnerWrapper, ILearner
from normalizer import INormalizer, DownsamplingNormalizer


class Tree:

    def __init__(self):
        self.children: Dict[numbers.Number, Tree | None] = {}
        self.parent: Tree | None = None
        self.concept: Concept | None = None
        self.depth: int = 0

    def add_child(self, key: numbers.Number, concept: Concept):
        node = Tree()
        node.depth = self.depth +1
        node.parent = self
        node.concept = concept
        self.children[key] = node

    def access_child(self, key: numbers.Number):
        try:
            return self.children[key]
        except:
            return None

    def is_leaf(self):
        return len(self.children) == 0


class Cal2Learner(ILearner):

    def __init__(self,  exclude, normalizer: INormalizer = None):
        self.excluded = exclude
        self.normalizer = normalizer
        self.tree = Tree()
        self.var_check_order: List[str] = []

    def learn(self, vectors: List[FeatureVector] | List[Dict]):

        # todo somewhat dirty, maybe consolidate signature?
        if len(vectors) > 0 and isinstance(vectors[0], FeatureVector):
            converted_vecs = []
            for vec in vectors:
                converted_vecs.append(vec.flatten())
            vectors = converted_vecs

        #dirty_count = len(vectors)
        epochs = 4
        current_vector = 0

        # we need to do this because our vectors are not really vectors, but dicts
        self.var_check_order = self.determine_stable_order(vectors[0])

        while epochs >= 0:

            vector = vectors[current_vector]
            if current_vector == 0:
                epochs -= 1

            # always returns a leaf, root in base-case
            okay, node = self._classify(vector, self.tree)
            if not okay:  # no transition found
                node.add_child(self.feature_access_helper(vector, node.depth), vector["concept"]) # access like this is workaround until i get my typing fixed. We currently receive dict insteaf of obj!
                #dirty_count += 1
            elif node.concept == vector["concept"]:
                current_vector = ((current_vector + 1) % (len(vectors)))
                #dirty_count -= 1
            else:
                node.add_child(self.feature_access_helper(vector, node.depth), vector["concept"])
                current_vector = ((current_vector + 1) % (len(vectors)))
                #dirty_count += 1

    def feature_access_helper(self, vector: FeatureVector | Dict, num: int):
        return vector[self.var_check_order[num]]

    def determine_stable_order(self, vector):
        res = []
        for k, v in vector.items():
            if not k.startswith("__") and not k in self.excluded:
                res.append(k)
        return sorted(res)

    def classify(self, vector: FeatureVector | Dict) -> Concept:

        if isinstance(vector, FeatureVector):
            vector = vector.flatten()

        okay, node = self._classify(vector, self.tree)
        if not okay:
            return Concept.NONE
        return node.concept

    def _classify(self, vector: Dict, node=None) -> (bool, Tree):
        """
        Yields True and leaf node if all transitions were found
        Yields False and last accessible node if a transition is not found
        :param vector: vector to classify
        """

        new_node = node.access_child(self.feature_access_helper(vector, node.depth))

        if new_node is None:
            return False, node

        if new_node.is_leaf():
            return True, new_node

        return self._classify(vector, new_node)


if __name__ == "__main__":
    vectors = Helpers.read_json_file("generated/jsons/test.16.fixed_extraction.bak.json") # needs to be retireved via extraction first

    with open("dump.txt", "a") as f:
        for i in range(100):
            learner = LearnerWrapper(Cal2Learner(exclude=["_red_pixels", "_yellow_pixels", "_blue_pixels", "_black_pixels", "clusters", "concept"]), normalizer=DownsamplingNormalizer())
            print("#################")
            f.write("#################\r\n")
            seed = time.time_ns()
            random.seed(seed)
            print(f"seed: {seed}")
            f.write(f"seed: {seed}\n")

            trainings_set, test_set = Evaluator.split_sets(vectors, None, (9, 1), len(vectors))
            evaluator = Evaluator(learner)
            evaluator.evaluate(trainings_set, test_set)

            f.write(f"correctness: {evaluator.median_correctness()}")
            f.write(f"confidences: {evaluator.sucess_confidence()}")
            evaluator.print_stats()