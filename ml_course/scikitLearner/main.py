import numpy as np

from ml_course.helpers import Helpers
from ml_course.preprocessor import Preprocessor
from ml_course.scikitLearner.ScikitLearner import SklearnLearner

if __name__ == "__main__":
    #Step 1: Read the data
    images = Helpers.read_images("path")
    labels = Helpers.read_labels("path")

    #Step 2: Preprocess the images
    preprocessor = Preprocessor()
    X = np.array([preprocessor.preprocess(image) for image in images])
    y = np.array(labels)

    #Step 3: Initialize the learner
    learner = SklearnLearner(n_neighbors=3)

    #Step 4: Train the learner
    learner.learn(X, y)

    #Step 5: Classify the images
    new_images = Helpers.read_images("path")
    X_new = np.array([preprocessor.preprocess(image) for image in new_images])
    predictions = learner.classify(X_new)

    print(predictions)