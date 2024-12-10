
from sklearn.neighbors import KNeighborsClassifier
import numpy as np

class SklearnLearner:
    def __init__(self, n_neighbors= 3):
        self.classifier = KNeighborsClassifier(n_neighbors=n_neighbors)
        self.is_trained = False

    def learn(self, X: np.ndarray, y: np.ndarray):
        self.classifier.fit(X,y)
        self.is_trained = True

    def classify(self, X: np.ndarray) -> np.ndarray:
        if not self.is_trained:
            raise ValueError("The model has not been trained yet")
        return self.classifier.predict(X)

