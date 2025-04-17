import numpy as np


class CategoricalCrossEntropy:
    def __init__(self, epsilon: float = 1e-15):
        self.min = epsilon
        self.max = 1 - epsilon

    def __call__(self, y_pred: np.ndarray, y_true: np.ndarray) -> float:
        y_pred = np.clip(y_pred, self.min, self.max)
        loss = -np.sum(y_true * np.log(y_pred), axis=-1)
        loss = np.mean(loss)

        return loss
