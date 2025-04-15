import numpy as np


class BinaryCrossEntropy:
    def __init__(self, epsilon: float = 1e-15):
        self.min = epsilon
        self.max = 1 - epsilon

    def __call__(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        y_pred = np.clip(y_pred, self.min, self.max)
        loss = y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred)
        loss = -np.mean(loss)

        return loss

    def derivative(self, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        y_pred = np.clip(y_pred, self.min, self.max)
        df = (y_pred - y_true) / (y_pred * (1 - y_pred)) / y_true.shape[0]

        return df


class CategoricalCrossEntropy:
    def __init__(self, epsilon: float = 1e-15):
        self.min = epsilon
        self.max = 1 - epsilon
    
    def __call__(self, y_pred: np.ndarray, y_true: np.ndarray) -> float:
        y_pred = np.clip(y_pred, self.min, self.max)
        loss = -np.sum(y_true * np.log(y_pred), axis=-1)

        return loss

    def derivative(self, y_pred: np.ndarray, y_true: np.ndarray):
        y_pred = np.clip(y_pred, self.min, self.max)
        df = -y_true / y_pred

        return df
