import numpy as np


class ReLU:
    def __call__(self, input: np.ndarray) -> np.ndarray:
        return np.maximum(0, input)
    
    def derivative(self, input: np.ndarray) -> np.ndarray:
        return np.where(input > 0, 1.0, 0.0)


class Softmax:
    def __call__(self, input: np.ndarray) -> np.ndarray:
        input -= np.max(input, axis=1, keepdims=True)
        exponential = np.exp(input)
        probs = exponential / (np.sum(exponential, axis=1, keepdims=True) + 1e-10)

        return probs
