import numpy as np


class Sigmoid:
    def __call__(self, input: np.ndarray) -> np.ndarray:
        return 1 / (1 + np.exp(-input))
    
    def derivative(self, input: np.ndarray) -> np.ndarray:
        s = self(input)
        df = s * (1 - s)

        return df


class ReLU:
    def __call__(self, input: np.ndarray) -> np.ndarray:
        return np.maximum(0, input)
    
    def derivative(self, input: np.ndarray) -> np.ndarray:
        return (input > 0).astype(float)


class Softmax:
    def __call__(self, input: np.ndarray) -> np.ndarray:
        input -= np.max(input, axis=1, keepdims=True)
        exponential = np.exp(input)
        probs = exponential / np.sum(exponential, axis=1, keepdims=True)

        return probs
