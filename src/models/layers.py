import numpy as np
from collections.abc import Callable


class BaseLayer:
    def __init__(
        self,
        features_in_size: int,
        features_out_size: int,
        activation: Callable
    ):
        self.weights = np.random.randn(features_in_size, features_out_size) * 0.01
        self.biases = np.zeros((1, features_out_size))
        self.activation = activation
        self.input = None
        self.output = None
    
    def forward(self, input: np.ndarray):
        self.input = input
        self.output = input @ self.weights + self.biases
        self.output = self.activation(self.output)

        return self.output

    def update_params(self, learning_rate: float):
        if self.input is not None and self.delta is not None:
            self.weights -= learning_rate * self.input.T @ self.delta
            self.biases -= learning_rate * np.sum(self.delta, axis=0, keepdims=True)


class HiddenLayer(BaseLayer):
    def backward(self, next_delta: np.ndarray) -> np.ndarray:
        error = next_delta @ self.weights.T
        self.delta = error * self.activation.derivative(self.output)

        return self.delta


class OutputLayer(BaseLayer):
    ...
