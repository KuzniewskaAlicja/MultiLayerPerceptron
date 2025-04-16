import numpy as np
from collections.abc import Callable
from dataclasses import dataclass


@dataclass
class InputLayer:
    input_size: int

    def forward(self, input: np.ndarray):
        return input
    
    def __repr__(self):
        return (
            f"InputLayer({self.input_size})"
        )


class BaseLayer:
    def __init__(
        self,
        features_in_size: int,
        features_out_size: int,
        activation: Callable
    ):
        self.features_in_size = features_in_size
        self.features_out_size = features_out_size
        self.weights = np.random.randn(self.features_in_size, self.features_out_size) * 0.01
        self.biases = np.zeros((self.features_out_size,))
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
    
    def __repr__(self):
        return (
            "{}"
            f"({self.features_in_size} -> {self.features_out_size})"
            f"activation={self.activation.__class__.__name__}"
        )


class HiddenLayer(BaseLayer):
    def backward(self, next_delta: np.ndarray) -> np.ndarray:
        error = next_delta @ self.weights.T
        self.delta = error * self.activation.derivative(self.output)

        return self.delta
    
    def __repr__(self):
        return super().__repr__().format("HiddenLayer")


class OutputLayer(BaseLayer):
    def __repr__(self):
        return super().__repr__().format("OutputLayer")
