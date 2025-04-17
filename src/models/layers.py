import numpy as np
from collections.abc import Callable
from abc import ABC, abstractmethod
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


class BaseLayer(ABC):
    def __init__(
        self,
        features_in_size: int,
        features_out_size: int,
        activation: Callable
    ):
        self.features_in_size = features_in_size
        self.features_out_size = features_out_size
        self.weights = None
        self.biases = np.zeros((self.features_out_size,))
        self.activation = activation
        self.input = None
        self.output = None
    
    def forward(self, input: np.ndarray):
        self.input = input
        self.output = input @ self.weights + self.biases
        self.output = self.activation(self.output)

        return self.output

    def update_params(self, learning_rate: float, momentum: float = 0.9):
        if self.input is not None and self.delta is not None:
            self.weights -= learning_rate * self.input.T @ self.delta
            self.biases -= learning_rate * np.sum(self.delta, axis=0)
    
    def clip_gradient(self, clip_value: float = 5.0):
        self.delta = np.clip(self.delta, -clip_value, clip_value)

    def __repr__(self):
        return (
            "{}"
            f"({self.features_in_size} -> {self.features_out_size}) "
            f"activation={self.activation.__class__.__name__}"
        )
    
    @abstractmethod
    def backward(*args):
        ...


class HiddenLayer(BaseLayer):
    instances_nb = 0 
    def __init__(
        self,
        features_in_size: int,
        features_out_size: int,
        activation: Callable
    ):
        super().__init__(features_in_size, features_out_size, activation)
        # He initialization
        self.weights = (
            np.random.randn(self.features_in_size, self.features_out_size)
            * np.sqrt(2 / self.features_in_size)
        )
        self.name = f"{self.__class__.__name__}_{HiddenLayer.instances_nb}"
        HiddenLayer.instances_nb += 1

    def backward(self, next_delta: np.ndarray, next_weights: np.ndarray) -> np.ndarray:
        error = next_delta @ next_weights.T
        self.delta = error * self.activation.derivative(self.output)
        self.clip_gradient(clip_value=10.0)

        return self.delta

    def __repr__(self):
        return super().__repr__().format(self.__class__.__name__)


class OutputLayer(BaseLayer):
    def __init__(
        self,
        features_in_size: int,
        features_out_size: int,
        activation: Callable
    ):
        super().__init__(features_in_size, features_out_size, activation)
        # Xavier/Glorot initialization
        self.weights = (
            np.random.randn(self.features_in_size, self.features_out_size)
            * np.sqrt(1 / self.features_in_size)
        )

    def backward(self, y: np.ndarray) -> np.ndarray:
        self.delta = self.output - y
        self.clip_gradient(clip_value=1.0)
        return self.delta

    def __repr__(self):
        return super().__repr__().format("OutputLayer")
