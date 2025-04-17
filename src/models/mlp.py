import numpy as np

from src.utils import ReLU, Softmax
from .layers import HiddenLayer, OutputLayer, InputLayer


class MultiLayerPerceptron:
    def __init__(self, input_size: int, classes_nb: int):
        self.layers = [
            InputLayer(input_size),
            HiddenLayer(input_size, 16, ReLU()),
            HiddenLayer(16, 8, ReLU()),
            OutputLayer(8, classes_nb, Softmax())
        ]
    
    def predict(self, input: np.ndarray) -> np.ndarray:
        output = input
        for layer in self.layers:
            output = layer.forward(output) 

        return output

    def backprogation(self, y: np.ndarray, learning_rate: float):
        # layers backpropagation
        next_delta = self.layers[-1].backward(y)
        next_weights = self.layers[-1].weights
        for layer in reversed(self.layers[1:-1]):
            next_delta = layer.backward(next_delta, next_weights)
            next_weights = layer.weights

        # layers parameters update
        for layer in self.layers[1:]:
            layer.update_params(learning_rate)
    
    def __repr__(self):
        arch = f"{self.__class__.__name__}(\n"
        for i, layer in enumerate(self.layers):
            arch += f"\t[{i}] {repr(layer)}\n"
        arch += ")"

        return arch

    def get_weights(self):
        weights = {}
        for i, layer in enumerate(self.layers):
            if hasattr(layer, "name"):
                key = layer.name
            else:
                key = layer.__class__.__name__
            if hasattr(layer, "weights") and hasattr(layer, "biases"):
                weights[key] = {
                    "weights": layer.weights,
                    "biases": layer.biases
                }

        return weights

    def load_weights(self, file_path: str):
        weights = np.load(file_path, allow_pickle=True)
        self.layer_names = [
            layer.name if hasattr(layer, "name") else None
            for layer in self.layers
        ]
        for layer_name, attrs in weights.items():
            try:
                layer_idx = self.layer_names.index(layer_name)
                attrs = attrs.item()
            except ValueError:
                continue
            self.layers[layer_idx].weights = attrs["weights"].copy()
            self.layers[layer_idx].biases = attrs["biases"].copy()
