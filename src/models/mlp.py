import numpy as np

from src.utils import ReLU, Softmax
from .layers import HiddenLayer, OutputLayer, InputLayer


class MultiLayerPerceptron:
    def __init__(self, input_size: int, classes_nb: int):
        self.input_layer = InputLayer(input_size)
        self.layer1 = HiddenLayer(self.input_layer.input_size, 16, ReLU())
        self.layer2 = HiddenLayer(16, 8, ReLU())
        self.out_layer = OutputLayer(8, classes_nb, Softmax())
    
    def predict(self, input: np.ndarray) -> np.ndarray:
        self.input = input
        output = self.layer1.forward(self.input)
        output = self.layer2.forward(output)
        output = self.out_layer(output)

        return output

    def backprogation(self, y: np.ndarray, loss: object, learning_rate: float):
        # layers backpropagation
        self.out_layer.delta = loss.derivative(y, self.out_layer.output)
        h2_delta = self.layer2.backward(self.out_layer.delta)
        h1_delta = self.layer1.backward(h2_delta)

        # layers parameters update
        self.layer1.update_params(learning_rate)
        self.layer2.update_params(learning_rate)
        self.out_layer.update_params(learning_rate)
    
    def __repr__(self):
        arch = f"{self.__class__.__name__}(\n"
        layers = [
            self.input_layer, self.layer1, self.layer2, self.output_layer
        ]
        for i, layer in layers:
            arch += f"\t[{i}] {repr(layer)}"
        arch += ")"

        return arch

    def save_model(self):
        ...
    
    def load_model(self):
        ...
