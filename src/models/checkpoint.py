import numpy as np
import os

from .mlp import MultiLayerPerceptron


class ModelCheckpoint:
    def __init__(
        self,
        model: MultiLayerPerceptron,
        metric: str,
        mode: str,
        patience: int
    ):
        self.model = model
        self.metric = metric
        self.mode = mode
        self.patience = patience
        self.best_weights = None
        self.best_value = float('inf') if mode == 'min' else float('-inf')
        self.wait = 0
    
    def check_improvement(self, current_value: float) -> bool:
        if (
            self.mode == "min" and current_value < self.best_value or
            self.mode == "max" and current_value > self.best_value
        ):
            self.best_value = current_value
            self.best_weights = self.model.get_weights()
            self.wait = 0
        else:
            self.wait += 1
            if self.patience is not None and self.wait >= self.patience:
                return True
        
        return False

    def save_model(self, file_path: str):
        if not os.path.exists(os.path.dirname(file_path)):
            os.makedirs(os.path.dirname(file_path))

        if self.best_weights is None:
            self.best_weights = self.model.get_weights()

        np.savez(file_path, **self.best_weights)
