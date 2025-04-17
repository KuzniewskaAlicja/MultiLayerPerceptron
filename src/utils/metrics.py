import numpy as np


class Accuracy:
    def __call__(self, predictions: np.ndarray, targets: np.ndarray) -> float:
        pred_classes = np.argmax(predictions, axis=1)
        true_classes = np.argmax(targets, axis=1)
        correct_predictions = np.sum(pred_classes == true_classes)

        return correct_predictions / targets.shape[0]