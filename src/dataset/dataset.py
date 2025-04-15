import numpy as np


class Dataset:
    def __init__(
        self,
        data: np.ndarray,
        labels: np.ndarray,
        split_factor: float = 0.2,
        batch_size: int = 16
    ):
        ...
    
    def split(self):
        ...
    
    def get_batches(self, subset_name: str):
        ...