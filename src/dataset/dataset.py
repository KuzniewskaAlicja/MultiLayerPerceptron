import numpy as np


class Dataset:
    def __init__(
        self,
        data: np.ndarray,
        labels: np.ndarray,
    ):
        self.data = data
        self.labels = labels
    
    def split(self, val_factor: float = 0.2, shuffle: bool = True):
        samples_nb = self.data.shape[0]
        val_size = int(samples_nb) * val_factor
        indices = np.arange(samples_nb)

        if shuffle:
            np.random.shuffle(indices)
        
        val_indices = indices[:val_size]
        train_indices = indices[val_size:]

        self.train_subset = (
            self.data[train_indices], self.labels[train_indices]
        )
        self.val_subset = (self.data[val_indices], self.labels[val_indices])

    def get_batches(self, subset_name: str | None, batch_size: int):
        if subset_name == "train":
            data, labels = self.train_subset
        elif subset_name == "val":
            data, labels = self.val_subset
        elif not subset_name:
            data, labels = self.data, self.labels
        else:
            raise ValueError("Uknown subset name")

        batches = []
        for i in range(0, data.shape[0], batch_size):
            batch_data = data[i:i + batch_size]
            batch_labels = labels[i:i + batch_size]
            batches.append((batch_data, batch_labels))

        return batches
