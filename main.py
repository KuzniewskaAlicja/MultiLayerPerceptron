import numpy as np
import argparse

from src.models import MultiLayerPerceptron
from src.utils import CategoricalCrossEntropy
from src.dataset import Dataset


def generate_data(
    n_samples: int,
    n_features: int,
    classes_nb: int,
    seed: int | None = None
):
    if seed is not None:
        np.random.seed(seed)
    
    samples_per_class = n_samples // classes_nb
    data = []
    labels = []

    for class_id in range(classes_nb):
        center = np.random.uniform(-5, 5, size=(n_features,))
        noise = np.random.normal(0, 1, size=(samples_per_class, n_features))
        class_samples = center + noise
        class_labels = np.full((samples_per_class,), class_id)

        data.append(class_samples)
        labels.append(class_labels)

    print(data[0].shape, labels[0].shape)

    data = np.vstack(data)
    labels = np.concatenate(labels)

    onehot_labels = np.zeros((labels.size, classes_nb))
    onehot_labels[np.arange(labels.size), labels] = 1

    return data, labels


def train(model: MultiLayerPerceptron, dataset: Dataset):
    learning_rate = 0.001
    cat_crossentropy = CategoricalCrossEntropy()
    epochs = 100
    batch_size = {"val": 64, "train": 32}

    dataset.split(val_factor=0.15, shuffle=True)

    best_loss = float("inf")
    for epoch in range(epochs):
        print(20 * "-" + f"Epoch: {epoch}/{epochs}" + 20 * "-")
        for learning_step in ["train", "val"]:
            loss = 0.0
            batch_data = dataset.get_batches(
                learning_step, batch_size[learning_step]
            )
            for data, labels in batch_data:
                predictions = model.predict(data)
                loss += cat_crossentropy(labels, predictions)
                if learning_step == "train":
                    model.backprogation(labels, cat_crossentropy, learning_rate)
            
            print(f"{learning_step.upper()}_loss: {loss / len(batch_data)}")

        # model checkpoint based on validation loss


def eval():
    ...


if __name__ == "__main__":
    classes_nb = 3
    n_features = 10
    data, labels = generate_data(
        n_samples=500,
        n_features=n_features,
        classes_nb=classes_nb
    )
    dataset = Dataset(data, labels)
    model = MultiLayerPerceptron(input_size=n_features, classes_nb=classes_nb)
    print("Model architecture: ", model)

    train(model, dataset)