import numpy as np
import argparse

from src.models import MultiLayerPerceptron, ModelCheckpoint
from src.utils import CategoricalCrossEntropy, Accuracy
from src.dataset import Dataset


parser = argparse.ArgumentParser()
parser.add_argument(
    "--early_stopping",
    type=int,
    default=None,
    help="Number of epochs to wait until stop training for no progress"
)
parser.add_argument(
    "--name",
    type=str,
    default="model",
    help="Model name"
)
parser.add_argument(
    "--model_weights",
    type=str,
    default=None,
    help="Path to model weights npz file"
)
args = parser.parse_args()


def generate_data(
    n_samples: int,
    n_features: int,
    classes_nb: int,
    seed: int = 0
):
    if seed:
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

    data = np.vstack(data)
    labels = np.concatenate(labels)

    onehot_labels = np.zeros((labels.size, classes_nb))
    onehot_labels[np.arange(labels.size), labels] = 1

    return data, onehot_labels


def train(model: MultiLayerPerceptron, dataset: Dataset):
    learning_rate = 0.001
    cat_crossentropy = CategoricalCrossEntropy()
    epochs = 20
    metric = Accuracy()
    batch_size = {"val": 64, "train": 32}
    checkpoint = ModelCheckpoint(
        model, "accuracy", "max", args.early_stopping
    )

    dataset.split(val_factor=0.15, shuffle=True)
    dataset.normalize()

    for epoch in range(epochs):
        print(20 * "-" + f"Epoch: {epoch}/{epochs}" + 20 * "-")
        for learning_step in ["train", "val"]:
            loss, acc = 0.0, 0.0
            batch_data = dataset.get_batches(
                learning_step, batch_size[learning_step]
            )
            for data, labels in batch_data:
                predictions = model.predict(data)
                loss += cat_crossentropy(predictions, labels)
                acc += metric(predictions, labels)
                if learning_step == "train":
                    model.backprogation(labels, learning_rate)

            loss /= len(batch_data)
            acc /= len(batch_data)
            print(f"{learning_step.upper()} - loss: {loss} acc: {acc}")

            if learning_step == "val":
                should_stop = checkpoint.check_improvement(acc)
                break
        if should_stop:
            print(f"Finished training on {epoch} epoch")
            break
    checkpoint.save_model(f"./models/{args.name}.npz")


if __name__ == "__main__":
    classes_nb = 3
    n_features = 10
    data, labels = generate_data(
        n_samples=500,
        n_features=n_features,
        classes_nb=classes_nb,
        seed=42
    )
    dataset = Dataset(data, labels)
    model = MultiLayerPerceptron(input_size=n_features, classes_nb=classes_nb)
    print("Model architecture: ", model)
    if args.model_weights:
        model.load_weights(args.model_weights)
    train(model, dataset)