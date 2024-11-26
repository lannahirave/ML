import click
import numpy as np
from sklearn.metrics import accuracy_score

import pandas as pd
from PIL import Image
import os
import matplotlib.pyplot as plt
import clip, torch
from sklearn.model_selection import train_test_split, KFold, StratifiedKFold
from LogisticRegression import MyLogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from logger import check_folder
import time


def load_data(image_folder: str, label_file: str, H: int = 224, W: int = 224):
    """Loads images and labels from the specified folder and file."""

    labels_df = pd.read_csv(label_file, delimiter="|")
    labels_df["label"] = labels_df["label"].map({"animal": 1, "human": 0})
    labels = labels_df["label"].tolist()

    images = []
    image_names = labels_df["image_name"].tolist()
    for image_name in image_names:
        image_file = (
            Image.open(os.path.join(image_folder, image_name))
            .convert("RGB")
            .resize((H, W))
        )
        image_array = np.array(image_file).transpose((2, 0, 1))
        images.append(image_array)

    images = np.array(images)

    comments = labels_df["comment"].tolist()
    return images, labels, image_names, comments


def vectorize_images(images: np.ndarray):
    """Vectorizes images into a matrix of size (N, D), where N is the number of images, and D is the dimensionality of the image."""

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load("ViT-B/32", device=device)
    images = torch.tensor(images, dtype=torch.float32).to(device) / 255.0

    with torch.no_grad():
        image_features = model.encode_image(images)

    image_features /= image_features.norm(dim=-1, keepdim=True)
    return image_features.cpu().numpy()


def validation_split(X: np.ndarray, y: np.ndarray, test_size: float):
    """Splits data into train and test."""
    return train_test_split(X, y, test_size=test_size, random_state=25112024)


def create_model(model_name: str):
    """Creates a model of the specified name.
    1. Use your LinearRegression implementation,
    2. KNN
    3. Decision Tree
    Args:
        model_name (str): Name of the model to use.
    Returns:
        model (object): Model of the specified name.
    """
    if model_name == "logistic_regression":
        model = MyLogisticRegression()
    elif model_name == "knn":
        model = KNeighborsClassifier()
    elif model_name == "decision_tree":
        model = DecisionTreeClassifier()
    else:
        raise ValueError(
            "Invalid model name. Choose from ['logistic_regression', 'knn', 'decision_tree']"
        )
    return model


def simple_train_test_validation(X_train, X_test, y_train, y_test, model):
    """Trains and evaluates the model on the test set."""
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    return accuracy


def k_fold_validation(model, X_train, y_train, k=5, use_stratified=False):
    """Performs K-fold or Stratified K-fold cross-validation."""
    accuracies = []
    if use_stratified:
        kf = StratifiedKFold(n_splits=k)
    else:
        kf = KFold(n_splits=k)

    accuracies = []
    for train_idx, val_idx in kf.split(X_train, y_train):
        X_tr, X_val = X_train[train_idx], X_train[val_idx]
        y_tr, y_val = y_train[train_idx], y_train[val_idx]

        model.fit(X_tr, y_tr)
        y_pred = model.predict(X_val)
        accuracies.append(accuracy_score(y_val, y_pred))

    avg_accuracy = np.mean(accuracies)

    return avg_accuracy


def plot_images(
    images, image_names, y_pred, y_test, folder_plots, model_name, method_name
):
    plt.title("First 10 test images " + method_name)
    fig, axs = plt.subplots(4, 5, figsize=(20, 25))
    for i in range(20):
        # Transpose the image to (H, W, 3) for displaying
        image_to_plot = images[i].transpose(1, 2, 0)
        axs[i // 5, i % 5].imshow(image_to_plot)
        pred_label = "animal" if y_pred[i] == 1 else "human"
        test_label = "animal" if y_test[i] == 1 else "human"
        axs[i // 5, i % 5].set_title(
            f"Prediction: {pred_label} \n Image: {image_names[i]} \n True label: {test_label}"
        )
        axs[i // 5, i % 5].axis("off")
    plt.savefig(folder_plots + method_name + "_first_10_test_images.png")

    # Confusion matrix
    plt.figure(figsize=(8, 8))
    cm = confusion_matrix(y_test, y_pred)
    plt.imshow(cm, cmap="Blues")
    plt.title("Confusion matrix" + method_name)
    plt.colorbar()
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.savefig(folder_plots + method_name + "_confusion_matrix.png")


@click.command()
@click.option("--image_folder", type=str, help="Path to the folder containing images")
@click.option("--label_file", type=str, help="Path to the file containing labels")
@click.option("--model_name", type=str, help="Name of the model to use")
@click.option("--test_size", type=float, default=0.2, help="Size of the test split")
def main(image_folder: str, label_file: str, model_name: str, test_size: float):

    image_folder = "dataset/images/"
    label_file = "dataset/labels.csv"

    test_size = 0.2
    images, labels, image_names, images_comments = load_data(image_folder, label_file)
    X = vectorize_images(images)
    y = labels
    X_train, X_test, y_train, y_test = validation_split(X, y, test_size)

    model_names = ["logistic_regression", "knn", "decision_tree"]
    for model_name in model_names:
        folder_plots = "lab1/part2/img/" + model_name + "/"
        check_folder(folder_plots)
        timer = time.time()
        print(f"Model: {model_name}")
        # simple train/test validation
        model = create_model(model_name)
        simple_train_test_validation(X_train, X_test, y_train, y_test, model)
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        print(f"Simple train/test accuracy: {accuracy:.2f}")
        plot_images(
            images,
            image_names,
            y_pred,
            y_test,
            folder_plots,
            model_name,
            "simple_train_test",
        )

        # K-Fold validation
        model = create_model(model_name)
        k = 5
        stratified = False
        accuracy = k_fold_validation(model, X_train, y_train, k, stratified)
        print(f"K-Fold Validation (k={k}) Accuracy: {accuracy:.2f}")
        y_pred = model.predict(X_test)
        plot_images(
            images, image_names, y_pred, y_test, folder_plots, model_name, "k_fold"
        )

        model = create_model(model_name)
        stratified = True
        accuracy = k_fold_validation(model, X_train, y_train, k, stratified)
        print(f"Stratified K-Fold Validation (k={k}) Accuracy: {accuracy:.2f}")
        y_pred = model.predict(X_test)
        plot_images(
            images,
            image_names,
            y_pred,
            y_test,
            folder_plots,
            model_name,
            "stratified_k_fold",
        )
        print(f"Time: {time.time() - timer:.2f}s")


if __name__ == "__main__":
    main()
