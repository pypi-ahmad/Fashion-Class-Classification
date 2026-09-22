"""
Fashion Neural Lab - Experiment Runner (train.py)

This script acts as the "Factory" for the Fashion Neural Lab.
It performs the following key tasks:
1.  **Data Setup**: Downloads and preprocesses FashionMNIST.
2.  **Model Instantiation**: Builds all seven registered architectures.
3.  **Training**: Fine-tunes all models for classification.
4.  **Embedding Extraction**: Runs the test set through the trained models to
    extract latent feature vectors (embeddings) for analysis.
5.  **Bundling**: Saves models, metrics, and embeddings into 'fashion_bundle.pth'.
"""

import os
import random

import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from torch.utils.data import DataLoader
from torchvision import datasets
from tqdm import tqdm

from model_registry import (
    MODEL_CONFIGS,
    MODEL_NAMES,
    build_model,
    build_transform,
    get_classifier_layer,
    serialize_model_config,
)

# --- Configuration ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EPOCHS = 5
SEED = 42


def set_seed(seed=SEED):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


# --- 1. Data Setup ---
def get_dataloaders(input_size=32, batch_size=None):
    """
    Downloads FashionMNIST and prepares DataLoaders.

    Returns:
        train_loader (DataLoader): For training.
        test_loader (DataLoader): For evaluation and embedding extraction.
        test_data (Dataset): Original dataset for access to raw images if needed.
    """
    if batch_size is None:
        batch_size = 64 if input_size == 32 else 16

    train_transform = build_transform(input_size, train=True)
    test_transform = build_transform(input_size, train=False)

    os.makedirs("./data", exist_ok=True)

    train_data = datasets.FashionMNIST(
        root="./data", train=True, download=True, transform=train_transform
    )
    test_data = datasets.FashionMNIST(
        root="./data", train=False, download=True, transform=test_transform
    )

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader, test_data


def get_models(*, pretrained=True):
    """
    Initializes the model zoo.

    Returns:
        models_dict (dict): Dictionary of {name: model_instance}.
    """
    return {
        name: build_model(name, pretrained=pretrained and MODEL_CONFIGS[name].pretrained)
        for name in MODEL_NAMES
    }


# --- 3. Training Loop ---
def train_model(
    model,
    train_loader,
    epochs=EPOCHS,
    learning_rate=1e-3,
    weight_decay=1e-4,
):
    """
    Standard PyTorch training loop.
    """
    model = model.to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), learning_rate, weight_decay=weight_decay)

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs}")
        for images, labels in pbar:
            try:
                images, labels = images.to(DEVICE), labels.to(DEVICE)
            except (TypeError, AttributeError) as exc:
                raise ValueError("Invalid batch data format encountered in DataLoader.") from exc

            if images.ndim != 4:
                raise ValueError(f"Expected image batch rank 4 [N,C,H,W], got rank {images.ndim}.")
            if labels.ndim != 1:
                raise ValueError(f"Expected label batch rank 1 [N], got rank {labels.ndim}.")
            if images.size(0) != labels.size(0):
                raise ValueError("Batch size mismatch between images and labels.")

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            pbar.set_postfix({"loss": running_loss / (pbar.n + 1)})

    return model


def evaluate_model(model, test_loader):
    """
    Evaluates model performance on the test set.
    """
    model = model.to(DEVICE)
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    acc = accuracy_score(all_labels, all_preds)
    p, r, f1, _ = precision_recall_fscore_support(
        all_labels, all_preds, average="weighted", zero_division=0
    )

    return {"Accuracy": acc, "Precision": p, "Recall": r, "F1": f1}


# --- 4. Search Index Generation (Embeddings) ---
def get_embeddings(model, loader, model_name):
    """
    Extracts latent representations (embeddings) for all images in the loader.
    Crucial for PCA and Latent Space Analysis.

    A pre-forward hook on the final linear classifier captures the exact
    feature vector used for classification across every architecture.
    """
    if model_name not in MODEL_NAMES:
        raise ValueError(f"Unsupported model for embedding extraction: {model_name}")

    model = model.to(DEVICE)
    model.eval()
    embeddings = []
    labels_list = []
    indices_list = []

    activation = {}

    def capture_classifier_input(_module, inputs):
        activation["embedding"] = inputs[0].detach()

    classifier = get_classifier_layer(model, model_name)
    hook_handle = classifier.register_forward_pre_hook(capture_classifier_input)

    print(f"Extracting embeddings for {model_name}...")
    with torch.no_grad():
        for batch_idx, (images, labels) in enumerate(tqdm(loader)):
            images = images.to(DEVICE)

            _ = model(images)
            emb = torch.flatten(activation["embedding"], 1)

            embeddings.append(emb.cpu())
            labels_list.append(labels)

            # Keep track of indices to map back to original dataset
            start_idx = batch_idx * loader.batch_size
            end_idx = start_idx + len(labels)
            indices_list.extend(list(range(start_idx, end_idx)))

    hook_handle.remove()

    return torch.cat(embeddings), torch.cat(labels_list), indices_list


# --- Main Execution ---
if __name__ == "__main__":
    print(f"Using device: {DEVICE}")
    set_seed()
    # The Bundle stores everything the App needs
    bundle = {
        "bundle_version": 2,
        "models": {},
        "metrics": {},
        "search_index": {},
        "model_config": {},
    }
    loaders = {}

    for name in MODEL_NAMES:
        config = MODEL_CONFIGS[name]
        loader_key = (config.input_size, config.batch_size)
        if loader_key not in loaders:
            loaders[loader_key] = get_dataloaders(*loader_key)
        train_loader, test_loader, _ = loaders[loader_key]
        model = build_model(name, pretrained=config.pretrained)
        print(f"\n--- Processing {name} ---")

        print("Training...")
        model = train_model(
            model,
            train_loader,
            epochs=config.epochs,
            learning_rate=config.learning_rate,
            weight_decay=config.weight_decay,
        )

        print("Evaluating...")
        metrics = evaluate_model(model, test_loader)
        print(f"Metrics: {metrics}")

        print("Generating Latent Embeddings...")
        vectors, labels, paths = get_embeddings(model, test_loader, name)

        bundle["models"][name] = {
            key: value.detach().cpu().clone() for key, value in model.state_dict().items()
        }
        bundle["metrics"][name] = metrics
        bundle["model_config"][name] = serialize_model_config(config)
        bundle["search_index"][name] = {
            "vectors": vectors,  # (N, Dim) Tensor
            "labels": labels,  # (N,) Tensor
            "paths": paths,  # List of indices
        }

        model.to("cpu")
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    torch.save(bundle, "fashion_bundle.pth")
    print("\n✅ Success! Saved 'fashion_bundle.pth'. You can now run 'streamlit run app.py'.")
