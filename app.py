"""
Fashion Neural Lab - Streamlit Dashboard (app.py)

This is the interactive dashboard for the "Comparative AI Laboratory".
It visualizes and compares the trained models from `train.py`.

Key Sections:
1.  **Diagnosis & Consensus**: Multi-model voting on inputs.
2.  **Explainability**: Grad-CAM visualization.
3.  **Performance**: Radar charts and Confusion Matrices.
4.  **Latent Space**: PCA visualization of embeddings.
"""

import os

import cv2
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import torch
from PIL import Image, UnidentifiedImageError
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix
from torchvision import datasets

from model_registry import (
    MODEL_CONFIGS,
    build_model,
    build_transform,
    get_classifier_layer,
    get_gradcam_layer,
)

# --- Configuration & Setup ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BUNDLE_PATH = "fashion_bundle.pth"
BUNDLE_LOAD_ERROR = None

st.set_page_config(page_title="Fashion Neural Lab", layout="wide", page_icon="🔬")


CLASSES = [
    "T-shirt/top",
    "Trouser",
    "Pullover",
    "Dress",
    "Coat",
    "Sandal",
    "Shirt",
    "Sneaker",
    "Bag",
    "Ankle boot",
]

# --- Helper Functions ---


@st.cache_resource
def load_data():
    """
    Load the FashionMNIST test dataset (images only).
    Used to pull random samples for testing.
    """
    # Ensure data directory exists
    os.makedirs("./data", exist_ok=True)
    test_data = datasets.FashionMNIST(root="./data", train=False, download=True)
    return test_data


@st.cache_resource
def load_bundle():
    """
    Load the trained models and metadata from disk.
    Returns None if file is missing.
    """
    global BUNDLE_LOAD_ERROR
    BUNDLE_LOAD_ERROR = None

    if not os.path.exists(BUNDLE_PATH):
        BUNDLE_LOAD_ERROR = f"File not found: {BUNDLE_PATH}"
        return None
    try:
        return torch.load(BUNDLE_PATH, map_location=DEVICE, weights_only=True)
    except Exception as exc:
        BUNDLE_LOAD_ERROR = str(exc)
        return None


def get_model_architecture(model_name):
    """
    Reconstruct the model architecture object based on name.
    Does NOT load weights.
    """
    return build_model(model_name, pretrained=False)


@st.cache_resource
def load_active_models(selected_models, _bundle_models):
    """
    Load dictionary of active models with weights loaded.

    Args:
        selected_models (list): List of model names.
        _bundle_models (dict): State dicts from the bundle.
                               Underscore prefix prevents hashing large dict.
    """
    loaded_models = {}
    for name in selected_models:
        if name not in _bundle_models:
            raise KeyError(f"Model '{name}' not found in bundle.")
        state_dict = _bundle_models[name]
        model = get_model_architecture(name)
        model.load_state_dict(state_dict)
        model.to(DEVICE)
        model.eval()
        loaded_models[name] = model
    return loaded_models


def preprocess_image(image, model_name, model_config=None):
    """
    Preprocess a PIL image for the model:
    Apply the resolution recorded in the bundle, or the registry default.
    """
    registry_size = MODEL_CONFIGS[model_name].input_size
    config = (model_config or {}).get(model_name, {})
    input_size = int(config.get("input_size", registry_size))
    return build_transform(input_size, train=False)(image).unsqueeze(0).to(DEVICE)


def get_gradcam(model, model_name, input_tensor, target_class_idx):
    """
    Generate Grad-CAM heatmap for explainability.

    Args:
        target_layers: The last convolutional layer to compute gradients against.
    """
    target_layers = [get_gradcam_layer(model, model_name)]

    cam = GradCAM(model=model, target_layers=target_layers)
    targets = [ClassifierOutputTarget(target_class_idx)]

    grayscale_cam = cam(input_tensor=input_tensor, targets=targets)  # ty: ignore[invalid-argument-type]
    return grayscale_cam[0, :]


# --- Main App ---

st.title("🔬 Fashion Neural Lab: Architecture Comparison")

# Load Bundle
bundle = load_bundle()
if bundle is None:
    if BUNDLE_LOAD_ERROR:
        st.error(f"Could not load '{BUNDLE_PATH}': {BUNDLE_LOAD_ERROR}")
    else:
        st.error(f"Could not find '{BUNDLE_PATH}'. Please run 'train.py' first.")
    st.stop()

# Load Data
test_data = load_data()

# --- Sidebar ---
st.sidebar.header("🎛️ Experiment Controls")

available_models = list(bundle["models"].keys())
bundle_model_config = bundle.get("model_config")
if bundle_model_config is None:
    bundle_model_config = {name: {"input_size": 32} for name in available_models}
selected_models = st.sidebar.multiselect(
    "Select Models to Compare", available_models, default=available_models
)

if not selected_models:
    st.warning("Please select at least one model.")
    st.stop()

input_mode = st.sidebar.selectbox("Input Source", ["Random Test Sample", "Upload Image"])

# Load selected models
active_models_dict = load_active_models(selected_models, bundle["models"])

# Input Handling
input_image = None
ground_truth_label = None

if input_mode == "Random Test Sample":
    if st.sidebar.button("🎲 Shuffle Sample"):
        st.session_state["random_idx"] = np.random.randint(0, len(test_data))

    idx = st.session_state.get("random_idx", 0)

    input_image, label = test_data[idx]
    ground_truth_label = CLASSES[label]

elif input_mode == "Upload Image":
    uploaded_file = st.sidebar.file_uploader(
        "Upload an image (jpg, png)", type=["jpg", "png", "jpeg"]
    )
    if uploaded_file is not None:
        MAX_UPLOAD_BYTES = 10 * 1024 * 1024  # 10 MB
        if uploaded_file.size is not None and uploaded_file.size > MAX_UPLOAD_BYTES:
            st.sidebar.error(
                f"File too large ({uploaded_file.size / 1024 / 1024:.1f} MB). Max allowed: 10 MB."
            )
            input_image = None
        else:
            try:
                input_image = Image.open(uploaded_file).convert("RGB")
                ground_truth_label = "Unknown"
            except (UnidentifiedImageError, OSError) as exc:
                st.sidebar.error(f"Invalid image file: {exc}")
                input_image = None

# --- Main Tabs ---
tab1, tab2, tab3, tab4 = st.tabs(
    ["🩺 Diagnosis & Consensus", "🧠 Explainability", "📊 Performance", "🌌 Latent Space"]
)

if input_image:
    # Run Inference on All Selected Models
    results = {}
    input_tensors = {}
    for name, model in active_models_dict.items():
        input_tensor = preprocess_image(input_image, name, bundle_model_config)
        input_tensors[name] = input_tensor
        with torch.no_grad():
            outputs = model(input_tensor)
            probs = torch.nn.functional.softmax(outputs, dim=1)
            conf, pred_idx = torch.max(probs, 1)
            results[name] = {
                "pred_label": CLASSES[int(pred_idx.item())],
                "confidence": conf.item(),
                "pred_idx": pred_idx.item(),
                "probs": probs.cpu().numpy().flatten(),
            }

# --- Tab 1: Diagnosis & Consensus ---
with tab1:
    col_img, col_stats = st.columns([1, 2])

    with col_img:
        if input_image:
            st.image(input_image, caption="Input Image", width=250)
            if ground_truth_label:
                st.write(f"**Ground Truth**: {ground_truth_label}")
        else:
            st.info("Upload or select an image.")

    with col_stats:
        if input_image:
            st.subheader("Model Consensus")

            # Vote Counting
            votes = [res["pred_label"] for res in results.values()]
            vote_counts = pd.Series(votes).value_counts()

            # Display Votes
            for label, count in vote_counts.items():
                st.write(f"**{label}**: {count} vote(s)")

            # Detailed Table
            st.subheader("Individual Diagnosis")
            res_df = pd.DataFrame(results).T
            res_df["confidence"] = res_df["confidence"].apply(lambda x: f"{x:.2%}")
            st.table(res_df[["pred_label", "confidence"]])
        else:
            st.write("Waiting for input...")

# --- Tab 2: Explainability (Grad-CAM) ---
with tab2:
    st.subheader("Visual Explanations (Grad-CAM)")
    st.write("Compare where each model is 'looking' to make its decision.")

    if input_image:
        cols = st.columns(len(selected_models))

        # Prepare background image for overlay
        img_np = np.array(input_image.resize((256, 256))) / 255.0

        for i, (name, model) in enumerate(active_models_dict.items()):
            with cols[i]:
                st.write(f"**{name}**")
                pred_idx = results[name]["pred_idx"]
                pred_lbl = results[name]["pred_label"]

                try:
                    grayscale_cam = get_gradcam(model, name, input_tensors[name], pred_idx)
                    cam_resized = cv2.resize(grayscale_cam, (256, 256))
                    visualization = show_cam_on_image(img_np, cam_resized, use_rgb=True)

                    st.image(visualization, caption=f"Pred: {pred_lbl}", width="stretch")
                except Exception as e:
                    st.error(f"Grad-CAM Error: {e}")
    else:
        st.info("Upload or select an image to see Grad-CAM.")

# --- Tab 3: Performance ---
with tab3:
    st.subheader("Model Performance Comparison")

    # 1. Radar Chart
    metrics_data = bundle["metrics"]
    categories = ["Accuracy", "Precision", "Recall", "F1"]

    fig = go.Figure()

    for m_name in selected_models:
        vals = [metrics_data[m_name][c] for c in categories]
        vals += [vals[0]]
        cats = [*categories, categories[0]]

        fig.add_trace(go.Scatterpolar(r=vals, theta=cats, fill="toself", name=m_name))

    fig.update_layout(polar={"radialaxis": {"visible": True, "range": [0, 1]}}, showlegend=True)
    st.plotly_chart(fig, width="stretch")

    # 2. Confusion Matrix
    st.subheader("Confusion Matrix Analysis")
    cm_model_name = st.selectbox("Select Model for Confusion Matrix", selected_models)

    if cm_model_name:
        # Calculate CM on the fly using Embeddings + Classifier Head
        vectors = bundle["search_index"][cm_model_name]["vectors"].to(DEVICE)
        labels_gt = bundle["search_index"][cm_model_name]["labels"].cpu().numpy()

        model = active_models_dict[cm_model_name]

        with torch.no_grad():
            classifier = get_classifier_layer(model, cm_model_name)
            logits = classifier(vectors)
            preds = torch.argmax(logits, dim=1).cpu().numpy()

        cm = confusion_matrix(labels_gt, preds)

        fig_cm = px.imshow(
            cm,
            text_auto=True,
            labels={"x": "Predicted", "y": "True", "color": "Count"},
            x=CLASSES,
            y=CLASSES,
            title=f"Confusion Matrix: {cm_model_name}",
        )
        st.plotly_chart(fig_cm, width="stretch")

# --- Tab 4: Latent Space ---
with tab4:
    st.subheader("Latent Space Visualization")
    st.write("Projecting high-dimensional embeddings into 2D using PCA.")

    primary_model = selected_models[0]
    st.info(f"Visualizing Latent Space for Primary Selection: **{primary_model}**")

    if st.button("Generate PCA Scatter Plot"):
        vectors = bundle["search_index"][primary_model]["vectors"].cpu().numpy()
        labels_indices = bundle["search_index"][primary_model]["labels"].cpu().numpy()
        labels_names = [CLASSES[i] for i in labels_indices]

        pca = PCA(n_components=2)
        components = pca.fit_transform(vectors)

        fig_pca = px.scatter(
            x=components[:, 0],
            y=components[:, 1],
            color=labels_names,
            title=f"PCA of {primary_model} Embeddings (10k Test Images)",
            labels={"x": "Principal Component 1", "y": "Principal Component 2"},
            opacity=0.6,
            hover_data={"Class": labels_names},
        )
        st.plotly_chart(fig_pca, width="stretch")
    else:
        st.write("Click the button to perform PCA.")
