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

import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms, datasets
from PIL import Image, UnidentifiedImageError
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image
import os
import cv2
import pandas as pd

# --- Configuration & Setup ---
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
BUNDLE_PATH = 'fashion_bundle.pth'
BUNDLE_LOAD_ERROR = None

IMAGE_TRANSFORM = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

st.set_page_config(page_title="Fashion Neural Lab", layout="wide", page_icon="🔬")

# --- Model Definitions (Must match train.py) ---
# We define SimpleCNN here to ensure we can load the state dict correctly
# without depending on importing from train.py (which can run code).
class SimpleCNN(nn.Module):
    """
    Same architecture as in train.py.
    """
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.flatten_dim = 64 * 4 * 4
        self.classifier = nn.Linear(self.flatten_dim, 10)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x
    
    def get_embedding(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return x

CLASSES = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

# --- Helper Functions ---

@st.cache_resource
def load_data():
    """
    Load the FashionMNIST test dataset (images only).
    Used to pull random samples for testing.
    """
    # Ensure data directory exists
    os.makedirs('./data', exist_ok=True)
    test_data = datasets.FashionMNIST(root='./data', train=False, download=True, transform=IMAGE_TRANSFORM)
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
    if model_name == 'ResNet18':
        model = models.resnet18(weights=None)
        model.fc = nn.Linear(model.fc.in_features, 10)
    elif model_name == 'EfficientNet-B0':
        model = models.efficientnet_b0(weights=None)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, 10)
    elif model_name == 'SimpleCNN':
        model = SimpleCNN()
    else:
        raise ValueError(f"Unsupported model architecture: {model_name}")
    return model

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

def preprocess_image(image):
    """
    Preprocess a PIL image for the model:
    Resize(32) -> Grayscale(3) -> Tensor -> Normalize -> Batch Dim.
    """
    return IMAGE_TRANSFORM(image).unsqueeze(0).to(DEVICE)

def get_gradcam(model, model_name, input_tensor, target_class_idx):
    """
    Generate Grad-CAM heatmap for explainability.
    
    Args:
        target_layers: The last convolutional layer to compute gradients against.
    """
    if model_name == 'ResNet18':
        target_layers = [model.layer4[-1]]
    elif model_name == 'EfficientNet-B0':
        # EfficientNet features are in .features
        target_layers = [model.features[-1]]
    elif model_name == 'SimpleCNN':
        target_layers = [model.features[-1]]
    else:
        raise ValueError(f"Unsupported model for Grad-CAM: {model_name}")
    
    cam = GradCAM(model=model, target_layers=target_layers)
    targets = [ClassifierOutputTarget(target_class_idx)]
    
    grayscale_cam = cam(input_tensor=input_tensor, targets=targets)
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

available_models = list(bundle['models'].keys())
selected_models = st.sidebar.multiselect(
    "Select Models to Compare",
    available_models,
    default=available_models
)

if not selected_models:
    st.warning("Please select at least one model.")
    st.stop()

input_mode = st.sidebar.selectbox("Input Source", ["Random Test Sample", "Upload Image"])

# Load selected models
active_models_dict = load_active_models(selected_models, bundle['models'])

# Input Handling
input_image = None
ground_truth_label = None

if input_mode == "Random Test Sample":
    if st.sidebar.button("🎲 Shuffle Sample"):
        st.session_state['random_idx'] = np.random.randint(0, len(test_data))
    
    idx = st.session_state.get('random_idx', 0)
    
    img_tensor, label = test_data[idx]
    ground_truth_label = CLASSES[label]
    
    # Denormalize for display
    inv_normalize = transforms.Normalize(
        mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225],
        std=[1/0.229, 1/0.224, 1/0.225]
    )
    img_display_tensor = inv_normalize(img_tensor)
    img_display = transforms.ToPILImage()(img_display_tensor.clamp(0, 1))
    input_image = img_display

elif input_mode == "Upload Image":
    uploaded_file = st.sidebar.file_uploader("Upload an image (jpg, png)", type=["jpg", "png", "jpeg"])
    if uploaded_file is not None:
        MAX_UPLOAD_BYTES = 10 * 1024 * 1024  # 10 MB
        if uploaded_file.size is not None and uploaded_file.size > MAX_UPLOAD_BYTES:
            st.sidebar.error(f"File too large ({uploaded_file.size / 1024 / 1024:.1f} MB). Max allowed: 10 MB.")
            input_image = None
        else:
            try:
                input_image = Image.open(uploaded_file).convert('RGB')
                ground_truth_label = "Unknown"
            except (UnidentifiedImageError, OSError) as exc:
                st.sidebar.error(f"Invalid image file: {exc}")
                input_image = None

# --- Main Tabs ---
tab1, tab2, tab3, tab4 = st.tabs(["🩺 Diagnosis & Consensus", "🧠 Explainability", "📊 Performance", "🌌 Latent Space"])

if input_image:
    input_tensor = preprocess_image(input_image)
    
    # Run Inference on All Selected Models
    results = {}
    for name, model in active_models_dict.items():
        with torch.no_grad():
            outputs = model(input_tensor)
            probs = torch.nn.functional.softmax(outputs, dim=1)
            conf, pred_idx = torch.max(probs, 1)
            results[name] = {
                'pred_label': CLASSES[pred_idx.item()],
                'confidence': conf.item(),
                'pred_idx': pred_idx.item(),
                'probs': probs.cpu().numpy().flatten()
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
            votes = [res['pred_label'] for res in results.values()]
            vote_counts = pd.Series(votes).value_counts()
            
            # Display Votes
            for label, count in vote_counts.items():
                st.write(f"**{label}**: {count} vote(s)")
            
            # Detailed Table
            st.subheader("Individual Diagnosis")
            res_df = pd.DataFrame(results).T
            res_df['confidence'] = res_df['confidence'].apply(lambda x: f"{x:.2%}")
            st.table(res_df[['pred_label', 'confidence']])
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
                pred_idx = results[name]['pred_idx']
                pred_lbl = results[name]['pred_label']
                
                try:
                    grayscale_cam = get_gradcam(model, name, input_tensor, pred_idx)
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
    metrics_data = bundle['metrics']
    categories = ['Accuracy', 'Precision', 'Recall', 'F1']
    
    fig = go.Figure()
    
    for m_name in selected_models:
        vals = [metrics_data[m_name][c] for c in categories]
        vals += [vals[0]]
        cats = categories + [categories[0]]
        
        fig.add_trace(go.Scatterpolar(
            r=vals,
            theta=cats,
            fill='toself',
            name=m_name
        ))
        
    fig.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0, 1])), showlegend=True)
    st.plotly_chart(fig, width="stretch")
    
    # 2. Confusion Matrix
    st.subheader("Confusion Matrix Analysis")
    cm_model_name = st.selectbox("Select Model for Confusion Matrix", selected_models)
    
    if cm_model_name:
        # Calculate CM on the fly using Embeddings + Classifier Head
        vectors = bundle['search_index'][cm_model_name]['vectors'].to(DEVICE)
        labels_gt = bundle['search_index'][cm_model_name]['labels'].cpu().numpy()
        
        model = active_models_dict[cm_model_name]
        
        with torch.no_grad():
            if cm_model_name == 'ResNet18':
                head = model.fc
                logits = head(vectors)
            elif cm_model_name == 'EfficientNet-B0':
                logits = model.classifier(vectors)
            else: # SimpleCNN
                head = model.classifier
                logits = head(vectors)
                
            preds = torch.argmax(logits, dim=1).cpu().numpy()
            
        cm = confusion_matrix(labels_gt, preds)
        
        fig_cm = px.imshow(cm, text_auto=True, 
                        labels=dict(x="Predicted", y="True", color="Count"),
                        x=CLASSES, y=CLASSES,
                        title=f"Confusion Matrix: {cm_model_name}")
        st.plotly_chart(fig_cm, width="stretch")

# --- Tab 4: Latent Space ---
with tab4:
    st.subheader("Latent Space Visualization")
    st.write("Projecting high-dimensional embeddings into 2D using PCA.")
    
    primary_model = selected_models[0]
    st.info(f"Visualizing Latent Space for Primary Selection: **{primary_model}**")
    
    if st.button("Generate PCA Scatter Plot"):
        vectors = bundle['search_index'][primary_model]['vectors'].cpu().numpy()
        labels_indices = bundle['search_index'][primary_model]['labels'].cpu().numpy()
        labels_names = [CLASSES[i] for i in labels_indices]
        
        pca = PCA(n_components=2)
        components = pca.fit_transform(vectors)
        
        fig_pca = px.scatter(
            x=components[:,0], y=components[:,1],
            color=labels_names,
            title=f"PCA of {primary_model} Embeddings (10k Test Images)",
            labels={'x': 'Principal Component 1', 'y': 'Principal Component 2'},
            opacity=0.6,
            hover_data={'Class': labels_names}
        )
        st.plotly_chart(fig_pca, width="stretch")
    else:
        st.write("Click the button to perform PCA.")
