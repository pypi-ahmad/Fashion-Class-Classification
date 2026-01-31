# Fashion Neural Lab 🔬

**A Comparative AI Laboratory for FashionMNIST Architectures**

[![Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://streamlit.io)
![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=flat&logo=PyTorch&logoColor=white)
![Python](https://img.shields.io/badge/python-3.10+-blue.svg)

**Fashion Neural Lab** is an interactive research dashboard designed to dissect and compare how different Deep Learning architectures perceive fashion items. It goes beyond simple accuracy metrics to explore the *why* and *how* of model decision-making.

---

## 🧪 Concept

Most tutorials stop at `Accuracy: 92%`. This project asks:
- *Why* did ResNet think that Shirt was a Coat?
- Does EfficientNet look at the same features as a simple Custom CNN?
- How well separable are the classes in the high-dimensional latent space?

We train three distinct architectures on **FashionMNIST** and bundle them into a unified laboratory interface:
1.  **ResNet18**: A standard industry backbone.
2.  **EfficientNet-B0**: A modern, optimized architecture.
3.  **SimpleCNN**: A lightweight, custom 3-layer baseline.

---

## ✨ Key Features

### 1. 🩺 Multi-Model Diagnosis
Feed an image (random test sample or upload) to **all three models simultaneously**.
- See a **Consensus Vote** (e.g., "2 models say Sneaker, 1 says Sandal").
- Compare confidence scores side-by-side.

### 2. 🧠 Explainable AI (XAI)
Visualize *where* the models are looking using **Grad-CAM** (Gradient-weighted Class Activation Mapping).
- Discover if the model focuses on the handle of a "Bag" or the heel of a "Ankle boot".
- Compare attention maps across architectures.

### 3. 🌌 Latent Space Explorer
Peek inside the "brain" of the neural network.
- We extract the **Feature Embeddings** (the vector representation before the final classification layer) for the entire 10,000-image test set.
- We use **PCA** (Principal Component Analysis) to project these high-dimensional vectors into 2D.
- **Goal**: See if semantically similar items (e.g., Sandals, Sneakers, Boots) cluster together.

### 4. 📊 Performance Benchmarking
- **Radar Charts**: Compare Accuracy, Precision, Recall, and F1-Score.
- **Interactive Confusion Matrix**: Drill down into specific class confusions (e.g., "How often is a Pullover confused with a Coat?").

---

## 🚀 Installation & Usage

### Prerequisites
- Python 3.8+
- CUDA-capable GPU recommended (but runs on CPU).

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Run the Experiment Runner
This script downloads data, trains all models, extracts embeddings, and saves the `fashion_bundle.pth`.
```bash
python train.py
```
*Note: Training takes approx. 5-10 minutes on a GPU.*

### 3. Launch the Laboratory
Start the Streamlit dashboard to explore the results.
```bash
streamlit run app.py
```

---

## 📂 Project Structure

- `train.py`: **The Factory**. Handles data loading, model definition, training loops, and embedding extraction. Saves the "State of the World" to `fashion_bundle.pth`.
- `app.py`: **The Interface**. A Streamlit app that loads the bundle and provides the interactive tools (Grad-CAM, PCA, etc.).
- `fashion_bundle.pth`: A compressed dictionary containing trained weights, validation metrics, and the 10k embedding vectors.

---

## 📈 Baseline Results (Example)

| Model | Accuracy | Precision | Recall | F1 Score |
| :--- | :--- | :--- | :--- | :--- |
| **ResNet18** | 93.2% | 0.93 | 0.93 | 0.93 |
| **EfficientNet-B0** | 91.8% | 0.92 | 0.92 | 0.92 |
| **SimpleCNN** | 89.5% | 0.89 | 0.90 | 0.89 |

*Results may vary based on random seed and hardware.*

---

## 📜 License
MIT License. Free for educational and research use.
