"""
Fashion Neural Lab - Experiment Runner (train.py)

This script acts as the "Factory" for the Fashion Neural Lab.
It performs the following key tasks:
1.  **Data Setup**: Downloads and preprocesses FashionMNIST.
2.  **Model Instantiation**: Sets up ResNet18, EfficientNet-B0, and SimpleCNN.
3.  **Training**: Fine-tunes all models for classification.
4.  **Embedding Extraction**: Runs the test set through the trained models to
    extract latent feature vectors (embeddings) for analysis.
5.  **Bundling**: Saves models, metrics, and embeddings into 'fashion_bundle.pth'.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from tqdm import tqdm
import os
import random

# --- Configuration ---
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
EPOCHS = 5
BATCH_SIZE = 64
SEED = 42

def set_seed(seed=SEED):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if hasattr(torch.backends, 'cudnn'):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

# --- 1. Data Setup ---
def get_dataloaders():
    """
    Downloads FashionMNIST and prepares DataLoaders.

    Returns:
        train_loader (DataLoader): For training.
        test_loader (DataLoader): For evaluation and embedding extraction.
        test_data (Dataset): Original dataset for access to raw images if needed.
    """
    # Transform: 
    # 1. Resize to 32x32 (standard for many backbones like ResNet on CIFAR/small images).
    # 2. Grayscale -> RGB (to support pre-trained models expecting 3 channels).
    # 3. Standard ImageNet Normalization.
    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    os.makedirs('./data', exist_ok=True)

    train_data = datasets.FashionMNIST(root='./data', train=True, download=True, transform=transform)
    test_data = datasets.FashionMNIST(root='./data', train=False, download=True, transform=transform)

    train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=False)
    
    return train_loader, test_loader, test_data

# --- 2. Model Architectures ---
class SimpleCNN(nn.Module):
    """
    A custom, lightweight Convolutional Neural Network.
    Serves as a baseline to compare against heavyweights like ResNet.
    """
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2), # 32 -> 16
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2), # 16 -> 8
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)  # 8 -> 4
        )
        # Flatten: 64 channels * 4 * 4 spatial size
        self.flatten_dim = 64 * 4 * 4
        self.classifier = nn.Linear(self.flatten_dim, 10)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x
    
    def get_embedding(self, x):
        """Returns the flattened feature vector before the classifier."""
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return x

def get_models():
    """
    Initializes the model zoo.
    
    Returns:
        models_dict (dict): Dictionary of {name: model_instance}.
    """
    models_dict = {}
    
    # 1. ResNet18: Lightweight residual network.
    print("Loading ResNet18...")
    resnet = models.resnet18(weights='DEFAULT')
    # Replace the final fully connected layer for 10 classes.
    resnet.fc = nn.Linear(resnet.fc.in_features, 10)
    models_dict['ResNet18'] = resnet
    
    # 2. EfficientNet-B0: Optimized for efficiency.
    print("Loading EfficientNet-B0...")
    effnet = models.efficientnet_b0(weights='DEFAULT')
    # EfficientNet's classifier is a Sequential block. We replace the Linear layer.
    effnet.classifier[1] = nn.Linear(effnet.classifier[1].in_features, 10)
    models_dict['EfficientNet-B0'] = effnet
    
    # 3. SimpleCNN: Custom baseline.
    print("Initializing SimpleCNN...")
    models_dict['SimpleCNN'] = SimpleCNN()
    
    return models_dict

# --- 3. Training Loop ---
def train_model(model, train_loader, epochs=EPOCHS):
    """
    Standard PyTorch training loop.
    """
    model = model.to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
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
            pbar.set_postfix({'loss': running_loss / (pbar.n + 1)})
            
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
    p, r, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average='weighted', zero_division=0)
    
    return {'Accuracy': acc, 'Precision': p, 'Recall': r, 'F1': f1}

# --- 4. Search Index Generation (Embeddings) ---
def get_embeddings(model, loader, model_name):
    """
    Extracts latent representations (embeddings) for all images in the loader.
    Crucial for PCA and Latent Space Analysis.

    For Backbones (ResNet, EffNet), we use Forward Hooks to intercept
    the output of the Global Average Pooling layer.
    """
    supported_models = {'ResNet18', 'EfficientNet-B0', 'SimpleCNN'}
    if model_name not in supported_models:
        raise ValueError(f"Unsupported model for embedding extraction: {model_name}")

    model = model.to(DEVICE)
    model.eval()
    embeddings = []
    labels_list = []
    indices_list = []
    
    # Dictionary to store hook output
    activation = {}
    def get_activation(name):
        def hook(model, input, output):
            activation[name] = output.detach()
        return hook

    hook_handle = None
    # Register hooks based on architecture
    if model_name == 'ResNet18':
        # avgpool layer is immediately before the fc (classifier)
        hook_handle = model.avgpool.register_forward_hook(get_activation('embedding'))
    elif model_name == 'EfficientNet-B0':
        # avgpool is the penultimate layer block
        hook_handle = model.avgpool.register_forward_hook(get_activation('embedding'))
    
    print(f"Extracting embeddings for {model_name}...")
    with torch.no_grad():
        for batch_idx, (images, labels) in enumerate(tqdm(loader)):
            images = images.to(DEVICE)
            
            if model_name == 'SimpleCNN':
                # Custom model has explicit method
                emb = model.get_embedding(images)
            else:
                # Trigger forward pass to fire the hook
                _ = model(images)
                emb = activation['embedding']
                emb = torch.flatten(emb, 1)
            
            embeddings.append(emb.cpu())
            labels_list.append(labels)
            
            # Keep track of indices to map back to original dataset
            start_idx = batch_idx * loader.batch_size
            end_idx = start_idx + len(labels)
            indices_list.extend(list(range(start_idx, end_idx)))

    if hook_handle:
        hook_handle.remove()
        
    return torch.cat(embeddings), torch.cat(labels_list), indices_list

# --- Main Execution ---
if __name__ == "__main__":
    print(f"Using device: {DEVICE}")
    set_seed()
    train_loader, test_loader, _ = get_dataloaders()
    models_collection = get_models()
    
    # The Bundle stores everything the App needs
    bundle = {
        'models': {},
        'metrics': {},
        'search_index': {}
    }
    
    for name, model in models_collection.items():
        print(f"\n--- Processing {name} ---")
        
        print(f"Training...")
        model = train_model(model, train_loader)
        
        print(f"Evaluating...")
        metrics = evaluate_model(model, test_loader)
        print(f"Metrics: {metrics}")
        
        print(f"Generating Latent Embeddings...")
        vectors, labels, paths = get_embeddings(model, test_loader, name)
        
        bundle['models'][name] = model.state_dict()
        bundle['metrics'][name] = metrics
        bundle['search_index'][name] = {
            'vectors': vectors, # (N, Dim) Tensor
            'labels': labels,   # (N,) Tensor
            'paths': paths      # List of indices
        }
        
    torch.save(bundle, 'fashion_bundle.pth')
    print("\n✅ Success! Saved 'fashion_bundle.pth'. You can now run 'streamlit run app.py'.")
