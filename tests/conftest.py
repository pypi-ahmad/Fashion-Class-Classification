import ast
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import Dataset
from torchvision import datasets, models, transforms


class FakeStreamlit:
    def cache_resource(self, func=None):
        if func is None:
            def decorator(inner):
                return inner
            return decorator
        return func


class DummyClassifierOutputTarget:
    def __init__(self, class_idx):
        self.class_idx = class_idx


class DummyGradCAM:
    def __init__(self, model, target_layers):
        self.model = model
        self.target_layers = target_layers

    def __call__(self, input_tensor, targets=None):
        height = int(input_tensor.shape[-2])
        width = int(input_tensor.shape[-1])
        return np.zeros((1, height, width), dtype=np.float32)


class FakeFashionMNIST(Dataset):
    def __init__(self, root, train, download, transform=None):
        self.root = root
        self.train = train
        self.download = download
        self.transform = transform
        self.length = 16 if train else 8

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        img_array = np.full((28, 28), idx % 255, dtype=np.uint8)
        image = Image.fromarray(img_array, mode="L")
        label = idx % 10
        if self.transform is not None:
            image = self.transform(image)
        return image, label


class TinyResNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, 4, kernel_size=3, padding=1),
            nn.ReLU(),
        )
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(4, 1000)

    def forward(self, x):
        x = self.stem(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x


class TinyEffNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 4, kernel_size=3, padding=1),
            nn.ReLU(),
        )
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(nn.Dropout(p=0.0), nn.Linear(4, 1000))

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


@pytest.fixture
def app_symbols():
    app_path = Path(__file__).resolve().parents[1] / "app.py"
    source = app_path.read_text(encoding="utf-8")
    tree = ast.parse(source)

    keep_names = {
        "SimpleCNN",
        "load_data",
        "load_bundle",
        "get_model_architecture",
        "load_active_models",
        "preprocess_image",
        "get_gradcam",
    }
    selected_nodes = []
    for node in tree.body:
        if isinstance(node, (ast.ClassDef, ast.FunctionDef)) and node.name in keep_names:
            selected_nodes.append(node)

    module = ast.Module(body=selected_nodes, type_ignores=[])

    namespace = {
        "st": FakeStreamlit(),
        "torch": torch,
        "nn": nn,
        "models": models,
        "transforms": transforms,
        "datasets": datasets,
        "Image": Image,
        "np": np,
        "os": __import__("os"),
        "DEVICE": torch.device("cpu"),
        "BUNDLE_PATH": "fashion_bundle.pth",
        "IMAGE_TRANSFORM": transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]),
        "GradCAM": DummyGradCAM,
        "ClassifierOutputTarget": DummyClassifierOutputTarget,
    }

    exec(compile(module, filename=str(app_path), mode="exec"), namespace)
    return namespace
