"""Shared model definitions and metadata for training and inference."""

from dataclasses import dataclass

import torch
import torch.nn as nn
from torchvision import models, transforms

NUM_CLASSES = 10
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


@dataclass(frozen=True)
class ModelConfig:
    input_size: int
    batch_size: int
    epochs: int
    learning_rate: float
    pretrained: bool
    weight_decay: float = 1e-4


MODEL_CONFIGS = {
    "ResNet18": ModelConfig(224, 16, 5, 1e-4, True),
    "EfficientNet-B0": ModelConfig(224, 16, 5, 1e-4, True),
    "SimpleCNN": ModelConfig(32, 64, 20, 1e-3, False),
    "WideResNet-28-10": ModelConfig(32, 64, 20, 1e-3, False),
    "ConvNeXt-Tiny": ModelConfig(224, 16, 5, 1e-4, True),
    "MobileNetV3-Large": ModelConfig(224, 16, 5, 1e-4, True),
    "EfficientNetV2-S": ModelConfig(224, 16, 5, 1e-4, True),
}
MODEL_NAMES = tuple(MODEL_CONFIGS)


class SimpleCNN(nn.Module):
    """Small convolutional baseline for 32x32 inputs."""

    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Linear(64 * 4 * 4, NUM_CLASSES)

    def forward(self, x):
        x = self.features(x)
        return self.classifier(torch.flatten(x, 1))

    def get_embedding(self, x):
        """Return the flattened feature vector before classification."""
        return torch.flatten(self.features(x), 1)


class WideBasicBlock(nn.Module):
    """Pre-activation residual block used by WideResNet."""

    def __init__(self, in_channels, out_channels, stride, dropout):
        super().__init__()
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(
            in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu2 = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(p=dropout) if dropout else nn.Identity()
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.has_identity_shortcut = stride == 1 and in_channels == out_channels
        self.shortcut = (
            nn.Identity()
            if self.has_identity_shortcut
            else nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False)
        )

    def forward(self, x):
        activated = self.relu1(self.bn1(x))
        shortcut = x if self.has_identity_shortcut else self.shortcut(activated)
        out = self.conv1(activated)
        out = self.conv2(self.dropout(self.relu2(self.bn2(out))))
        return shortcut + out


class WideResNet(nn.Module):
    """CIFAR-style WideResNet-28-10 adapted to ten Fashion-MNIST classes."""

    def __init__(self, depth=28, widen_factor=10, dropout=0.3):
        super().__init__()
        if (depth - 4) % 6 != 0:
            raise ValueError("WideResNet depth must satisfy (depth - 4) % 6 == 0.")

        blocks_per_group = (depth - 4) // 6
        channels = [16, 16 * widen_factor, 32 * widen_factor, 64 * widen_factor]
        self.conv1 = nn.Conv2d(3, channels[0], kernel_size=3, padding=1, bias=False)
        self.block1 = self._make_group(
            channels[0], channels[1], blocks_per_group, stride=1, dropout=dropout
        )
        self.block2 = self._make_group(
            channels[1], channels[2], blocks_per_group, stride=2, dropout=dropout
        )
        self.block3 = self._make_group(
            channels[2], channels[3], blocks_per_group, stride=2, dropout=dropout
        )
        self.bn = nn.BatchNorm2d(channels[3])
        self.relu = nn.ReLU(inplace=True)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(channels[3], NUM_CLASSES)
        self._initialize_weights()

    @staticmethod
    def _make_group(in_channels, out_channels, block_count, stride, dropout):
        layers = [WideBasicBlock(in_channels, out_channels, stride, dropout)]
        layers.extend(
            WideBasicBlock(out_channels, out_channels, 1, dropout) for _ in range(block_count - 1)
        )
        return nn.Sequential(*layers)

    def _initialize_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Linear):
                nn.init.zeros_(module.bias)

    def forward(self, x):
        x = self.block3(self.block2(self.block1(self.conv1(x))))
        x = self.avgpool(self.relu(self.bn(x)))
        return self.fc(torch.flatten(x, 1))


def build_transform(input_size, *, train):
    """Build the augmentation or deterministic transform for an input size."""
    if input_size == 32:
        spatial = (
            [
                transforms.Resize((32, 32)),
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
            ]
            if train
            else [transforms.Resize((32, 32))]
        )
    elif input_size == 224:
        spatial = (
            [transforms.Resize(256), transforms.RandomCrop(224), transforms.RandomHorizontalFlip()]
            if train
            else [transforms.Resize(256), transforms.CenterCrop(224)]
        )
    else:
        raise ValueError(f"Unsupported input size: {input_size}")

    return transforms.Compose(
        [
            *spatial,
            transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )


def build_model(model_name, *, pretrained):
    """Construct a supported model and replace its classifier for ten classes."""
    if model_name == "ResNet18":
        model = models.resnet18(weights="DEFAULT" if pretrained else None)
        model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)
    elif model_name == "EfficientNet-B0":
        model = models.efficientnet_b0(weights="DEFAULT" if pretrained else None)
        model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, NUM_CLASSES)
    elif model_name == "SimpleCNN":
        model = SimpleCNN()
    elif model_name == "WideResNet-28-10":
        model = WideResNet()
    elif model_name == "ConvNeXt-Tiny":
        model = models.convnext_tiny(weights="DEFAULT" if pretrained else None)
        model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, NUM_CLASSES)
    elif model_name == "MobileNetV3-Large":
        model = models.mobilenet_v3_large(weights="DEFAULT" if pretrained else None)
        model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, NUM_CLASSES)
    elif model_name == "EfficientNetV2-S":
        model = models.efficientnet_v2_s(weights="DEFAULT" if pretrained else None)
        model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, NUM_CLASSES)
    else:
        raise ValueError(f"Unsupported model architecture: {model_name}")
    return model


def get_classifier_layer(model, model_name):
    """Return the final linear layer receiving stored embeddings."""
    if model_name in {"ResNet18", "WideResNet-28-10"}:
        layer = model.fc
    elif model_name == "SimpleCNN":
        layer = model.classifier
    elif model_name in {
        "EfficientNet-B0",
        "ConvNeXt-Tiny",
        "MobileNetV3-Large",
        "EfficientNetV2-S",
    }:
        layer = model.classifier[-1]
    else:
        raise ValueError(f"Unsupported model architecture: {model_name}")

    if not isinstance(layer, nn.Linear):
        raise TypeError(f"Classifier for {model_name} is not a linear layer.")
    return layer


def get_gradcam_layer(model, model_name):
    """Return the final spatial feature layer suitable for Grad-CAM."""
    if model_name == "ResNet18":
        return model.layer4[-1]
    if model_name in {"EfficientNet-B0", "MobileNetV3-Large", "EfficientNetV2-S"}:
        return model.features[-1]
    if model_name == "SimpleCNN":
        return model.features[6]
    if model_name == "WideResNet-28-10":
        return model.block3[-1]
    if model_name == "ConvNeXt-Tiny":
        return model.features[-1][-1]
    raise ValueError(f"Unsupported model for Grad-CAM: {model_name}")


def serialize_model_config(config):
    """Return stable, torch-serializable training metadata."""
    return {
        "input_size": config.input_size,
        "batch_size": config.batch_size,
        "epochs": config.epochs,
        "optimizer": "AdamW",
        "learning_rate": config.learning_rate,
        "weight_decay": config.weight_decay,
        "pretrained": config.pretrained,
    }
