import numpy as np
import torch
from PIL import Image

import train
from tests.conftest import FakeFashionMNIST, TinyEffNet, TinyResNet


def test_training_pipeline_integration(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(train, "DEVICE", torch.device("cpu"))
    monkeypatch.setattr(train.datasets, "FashionMNIST", FakeFashionMNIST)
    monkeypatch.setattr(train.models, "resnet18", lambda weights=None: TinyResNet())
    monkeypatch.setattr(train.models, "efficientnet_b0", lambda weights=None: TinyEffNet())

    train_loader, test_loader, _ = train.get_dataloaders()
    model_zoo = train.get_models()

    assert set(model_zoo.keys()) == {"ResNet18", "EfficientNet-B0", "SimpleCNN"}

    for model_name, model in model_zoo.items():
        trained = train.train_model(model, train_loader, epochs=1)
        metrics = train.evaluate_model(trained, test_loader)
        vectors, labels, indices = train.get_embeddings(trained, test_loader, model_name)

        assert set(metrics.keys()) == {"Accuracy", "Precision", "Recall", "F1"}
        assert vectors.shape[0] == len(test_loader.dataset)
        assert labels.shape[0] == len(test_loader.dataset)
        assert len(indices) == len(test_loader.dataset)


def test_inference_pipeline_integration(app_symbols):
    get_model_architecture = app_symbols["get_model_architecture"]
    load_active_models = app_symbols["load_active_models"]
    preprocess_image = app_symbols["preprocess_image"]

    seed_model = get_model_architecture("SimpleCNN")
    active_models = load_active_models(["SimpleCNN"], {"SimpleCNN": seed_model.state_dict()})

    image = Image.fromarray(np.random.randint(0, 255, size=(64, 64, 3), dtype=np.uint8), mode="RGB")
    input_tensor = preprocess_image(image)

    outputs = active_models["SimpleCNN"](input_tensor)
    probs = torch.nn.functional.softmax(outputs, dim=1)
    confidence, pred_idx = torch.max(probs, 1)

    assert outputs.shape == (1, 10)
    assert probs.shape == (1, 10)
    assert probs.dtype.is_floating_point
    assert 0 <= int(pred_idx.item()) <= 9
    assert float(confidence.item()) >= 0.0
    assert torch.isclose(probs.sum(), torch.tensor(1.0), atol=1e-4)


def test_inference_pipeline_resnet18(app_symbols):
    get_model_architecture = app_symbols["get_model_architecture"]
    load_active_models = app_symbols["load_active_models"]
    preprocess_image = app_symbols["preprocess_image"]

    seed_model = get_model_architecture("ResNet18")
    active_models = load_active_models(["ResNet18"], {"ResNet18": seed_model.state_dict()})

    image = Image.fromarray(np.random.randint(0, 255, size=(64, 64, 3), dtype=np.uint8), mode="RGB")
    input_tensor = preprocess_image(image)

    outputs = active_models["ResNet18"](input_tensor)
    probs = torch.nn.functional.softmax(outputs, dim=1)
    confidence, pred_idx = torch.max(probs, 1)

    assert outputs.shape == (1, 10)
    assert 0 <= int(pred_idx.item()) <= 9
    assert torch.isclose(probs.sum(), torch.tensor(1.0), atol=1e-4)


def test_inference_pipeline_efficientnet_b0(app_symbols):
    get_model_architecture = app_symbols["get_model_architecture"]
    load_active_models = app_symbols["load_active_models"]
    preprocess_image = app_symbols["preprocess_image"]

    seed_model = get_model_architecture("EfficientNet-B0")
    active_models = load_active_models(["EfficientNet-B0"], {"EfficientNet-B0": seed_model.state_dict()})

    image = Image.fromarray(np.random.randint(0, 255, size=(64, 64, 3), dtype=np.uint8), mode="RGB")
    input_tensor = preprocess_image(image)

    outputs = active_models["EfficientNet-B0"](input_tensor)
    probs = torch.nn.functional.softmax(outputs, dim=1)
    confidence, pred_idx = torch.max(probs, 1)

    assert outputs.shape == (1, 10)
    assert 0 <= int(pred_idx.item()) <= 9
    assert torch.isclose(probs.sum(), torch.tensor(1.0), atol=1e-4)


def test_end_to_end_bundle_roundtrip(monkeypatch, app_symbols, tmp_path):
    monkeypatch.setattr(train, "DEVICE", torch.device("cpu"))

    get_model_architecture = app_symbols["get_model_architecture"]
    load_bundle = app_symbols["load_bundle"]
    load_active_models = app_symbols["load_active_models"]

    model = get_model_architecture("SimpleCNN")

    vectors = torch.randn(5, 64 * 4 * 4)
    labels = torch.tensor([0, 1, 2, 3, 4])
    paths = [0, 1, 2, 3, 4]

    bundle = {
        "models": {"SimpleCNN": model.state_dict()},
        "metrics": {"SimpleCNN": {"Accuracy": 0.1, "Precision": 0.1, "Recall": 0.1, "F1": 0.1}},
        "search_index": {"SimpleCNN": {"vectors": vectors, "labels": labels, "paths": paths}},
    }

    bundle_path = tmp_path / "fashion_bundle.pth"
    torch.save(bundle, bundle_path)
    app_symbols["BUNDLE_PATH"] = str(bundle_path)

    loaded_bundle = load_bundle()
    active_models = load_active_models(["SimpleCNN"], loaded_bundle["models"])

    assert "SimpleCNN" in loaded_bundle["models"]
    assert "SimpleCNN" in loaded_bundle["metrics"]
    assert "SimpleCNN" in loaded_bundle["search_index"]
    assert "SimpleCNN" in active_models
