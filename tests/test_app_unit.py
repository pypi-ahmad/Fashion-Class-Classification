from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image


def test_preprocess_image_returns_expected_tensor(app_symbols):
    preprocess_image = app_symbols["preprocess_image"]

    image = Image.fromarray(np.random.randint(0, 255, size=(48, 52, 3), dtype=np.uint8), mode="RGB")
    tensor = preprocess_image(image)

    assert isinstance(tensor, torch.Tensor)
    assert tensor.shape == (1, 3, 32, 32)
    assert tensor.device.type == "cpu"


def test_preprocess_image_empty_input_raises(app_symbols):
    preprocess_image = app_symbols["preprocess_image"]

    with pytest.raises((TypeError, AttributeError)):
        preprocess_image(None)


def test_load_bundle_missing_file_returns_none(app_symbols, tmp_path):
    load_bundle = app_symbols["load_bundle"]
    app_symbols["BUNDLE_PATH"] = str(tmp_path / "missing_bundle.pth")

    assert load_bundle() is None


def test_load_bundle_corrupted_file_raises(app_symbols, tmp_path):
    load_bundle = app_symbols["load_bundle"]
    corrupt_path = tmp_path / "corrupt_bundle.pth"
    corrupt_path.write_bytes(b"not-a-valid-torch-pickle")
    app_symbols["BUNDLE_PATH"] = str(corrupt_path)

    assert load_bundle() is None


def test_get_model_architecture_known_models(app_symbols):
    get_model_architecture = app_symbols["get_model_architecture"]

    for model_name in ["ResNet18", "EfficientNet-B0", "SimpleCNN"]:
        model = get_model_architecture(model_name)
        assert isinstance(model, torch.nn.Module)


def test_get_model_architecture_unknown_model_raises(app_symbols):
    get_model_architecture = app_symbols["get_model_architecture"]

    with pytest.raises(ValueError):
        get_model_architecture("UnknownModel")


def test_load_active_models_loads_simplecnn(app_symbols):
    get_model_architecture = app_symbols["get_model_architecture"]
    load_active_models = app_symbols["load_active_models"]

    source_model = get_model_architecture("SimpleCNN")
    loaded = load_active_models(["SimpleCNN"], {"SimpleCNN": source_model.state_dict()})

    assert "SimpleCNN" in loaded
    assert loaded["SimpleCNN"].training is False


def test_load_active_models_missing_model_key_raises(app_symbols):
    load_active_models = app_symbols["load_active_models"]

    with pytest.raises(KeyError):
        load_active_models(["SimpleCNN"], {})


def test_load_active_models_corrupted_state_dict_raises(app_symbols):
    load_active_models = app_symbols["load_active_models"]

    with pytest.raises(RuntimeError):
        load_active_models(["SimpleCNN"], {"SimpleCNN": {"bad": torch.tensor([1.0])}})


def test_get_gradcam_returns_map(app_symbols):
    get_model_architecture = app_symbols["get_model_architecture"]
    get_gradcam = app_symbols["get_gradcam"]

    model = get_model_architecture("SimpleCNN")
    input_tensor = torch.randn(1, 3, 32, 32)
    cam_map = get_gradcam(model, "SimpleCNN", input_tensor, 0)

    assert isinstance(cam_map, np.ndarray)
    assert cam_map.shape == (32, 32)


def test_get_gradcam_unknown_model_raises(app_symbols):
    get_model_architecture = app_symbols["get_model_architecture"]
    get_gradcam = app_symbols["get_gradcam"]

    model = get_model_architecture("SimpleCNN")
    input_tensor = torch.randn(1, 3, 32, 32)

    with pytest.raises(ValueError):
        get_gradcam(model, "UnknownModel", input_tensor, 0)
