import pytest
import torch
from PIL import Image

from model_registry import (
    MODEL_CONFIGS,
    MODEL_NAMES,
    WideBasicBlock,
    WideResNet,
    build_model,
    build_transform,
    get_classifier_layer,
    get_gradcam_layer,
    serialize_model_config,
)


@pytest.mark.parametrize("model_name", MODEL_NAMES)
def test_registered_model_forward_shape(model_name):
    model = build_model(model_name, pretrained=False).eval()
    input_size = MODEL_CONFIGS[model_name].input_size
    classifier = get_classifier_layer(model, model_name)
    captured = {}

    def capture_classifier_input(_module, inputs):
        captured["embedding"] = inputs[0].detach()

    hook = classifier.register_forward_pre_hook(capture_classifier_input)

    with torch.inference_mode():
        output = model(torch.randn(1, 3, input_size, input_size))
        reconstructed = classifier(captured["embedding"])
    hook.remove()

    assert output.shape == (1, 10)
    assert torch.allclose(output, reconstructed)
    assert classifier.out_features == 10
    assert isinstance(get_gradcam_layer(model, model_name), torch.nn.Module)


@pytest.mark.parametrize("input_size", [32, 224])
def test_transforms_match_configured_resolution(input_size):
    image = Image.new("L", (28, 28), color=128)

    train_tensor = build_transform(input_size, train=True)(image)
    test_tensor = build_transform(input_size, train=False)(image)

    assert train_tensor.shape == (3, input_size, input_size)
    assert test_tensor.shape == (3, input_size, input_size)


def test_serialized_config_records_training_policy():
    serialized = serialize_model_config(MODEL_CONFIGS["WideResNet-28-10"])

    assert serialized == {
        "input_size": 32,
        "batch_size": 64,
        "epochs": 20,
        "optimizer": "AdamW",
        "learning_rate": 1e-3,
        "weight_decay": 1e-4,
        "pretrained": False,
    }


def test_wideresnet_rejects_invalid_depth():
    with pytest.raises(ValueError, match="depth"):
        WideResNet(depth=27)


def test_wide_block_preserves_identity_shortcut():
    block = WideBasicBlock(4, 4, stride=1, dropout=0).eval()
    torch.nn.init.zeros_(block.conv1.weight)
    torch.nn.init.zeros_(block.conv2.weight)
    input_tensor = torch.randn(2, 4, 8, 8)

    with torch.inference_mode():
        output = block(input_tensor)

    assert torch.equal(output, input_tensor)
