import torch
import pytest
from torch.utils.data import DataLoader

import train
from tests.conftest import FakeFashionMNIST, TinyResNet


def test_simplecnn_forward_and_embedding_shapes(monkeypatch):
    monkeypatch.setattr(train, "DEVICE", torch.device("cpu"))
    model = train.SimpleCNN()
    batch = torch.randn(4, 3, 32, 32)

    logits = model(batch)
    embeddings = model.get_embedding(batch)

    assert logits.shape == (4, 10)
    assert embeddings.shape == (4, 64 * 4 * 4)


def test_get_dataloaders_returns_expected_shapes(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(train.datasets, "FashionMNIST", FakeFashionMNIST)

    train_loader, test_loader, test_data = train.get_dataloaders()

    images, labels = next(iter(train_loader))
    assert len(train_loader.dataset) == 16
    assert len(test_loader.dataset) == 8
    assert len(test_data) == 8
    assert images.shape[1:] == (3, 32, 32)
    assert labels.ndim == 1


def test_train_model_updates_parameters(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(train, "DEVICE", torch.device("cpu"))
    monkeypatch.setattr(train.datasets, "FashionMNIST", FakeFashionMNIST)

    train_loader, _, _ = train.get_dataloaders()
    model = train.SimpleCNN()

    before = [param.detach().clone() for param in model.parameters()]
    trained_model = train.train_model(model, train_loader, epochs=1)
    after = list(trained_model.parameters())

    assert any(not torch.equal(a, b) for a, b in zip(before, after))


def test_evaluate_model_returns_metrics(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(train, "DEVICE", torch.device("cpu"))
    monkeypatch.setattr(train.datasets, "FashionMNIST", FakeFashionMNIST)

    _, test_loader, _ = train.get_dataloaders()
    model = train.SimpleCNN().to(torch.device("cpu"))

    metrics = train.evaluate_model(model, test_loader)

    assert set(metrics.keys()) == {"Accuracy", "Precision", "Recall", "F1"}
    assert all(0.0 <= float(value) <= 1.0 for value in metrics.values())


def test_get_embeddings_simplecnn_shapes(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(train, "DEVICE", torch.device("cpu"))
    monkeypatch.setattr(train.datasets, "FashionMNIST", FakeFashionMNIST)

    _, test_loader, _ = train.get_dataloaders()
    model = train.SimpleCNN().to(torch.device("cpu"))

    vectors, labels, indices = train.get_embeddings(model, test_loader, "SimpleCNN")

    assert vectors.shape[0] == len(test_loader.dataset)
    assert vectors.shape[1] == 64 * 4 * 4
    assert labels.shape[0] == len(test_loader.dataset)
    assert len(indices) == len(test_loader.dataset)


def test_get_embeddings_resnet_path_uses_hook(monkeypatch):
    monkeypatch.setattr(train, "DEVICE", torch.device("cpu"))

    dataset = [(torch.randn(3, 32, 32), i % 10) for i in range(6)]
    loader = DataLoader(dataset, batch_size=3, shuffle=False)
    model = TinyResNet().to(torch.device("cpu"))

    vectors, labels, indices = train.get_embeddings(model, loader, "ResNet18")

    assert vectors.shape == (6, 4)
    assert labels.shape[0] == 6
    assert indices == list(range(6))


def test_get_embeddings_unknown_model_name_raises(monkeypatch):
    monkeypatch.setattr(train, "DEVICE", torch.device("cpu"))

    dataset = [(torch.randn(3, 32, 32), 0)]
    loader = DataLoader(dataset, batch_size=1, shuffle=False)
    model = TinyResNet().to(torch.device("cpu"))

    with pytest.raises(ValueError):
        train.get_embeddings(model, loader, "UnknownModel")


def test_train_model_invalid_batch_schema_raises_value_error(monkeypatch):
    monkeypatch.setattr(train, "DEVICE", torch.device("cpu"))

    dataset = [({"image": torch.randn(3, 32, 32)}, 0)]
    loader = DataLoader(dataset, batch_size=1)
    model = train.SimpleCNN().to(torch.device("cpu"))

    with pytest.raises(ValueError):
        train.train_model(model, loader, epochs=1)
