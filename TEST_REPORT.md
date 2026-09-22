# Test report

Date: 2026-09-22
Project: Fashion-Class-Classification

## Environment

- Windows 11
- Python 3.13.15 from the project `.venv`
- PyTorch 2.14.0+cu130 with CUDA 13.0
- NVIDIA CUDA runtime available: `torch.cuda.is_available() == True`

## System overview

- Training entrypoint: [train.py](train.py)
- Dashboard entrypoint: [app.py](app.py)
- Shared model registry: [model_registry.py](model_registry.py)
- Bundle artifact: `fashion_bundle.pth`
- Dependency and tool configuration: [pyproject.toml](pyproject.toml)

The training pipeline supports seven models with 32×32 or 224×224 transforms, AdamW policies, classifier-input embeddings, and versioned bundle metadata. The dashboard rebuilds models from the same registry and uses 32×32 preprocessing for legacy bundles.

## Test coverage

The suite contains 37 tests:

- [tests/test_app_unit.py](tests/test_app_unit.py): 12 helper and error-path tests.
- [tests/test_train_unit.py](tests/test_train_unit.py): 8 data, training, evaluation, and embedding tests.
- [tests/test_integration.py](tests/test_integration.py): 5 training, inference, and bundle-roundtrip tests.
- [tests/test_model_registry.py](tests/test_model_registry.py): 12 real architecture, transform, metadata, and validation tests.
- [tests/conftest.py](tests/conftest.py): isolated Streamlit, Grad-CAM, dataset, and model stubs.

## Verification

The following checks passed on 2026-09-22:

- `uv lock --check`
- `uv sync --all-groups --check`
- `uv run python -m compileall -q app.py model_registry.py train.py tests`
- `uv run ruff format --check .`
- `uv run ruff check .`
- `uv run ty check .`
- `uv run pytest -q`: 37 passed

## Verification limits

- The test suite uses local stubs and does not download FashionMNIST or pretrained weights.
- Real model forward passes use randomly initialized torchvision architectures.
- The full seven-model GPU training run and bundle regeneration were not run.
- Browser-level Streamlit rendering was not tested.
- Linux and macOS were not tested.
