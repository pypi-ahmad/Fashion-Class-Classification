# Changelog

This file records notable changes to the project.

This project follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/)
and aims to follow [Semantic Versioning](https://semver.org/).

## [Unreleased]

### Added

- UV project metadata, Python 3.13 development pin, and committed dependency lockfile.
- Ruff and ty development checks.
- WideResNet-28-10, ConvNeXt-Tiny, MobileNetV3-Large, and EfficientNetV2-S model support.
- Shared model registry with model-specific preprocessing, training metadata, classifier layers, and Grad-CAM targets.
- Developer and operator technical guide covering architecture, internal APIs, bundle compatibility, model extension, and troubleshooting.

### Changed

- Migrated setup, run, and test workflows from pip requirements to UV.
- Added 32×32 and 224×224 training policies with augmentation and AdamW.
- Extended the bundle format with versioned per-model configuration while retaining legacy bundle loading.

### Fixed

- Made embedding extraction and confusion-matrix reconstruction architecture-independent.

## [2026-06-13]

### Added

- OSS companion documentation initialized (license, contributing, security, conduct, changelog).
