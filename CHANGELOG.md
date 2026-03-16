# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.1] - 2026-03-16

### Added
- Official package release preparation
- Multi-model framework with base model implementation
- RFM (Reflex Feedback Model) implementation  
- Support for Peterka 2018 and Assländer 2023 model variants
- Data class for standardized time series handling
- Frequency domain analysis tools
- Biomechanics calculation utilities
- Integration with anaropia virtual reality software
- Comprehensive documentation with Sphinx
- Optional dependencies for notebooks and development

### Changed
- Relaxed Python version requirement to >=3.9 (was restricted to 3.12 only)
- Reorganized dependencies into core and optional groups
- Improved package metadata and classifiers
- Enhanced __init__.py with proper __all__ exports

### Fixed
- Removed hatchling from runtime dependencies (build-system only)
- Fixed typos in README and documentation
- Improved type hints coverage

## [0.0.1] - 2024

### Added
- Initial development version
- Core functionality for balance analysis
- Model identification and parameter recovery
