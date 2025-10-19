# Changelog

All notable changes to NeuroScope will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/), and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).


## [0.2.2] - 2025-10-19

This release fixes critical alpha dropout implementation bug and incoporates reimplementation according to official paper Klambauer et al. (2017).

## [0.2.1] - 2025-10-17

This release fixes three critical gradient computation bugs that affected backpropagation accuracy:

### Fixed
- **MSE Loss Gradient**: Fixed missing `2/n_out` scaling factor
- **MSE with Sigmoid/Tanh**: Now correctly applies activation derivatives
- **Dropout Backpropagation**: Masks now properly saved and applied
- Gradient checking now passes with errors < 1e-06

### Added
- New functions: `inverted_dropout_with_mask()` and `alpha_dropout_with_mask()`
- Comprehensive gradient verification suite
- 20+ new tests for dropout and gradient checking

### Changed
- `forward_mlp()` now returns 3-tuple: (activations, z_values, dropout_masks)
- `backward_mlp()` accepts optional `dropout_masks` parameter
- All changes are backward compatible


## [0.2.0] - 2025-10-16

### Major Features

#### Advanced Optimizer Suite
- **SGD with Momentum** - Polyak momentum for accelerated convergence (Polyak, 1964)
  - Configurable momentum parameter (beta=0.9 default)
  - Smooth gradient updates with velocity accumulation
  - Improved convergence on non-convex surfaces
  
- **SGD with Nesterov Momentum** - Lookahead momentum for superior convergence (Nesterov, 1983)
  - Nesterov accelerated gradient (NAG) implementation
  - Anticipatory gradient computation at lookahead position
  - Better handling of sharp curvature regions
  - Based on Sutskever et al. (2013) formulation
  
- **RMSprop** - Adaptive learning rate optimization (Hinton, 2012)
  - Per-parameter adaptive learning rates
  - Exponential moving average of squared gradients
  - Configurable decay rate (beta=0.9 default)
  - Epsilon parameter for numerical stability (eps=1e-8)
  - Optional Nesterov momentum support
  
- **Enhanced Adam** - Production-ready adaptive optimizer (Kingma & Ba, 2014)
  - First and second moment estimation with bias correction
  - Configurable beta1=0.9, beta2=0.999, eps=1e-8
  - Optimized implementation with < 1e-6 numerical difference from PyTorch
  - Nesterov momentum variant support
  
- **Optimizer Selection Guide** - Choose the right optimizer:
  - `"sgd"` - Simple SGD for basic tasks and full control
  - `"sgdm"` - SGD with momentum for faster convergence
  - `"sgdnm"` - SGD with Nesterov momentum for best momentum-based convergence
  - `"rmsprop"` - Adaptive learning for non-stationary objectives
  - `"adam"` - Default choice for most deep learning tasks (recommended)

#### Model Persistence System
- **Save Trained Models** - Complete model serialization
  - `.save(filepath)` method with custom `.ns` format
  - Saves weights, biases, architecture, and training configuration
  - Automatic directory creation with pathlib support
  - Optional optimizer state persistence with `save_optimizer=True`
  
- **Load Models** - Full model restoration
  - `.load(filepath)` class method for model loading
  - Automatic architecture reconstruction from saved metadata
  - Optional optimizer state restoration with `load_optimizer=True`
  - Exact prediction matching (< 1e-10 difference after load)
  
- **Training Resumption** - Continue training from checkpoints
  - Load model with optimizer state for seamless continuation
  - Preserves momentum buffers, Adam moments, and RMSprop caches
  - Smooth loss continuation (< 10% jump after loading)
  - Example: `model = MLP.load('checkpoint.ns', load_optimizer=True)`
  
- **State Persistence** - Complete training state capture
  - Optimizer internal states (velocity, momentum, adaptive rates)
  - Training history and epoch counters
  - Regularization and dropout configurations
  - Enable with `model.save('model.ns', save_optimizer=True)`

### Documentation 
  - Complete guide coverage
  - Quickstart guide with optimizer examples
  - Advanced usage patterns and best practices
  - Optimizer selection 
  - Save/load workflow examples

### Implementation Quality

- **Literature-Validated** - Exact implementations from original papers
  - SGD: Robbins & Monro (1951)
  - Momentum: Polyak (1964)
  - Nesterov: Nesterov (1983), Sutskever et al. (2013)
  - RMSprop: Hinton (2012) Lecture 6.5
  - Adam: Kingma & Ba (2014) Algorithm 1

### API Enhancements

```python
# New optimizer options in compile()
model.compile(
    optimizer="adam",      # or "sgd", "sgdm", "sgdnm", "rmsprop"
    lr=1e-3,
)

# Save model with optimizer state
model.save("trained_model.ns", save_optimizer=True)

# Load and continue training
model = MLP.load("trained_model.ns", load_optimizer=True)
history = model.fit(X_train, y_train, epochs=50)  # Seamless continuation
```

### Breaking Changes

**None** - This release is fully backward compatible with v0.1.3

### Migration Guide

No migration required! All existing code continues to work:
- Default optimizer remains "adam"
- All v0.1.3 APIs unchanged
- New features are opt-in only

To use new features:
```python
# Try different optimizers
model.compile(optimizer="sgdnm", lr=1e-2)  # Nesterov momentum

# Save your trained model
model.save("my_model.ns", save_optimizer=True)

# Load and continue training later
model = MLP.load("my_model.ns", load_optimizer=True)
```

### Performance Notes

- No performance regression in existing functionality
- New optimizers have comparable computational cost to Adam
- Save/load operations are fast (< 1 second for typical models)
- Memory overhead for optimizer state is minimal (< 2x parameter count)

### Acknowledgments

This release represents a major step forward in NeuroScope's maturity, bringing production-grade optimizer implementations validated against academic literature and industry-standard frameworks. Special thanks to the neural network research community for the foundational work that made these implementations possible.

## [0.1.3] - 2025-09-24

### Fixed:
- Minimal update in docs (added plot_curves_fast())
- Resolved unwanted verbose in training caused by numerical checks 
- some bugs and code formatting fixed


## [0.1.2] - 2025-09-23

### Important:
Earlier releases v0.1.0 / v0.1.1 contained a critical bug and have been yanked on PyPI.
Upgrade: 
```pip install --upgrade neuroscope (or pip install neuroscope==0.1.2)```

### Fixed:

- Fixed plot_learning_curves() breaking for fit_fast() and made seperate method plot_curves_fast().

- Added tests and CI checks to prevent regression.

- Packaging/docs cleanups.

## [0.1.1] - 2024-09-23

### Fixed
- **PyPI Classification** - Updated development status from Alpha to Production/Stable
- **Package Metadata** - Added comprehensive PyPI classifiers for better discoverability
- **Status Badges** - PyPI badge now correctly shows "Stable" instead of "Alpha"

## [0.1.0] - 2024-09-23

### Initial Release

NeuroScope's first stable release provides a comprehensive framework for neural network training, diagnostics, and visualization with a focus on education and rapid prototyping.

### Features

#### Core Neural Network
- **Multi-Layer Perceptron (MLP)** - Modern, flexible architecture with configurable layers
- **Activation Functions** - ReLU, Leaky ReLU, Sigmoid, Tanh, SELU, Softmax
- **Loss Functions** - MSE, Binary/Categorical Cross-Entropy with L2 regularization support
- **Metrics** - Accuracy, Precision, Recall, F1-Score, R2-Score, MAE, MSE, RMSE
- **Optimizers** - Adam, SGD
- **Weight Initialization** - He, Xavier, Random, SELU initialization strategies
- **Dropout Regularization** - Configurable dropout rates for overfitting prevention
- **Dropout Type** - Standard, AlphaDropout

#### Advanced Diagnostics
- **Pre-Training Analysis** - Comprehensive model checks before training begins;
  - Initial Loss Analysis
  - Layer Capacity Analysis
  - Archtecture Sanity Analysis
  - Weight initialization Analysis
  - Capacity to data ratio Analysis
  - Convergance feasibilty Analysis
- **Real-Time Training Monitoring** - Live diagnostics during training
    - Learning progress
    - Overfitting detection
    - Weight health analysis
    - Training plateau detection
    - Dead ReLU neurons detection
    - Gradient signal-to-noise ratio
    - Weight update /magnitude ratios detection
    - Vanishing Gradient Problem (VGP) detection
    - Exploding Gradient Problem (EGP) detection  
    - Activation saturation detection (tanh/sigmoid)

- **Post-Training Evaluation** - Comprehensive model assessment
  - Robustness Evaluation
  - Model Stability Evaluation
  - Model Performance Evaluation
  - Performance metrics (accuracy, precision, recall, F1-score)

#### High-Performance Training
- **Standard Training** (`fit()`) - Full diagnostic capabilities with monitoring
- **Fast Training** (`fit_fast()`) - Optimized for fast training with ~5-10× speedup
- **Batch Processing** - Efficient mini-batch gradient descent
- **Early Stopping** - Automatic training termination on convergence
- **Learning Rate Scheduling** - Adaptive learning rate adjustment

#### Visualization & Analysis
- **Learning Curves** - Training/validation loss and metrics with confidence intervals
- **Gradient Distribution** - Layer-wise gradient histogram analysis
- **Weight Distribution** - Layer-wise weight histogram analysis
- **Activation Distribution** - Layer-wise activation histogram analysis
- **Gradient Stats over epochs** - Layer-wise gradient stats analysis over epochs
- **Weight Stats over epochs** - Layer-wise weight stats analysis over epochs
- **Activation Stats over epochs** - Layer-wise activation stats analysis over epochs
- **Gradient Norms over epochs** - Gradient norm tracking over epochs
- **Training Animation** - Beautiful training progress animation of training dynamics

#### Developer Experience
- **Clean API** - Intuitive, education-focused interface
- **Examples** - Jupyter notebooks for all use cases
- **Type Hints** - Full type annotation for IDE support
- **Documentation** - Extensive API documentation and tutorials
- **Testing** - Comprehensive test suite with 55%+ coverage

### Supported Tasks
- **Binary Classification** - Sigmoid activation with BCE loss
- **Multiclass Classification** - Softmax activation with CCE loss  
- **Regression** - Linear activation with MSE loss

### Key Metrics & Performance
- **Training Speed** - ~5-10× faster with `fit_fast()` mode
- **Memory Efficiency** - 60-80% memory reduction in fast mode
- **Cross-Platform** - Windows, macOS, Linux support
- **Python Compatibility** - Python 3.11+ support
- **Dependencies** - Minimal dependency footprint (NumPy + Matplotlib)

### Documentation & Examples
- **Getting Started Guide** - Step-by-step tutorial for beginners
- **API Reference** - Complete function and class documentation
- **Jupyter Notebooks** - Interactive examples for all features:
  - Binary Classification with diagnostics
  - Multiclass Classification with monitoring
  - Regression with performance analysis
  - High-speed training for speed boosts    
- **Performance Benchmarks** - Speed and accuracy comparisons
- **Best Practices** - Guidelines for optimal usage

### Technical Specifications
- **Architecture** - Modular design with clear separation of concerns
- **Numerical Stability** - Robust implementations preventing overflow/underflow
- **Error Handling** - Comprehensive error messages and validation
- **Extensibility** - Plugin architecture for custom components

### Getting Started

```python
import numpy as np
from neuroscope import MLP, PreTrainingAnalyzer, TrainingMonitor, Visualizer

# Create model
model = MLP([20, 64, 32, 1], activation="leaky_relu", out_activation="sigmoid")
model.compile(optimizer="adam", lr=0.001, reg="l2", lamda=0.01)

# Pre-training analysis
analyzer = PreTrainingAnalyzer(model)
analyzer.analyze(X_train, y_train)

# Train with monitoring
monitor = TrainingMonitor()
history = model.fit(X_train, y_train, X_val, y_val, 
                   epochs=100, monitor=monitor)

# Visualize results
viz = Visualizer(history)
viz.plot_learning_curves()
```

### Target Audience
- **Students & Educators** - Learning neural network fundamentals
- **Researchers** - Rapid prototyping and experimentation
- **Practitioners** - Informed model development

### Installation
```bash
pip install neuroscope
```

### Links
- **Documentation**: https://www.neuroscope.dev/
- **Repository**: https://github.com/ahmadrazacdx/neuro-scope
- **PyPI**: https://pypi.org/project/neuroscope/
- **Examples**: https://github.com/ahmadrazacdx/neuro-scope/blob/main/examples/

---

**Note**: This is NeuroScope's inaugural release. Future versions will maintain backward compatibility while adding advanced features like convolutional layers, recurrent networks, and distributed training capabilities.

**Contributors**: Ahmad Raza (@ahmadrazacdx)
**License**: Apache 2.0
