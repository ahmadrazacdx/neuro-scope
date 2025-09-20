# NeuroScope

[![PyPI](https://img.shields.io/pypi/v/neuroscope.svg)][pypi status]
[![Status](https://img.shields.io/pypi/status/neuroscope.svg)][pypi status]
[![Python Version](https://img.shields.io/pypi/pyversions/neuroscope)][pypi status]
[![License](https://img.shields.io/pypi/l/neuroscope)][license]

[![Documentation](https://img.shields.io/badge/docs-github--pages-blue)][read the docs]
[![Tests](https://github.com/ahmadrazacdx/neuro-scope/workflows/Tests/badge.svg)][tests]
[![Codecov](https://codecov.io/gh/ahmadrazacdx/neuro-scope/branch/main/graph/badge.svg)][codecov]

[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit&logoColor=white)][pre-commit]
[![Black](https://img.shields.io/badge/code%20style-black-000000.svg)][black]

[pypi status]: https://pypi.org/project/neuroscope/
[read the docs]: https://ahmadrazacdx.github.io/neuroscope/
[tests]: https://github.com/ahmadrazacdx/neuro-scope/actions?workflow=Tests
[codecov]: https://app.codecov.io/gh/ahmadrazacdx/neuro-scope
[pre-commit]: https://github.com/pre-commit/pre-commit
[black]: https://github.com/psf/black

**A microscope for neural networks** - Comprehensive framework for building, training, and diagnosing multi-layer perceptrons with advanced monitoring and visualization capabilities.

## Features

### Modern MLP Implementation
- **Flexible Architecture**: Arbitrary layer sizes with customizable activations
- **Advanced Optimizers**: Adam, SGD with momentum, RMSprop, and adaptive learning rates
- **Smart Initialization**: He, Xavier, SELU, and intelligent auto-selection
- **Regularization**: L1/L2 regularization, dropout with multiple variants

### Comprehensive Diagnostics
- **Pre-Training Analysis**: Architecture validation, weight initialization checks
- **Real-Time Monitoring**: Dead neuron detection, gradient flow analysis
- **Post-Training Evaluation**: Robustness testing, performance profiling
- **Research-Validated Metrics**: Based on established deep learning principles

### Publication-Quality Visualization
- **Training Dynamics**: Learning curves, loss landscapes, convergence analysis
- **Network Internals**: Activation distributions, gradient flows, weight evolution
- **Diagnostic Plots**: Health indicators, training stability metrics
- **Interactive Animations**: Training progress visualization

### Developer Experience
- **Clean API**: Intuitive interface with sensible defaults
- **Type Safety**: Full type hints and runtime validation
- **Comprehensive Testing**: 95%+ test coverage with property-based testing
- **Production Ready**: Extensive documentation, CI/CD, and quality assurance

## Requirements

- **Python**: 3.11+ (3.12 recommended)
- **Core Dependencies**: NumPy 2.3+, Matplotlib 3.10+
- **Optional**: Jupyter for interactive examples

## Quick Start

### Installation

```bash
# Install from PyPI (recommended)
pip install neuroscope

# Install from source (development)
git clone https://github.com/ahmadrazacdx/neuro-scope.git
cd neuro-scope
pip install -e .
```

### Basic Usage

```python
import numpy as np
from neuroscope import MLP, PreTrainingAnalyzer, TrainingMonitor, Visualizer

# Create and configure model
model = MLP([784, 128, 64, 10], 
           hidden_activation="relu", 
           out_activation="softmax")
model.compile(optimizer="adam", lr=1e-3)

# Pre-training analysis
analyzer = PreTrainingAnalyzer(model)
pre_results = analyzer.analyze(X_train, y_train)

# Train with monitoring
monitor = TrainingMonitor()
history = model.fit(X_train, y_train, X_test=X_val, y_test=y_val,
                   epochs=100, monitor=monitor)

# Visualize results
viz = Visualizer(history)
viz.plot_learning_curves()
viz.plot_activation_hist()
```

### Direct Function Access

```python
from neuroscope import mse, accuracy_binary, relu, he_init

# Use functions directly without class instantiation
loss = mse(y_true, y_pred)
acc = accuracy_binary(y_true, y_pred)
activated = relu(z)
weights, biases = he_init([784, 128, 10])
```

## Documentation

- **[Full Documentation](https://ahmadrazacdx.github.io/neuroscope/)**: Complete API reference and guides
- **[Quickstart Guide](https://ahmadrazacdx.github.io/neuroscope/quickstart.html)**: Get up and running in minutes
- **[API Reference](https://ahmadrazacdx.github.io/neuroscope/reference.html)**: Detailed function and class documentation
- **[Examples](https://github.com/ahmadrazacdx/neuro-scope/tree/main/examples)**: Jupyter notebooks and scripts

## Use Cases

### Educational
- **Learning Deep Learning**: Understand neural network internals with detailed diagnostics
- **Research Projects**: Rapid prototyping with comprehensive analysis tools
- **Teaching**: Visual demonstrations of training dynamics and common issues

### Research & Development
- **Algorithm Development**: Test new optimization techniques and architectures
- **Hyperparameter Tuning**: Systematic analysis of training configurations
- **Debugging**: Identify and resolve training issues with diagnostic tools

### Production Prototyping
- **Proof of Concepts**: Quick validation of neural network approaches
- **Baseline Models**: Establish performance benchmarks for complex systems
- **Model Analysis**: Understand model behavior before scaling to larger frameworks

## Comparison with Other Frameworks

| Feature | NeuroScope | PyTorch | TensorFlow | Scikit-learn |
|---------|------------|---------|------------|--------------|
| **Learning Focus** | Educational | Production | Production | Traditional ML |
| **Built-in Diagnostics** | Comprehensive | Manual | Manual | Limited |
| **Visualization** | Publication-ready | Manual | Manual | Basic |
| **Ease of Use** | Intuitive | Complex | Complex | Simple |
| **MLP Focus** | Specialized | General | General | Limited |

## Contributing

We welcome contributions! Please see our [Contributing Guide](CONTRIBUTING.md) for details.

### Development Setup

```bash
# Clone the repository
git clone https://github.com/ahmadrazacdx/neuro-scope.git
cd neuro-scope

# Set up development environment
make dev

# Run tests
make test

# Build documentation
make docs
```

## License

Distributed under the terms of the [Apache 2.0 license][license],
NeuroScope is free and open source software.

## Issues & Support

If you encounter any problems:
- **[File an Issue](https://github.com/ahmadrazacdx/neuro-scope/issues)**: Bug reports and feature requests
- **[Discussions](https://github.com/ahmadrazacdx/neuro-scope/discussions)**: Questions and community support
- **[Documentation](https://ahmadrazacdx.github.io/neuroscope/)**: Comprehensive guides and API reference

## Acknowledgments

Built with modern Python best practices and inspired by the educational philosophy of making neural networks transparent and understandable.

---

**[Star us on GitHub](https://github.com/ahmadrazacdx/neuro-scope)** if you find NeuroScope useful!

<!-- github-only -->

[license]: https://github.com/ahmadrazacdx/neuro-scope/blob/main/LICENSE
[contributor guide]: https://github.com/ahmadrazacdx/neuro-scope/blob/main/CONTRIBUTING.md
