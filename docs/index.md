# NeuroScope Documentation

**A microscope for neural networks** - Comprehensive framework for building, training, and diagnosing multi-layer perceptrons with advanced monitoring and visualization capabilities.

NeuroScope provides a clean, education-oriented interface for building and analyzing multi-layer perceptrons with advanced diagnostic capabilities. Designed for rapid experimentation with comprehensive monitoring and visualization tools.

## Getting Started

New to neuroscope? Start here:

::::{grid} 2
:gutter: 3

:::{grid-item-card} Quickstart Guide
:link: quickstart
:link-type: doc

Get up and running in minutes with installation, basic usage, and your first neural network.
:::

:::{grid-item-card} Advanced Usage
:link: usage
:link-type: doc

Deep dive into advanced features, customization options, and comprehensive diagnosis book.
:::

:::{grid-item-card} Technical Deep Dive
:link: technical_deep_dive
:link-type: doc

Research-backed explanations of all diagnostic issues with mathematical foundations and literature citations.
:::

:::{grid-item-card} API Reference
:link: reference
:link-type: doc

Complete API documentation with detailed function and class references.
:::
::::

::::{grid} 2
:gutter: 3

:::{grid-item-card} Examples
:link: https://github.com/ahmadrazacdx/neuro-scope/tree/main/examples
:link-type: url

Jupyter notebooks and Python scripts demonstrating real-world applications.
:::

:::{grid-item-card} GitHub Repository
:link: https://github.com/ahmadrazacdx/neuro-scope
:link-type: url

Source code, issues, and contributions on GitHub.
:::
::::

## Key Features

### Modern MLP Implementation
- **Flexible Architecture**: Arbitrary layer sizes with customizable activations
- **Advanced Optimizers**: Adam, SGD with momentum, RMSprop, and adaptive learning rates
- **Smart Initialization**: He, Xavier, SELU, and intelligent auto-selection
- **Regularization**: L2 regularization, dropout with multiple variants

### Comprehensive Diagnostics
- **Pre-Training Analysis**: Architecture validation, weight initialization checks
- **Real-Time Monitoring**: Dead neuron detection, gradient flow analysis
- **Post-Training Evaluation**: Robustness testing, performance profiling
- **Research-Validated Metrics**: Based on established deep learning principles

### High Quality Visualization
- **Training Dynamics**: Learning curves, loss landscapes, convergence analysis
- **Network Internals**: Activation distributions, gradient flows, weight evolution
- **Diagnostic Plots**: Health indicators, training stability metrics
- **Interactive Animations**: Training progress visualization

## Learning Path

1. **Install neuroscope** → Start with installation and setup
2. **Quickstart Guide** → Learn the basics with your first neural network
3. **Build First Model** → Create and train a simple classifier
4. **Add Diagnostics** → Use pre-training analysis and monitoring
5. **Visualize Results** → Create publication-quality plots
6. **Advanced Usage** → Explore customization and optimization
7. **API Reference** → Deep dive into all available functions
8. **Examples Gallery** → Study real-world applications
9. **Contribute** → Help improve the framework

## Quick Example

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
history = model.fit(X_train, y_train, 
                   X_test=X_val, y_test=y_val,
                   epochs=100, 
                   monitor=monitor)

# Visualize results
viz = Visualizer(history)
viz.plot_learning_curves()
viz.plot_activation_hist()
```

## Community & Support

::::{grid} 3
:gutter: 3

:::{grid-item-card} Issues
:link: https://github.com/ahmadrazacdx/neuro-scope/issues
:link-type: url

Report bugs and request features
:::

:::{grid-item-card} Discussions
:link: https://github.com/ahmadrazacdx/neuro-scope/discussions
:link-type: url

Ask questions and get help
:::

:::{grid-item-card} Contributing
:link: contributing
:link-type: doc

{{ ... }}
Help improve NeuroScope
:::
::::

```{toctree}
:maxdepth: 2
:hidden:

quickstart
usage
technical_deep_dive
reference
contributing
Code of Conduct <codeofconduct>
License <license>
Changelog <https://github.com/ahmadrazacdx/neuro-scope/releases>
