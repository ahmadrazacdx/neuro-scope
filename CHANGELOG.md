# Changelog

All notable changes to NeuroScope will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/), and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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
- **Fast Training** (`fit_fast()`) - Optimized for fast training with 10-80x speedup
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
- **Training Speed** - Up to 80x faster with `fit_fast()` mode
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
