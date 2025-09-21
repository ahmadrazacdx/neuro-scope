# Quickstart Guide

Get up and running with NeuroScope in minutes! This guide covers installation, basic usage, and common patterns to help you start building and analyzing neural networks immediately.

## Installation

### Prerequisites

- **Python 3.11+** (Python 3.12 recommended)
- **pip** or **conda** package manager

### Install from PyPI (Recommended)

```bash
pip install neuroscope
```

### Install from Source (Development)

```bash
git clone https://github.com/ahmadrazacdx/neuro-scope.git
cd neuro-scope
pip install -e .
```

### Verify Installation

```python
import neuroscope
print(f"NeuroScope version: {neuroscope.__version__}")
```

## Your First Neural Network

Let's build a simple classifier for the classic MNIST-like problem:

```python
import numpy as np
from neuroscope import MLP, PreTrainingAnalyzer, TrainingMonitor, Visualizer

# Generate sample data (replace with your dataset)
np.random.seed(42)
X_train = np.random.randn(1000, 784)  # 1000 samples, 784 features
y_train = np.random.randint(0, 10, (1000, 10))  # One-hot encoded labels
X_val = np.random.randn(200, 784)
y_val = np.random.randint(0, 10, (200, 10))

# Step 1: Create and configure the model
model = MLP(
    layer_dims=[784, 128, 64, 10],           # Input -> Hidden -> Hidden -> Output
    hidden_activation="relu",                 # ReLU for hidden layers
    out_activation="softmax",                 # Softmax for classification
    init_method="he",                        # He initialization for ReLU
    dropout_rate=0.2                         # 20% dropout for regularization
)

# Step 2: Compile the model
model.compile(
    optimizer="adam",                        # Adam optimizer
    lr=1e-3,                                # Learning rate
    loss="cce",                             # Categorical crossentropy
    metric="accuracy"                     # Track accuracy
)

# Step 3: Pre-training analysis (optional but recommended)
analyzer = PreTrainingAnalyzer(model)
analyzer.analyze(X_train, y_train)

# Step 4: 
# Train with monitoring
monitor = TrainingMonitor()
history = model.fit(X_train, y_train, 
                   X_test=X_val, y_test=y_val,
                   epochs=100, 
                   monitor=monitor,
                   verbose=True
)

# Fast training - 10-100x speedup! without monitors
history = model.fit_fast(
    X_train, y_train, X_val, y_val,
    epochs=100, 
    batch_size=256,
    eval_freq=5 
)

# Step 5: Visualize results
viz = Visualizer(history)
viz.plot_learning_curves()
viz.plot_activation_hist(epoch=25) # see 25th epoch
```

## Understanding the Output

### Training Progress
```
Epoch 1/50: loss=2.3456, accuracy=0.1234, val_loss=2.4567, val_accuracy=0.1345
Epoch 2/50: loss=2.1234, accuracy=0.2345, val_loss=2.2345, val_accuracy=0.2456
...
```

### Pre-training Analysis Results
- **Initial Loss**: Should be close to theoretical baseline (e.g., ~2.30 for 10-class classification)
- **Weight Init Quality**: "Good", "Acceptable", or "Poor" based on distribution analysis
- **Architecture Warnings**: Potential issues with layer sizes or activation choices

## Common Patterns

### 1. Binary Classification

```python
# Binary classification setup
model = MLP([784, 64, 32, 1], 
           hidden_activation="relu", 
           out_activation="sigmoid")
model.compile(optimizer="adam", lr=1e-3)

# For binary targets (0/1)
y_binary = np.random.randint(0, 2, (1000, 1))
history = model.fit(X_train, y_binary, epochs=30)
```

### 2. Regression

```python
# Regression setup
model = MLP([784, 128, 64, 1], 
           hidden_activation="relu", 
           out_activation=None)  # No output activation for regression
model.compile(optimizer="adam", lr=1e-3)

# For continuous targets
y_regression = np.random.randn(1000, 1)
history = model.fit(X_train, y_regression, epochs=50)
```

### 3. Direct Function Usage

```python
from neuroscope import mse, accuracy_binary, relu, he_init

# Use functions directly without classes
predictions = model.predict(X_test)
loss = mse(y_true, predictions)
accuracy = accuracy_binary(y_true, predictions > 0.5)

# Initialize weights manually
weights, biases = he_init([784, 128, 10])
```

## Hyperparameter Tuning

### Regularization Options

```python
# L2 regularization
model.compile(optimizer="adam", lr=1e-3, reg="l2", lamda=1e-4)

# Dropout (configured during model creation)
model = MLP([784, 128, 10], 
           hidden_activation="relu",
           dropout_rate=0.3)      # 30% dropout

# Gradient clipping
model.compile(optimizer="adam", lr=1e-3, gradient_clip=5.0)
```

### Optimizer Options

```python
# Adam optimizer (default)
model.compile(optimizer="adam", lr=1e-3)

# SGD optimizer
model.compile(optimizer="sgd", lr=1e-2)

# Note: NeuroScope currently supports "adam" and "sgd" optimizers
```

## Diagnostic Tools

### Monitor Training Health

```python
# Real-time monitoring
monitor = TrainingMonitor(history_size=50)
history = model.fit(X_train, y_train, X_test=X_val, y_test=y_val, 
                   monitor=monitor, epochs=100)

# TrainingMonitor provides real-time emoji-based status during training:
# Epoch   5  Train loss: 0.377493, Train Accuracy: 0.8581 Val loss: 0.3613807, Val Accuracy: 0.86500
# ----------------------------------------------------------------------------------------------------
# SNR: 🟡 (0.58),     | Dead Neurons: 🟢 (0.00%)  | VGP:      🟢  | EGP:     🟢  | Weight Health: 🟢
# WUR: 🟡 (7.74e-04)  | Saturation:   🟢 (0.00)   | Progress: 🟢  | Plateau: 🟢  | Overfitting:   🟢
# ----------------------------------------------------------------------------------------------------
```

### Post-training Evaluation

```python
from neuroscope.diagnostics import PostTrainingEvaluator

evaluator = PostTrainingEvaluator(model)

# Comprehensive evaluation with detailed report
evaluator.evaluate(X_test, y_test)

# Individual evaluations
robustness = evaluator.evaluate_robustness(X_test, y_test)
performance = evaluator.evaluate_performance(X_test, y_test)
stability = evaluator.evaluate_stability(X_test, y_test)

print(f"Robustness: {robustness['overall_robustness']:.3f}")
print(f"Performance: {performance['accuracy']:.3f}")
print(f"Stability: {stability['overall_stability']:.3f}")
```

## Visualization Gallery

### Learning Curves

```python
viz = Visualizer(history)

# Basic learning curves
viz.plot_learning_curves()

# Advanced learning curves with confidence intervals
viz.plot_learning_curves(figsize=(9, 4), ci=True, markers=True)
```

### Network Internals

```python
# Activation distributions
viz.plot_activation_hist(epoch=25, kde=True)

# Gradient flow analysis
viz.plot_gradient_hist(epoch=25, last_layer=False)

# Weight evolution
viz.plot_weight_hist(epoch=25)
```

### Training Animation

```python
# Create animated training visualization
viz.plot_training_animation(bg="dark", save_path="training_animation.gif")
```

## Troubleshooting

### Common Issues and Solutions

| Issue | Symptoms | Solutions |
|-------|----------|-----------|
| **Vanishing Gradients** | Loss plateaus early, poor learning | Use ReLU/LeakyReLU, He initialization, lower learning rate |
| **Exploding Gradients** | Loss becomes NaN, unstable training | Gradient clipping, lower learning rate, batch normalization |
| **Dead Neurons** | Many zero activations | LeakyReLU, better initialization, check learning rate |
| **Overfitting** | Training accuracy >> validation accuracy | Dropout, L1/L2 regularization, more data, early stopping |
| **Underfitting** | Both accuracies low | Larger network, lower regularization, higher learning rate |

### Debug Mode

```python
# Enable detailed logging
import logging
logging.basicConfig(level=logging.DEBUG)

# Verbose training
history = model.fit(X_train, y_train, epochs=10, verbose=True)
```

## Next Steps

1. **[Advanced Usage](usage.md)**: Deep dive into advanced features and comprehensive diagnosis book
2. **[Technical Deep Dive](technical_deep_dive.md)**: Research-backed explanations with mathematical foundations
3. **[API Reference](reference.md)**: Explore all available functions and classes
4. **[Examples](https://github.com/ahmadrazacdx/neuro-scope/tree/main/examples)**: Jupyter notebooks with real-world examples
5. **[Contributing](https://github.com/ahmadrazacdx/neuro-scope/blob/main/CONTRIBUTING.md)**: Help improve NeuroScope

## Pro Tips

- **Start Simple**: Begin with basic architectures and gradually add complexity
- **Monitor Early**: Use pre-training analysis to catch issues before training
- **Visualize Often**: Regular visualization helps understand training dynamics
- **Experiment Systematically**: Change one hyperparameter at a time
- **Save Your Work**: Store model weights and training history for important experiments

---

Ready to dive deeper? Check out our [comprehensive examples](https://github.com/ahmadrazacdx/neuro-scope/tree/main/examples) or explore the [full API reference](reference.md)!
