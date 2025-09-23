# Advanced Usage Guide

Comprehensive guide to advanced NeuroScope features, customization options, and best practices for neural network development.

## Architecture Design

### Layer Configuration

```python
from neuroscope import MLP

# Basic architecture
model = MLP([784, 128, 64, 10])

# Advanced configuration
model = MLP(
    layer_dims=[784, 256, 128, 64, 10],
    hidden_activation="leaky_relu",
    out_activation="softmax",
    init_method="smart",           # Auto-selects best initialization
    dropout_rate=0.3,
    dropout_type="normal",         # default (alpha)for SELU networks
)

model.compile(
    optimizer='adam',
    lr=0.001,
    reg=None,                      # l2 for reg to work
    lamda=0.01,                    # regularization impact
    gradient_clip=None             # gradient clip norm
)
```

### Activation Function Selection

```python
# Available activation functions
activations = {
    "relu": "Standard ReLU - good default choice",
    "leaky_relu": "Leaky ReLU - prevents dead neurons",
    "selu": "SELU - self-normalizing, use with selu_init",
    "tanh": "Hyperbolic tangent - symmetric, zero-centered",
    "sigmoid": "Sigmoid - for binary classification output"
}

# Use SELU for deep networks
deep_model = MLP(
    [784, 512, 256, 128, 64, 10],
    hidden_activation="selu",
    init_method="selu_init",       # Specialized initialization for SELU
    dropout_type="alpha"           # Alpha dropout maintains self-normalization
)
```

## Training Configuration

### Optimizer Selection

```python
# Adam (default) - adaptive learning rate with momentum
model.compile(optimizer="adam", lr=1e-3)

# SGD - stochastic gradient descent
model.compile(optimizer="sgd", lr=1e-2)

# Note: NeuroScope currently supports "adam" and "sgd" optimizers
# Adam uses built-in beta1=0.9, beta2=0.999, eps=1e-8 parameters
```

### Regularization Options

```python
# L2 regularization
model.compile(optimizer="adam", lr=1e-3, reg="l2", lamda=0.01)

# Gradient clipping
model.compile(optimizer="adam", lr=1e-3, gradient_clip=5.0)

# Combined regularization
model.compile(
    optimizer="adam", 
    lr=1e-3, 
    reg="l2", 
    lamda=0.001,
    gradient_clip=1.0
)
```

### Dropout Configuration

```python
# Standard dropout during model creation
model = MLP(
    [784, 128, 64, 10],
    hidden_activation="relu",
    dropout_rate=0.3,           # 30% dropout
    dropout_type="normal"       # Standard dropout
)

# Alpha dropout for SELU networks
model = MLP(
    [784, 128, 64, 10],
    hidden_activation="selu",
    dropout_rate=0.1,
    dropout_type="alpha"        # Alpha dropout maintains normalization
)
```

## Advanced Diagnostics

### Pre-Training Analysis

```python
from neuroscope.diagnostics import PreTrainingAnalyzer

analyzer = PreTrainingAnalyzer(model)
# prints a beautiful summary report
analyzer.analyze(X_train, y_train)

```

### Post-training Evaluation

```python
from neuroscope.diagnostics import PostTrainingEvaluator

evaluator = PostTrainingEvaluator(model)

# Robustness testing
robustness = evaluator.evaluate_robustness(X_test, y_test, 
                                         noise_levels=[0.01, 0.05, 0.1, 0.2])
print(f"Overall robustness: {robustness['overall_robustness']:.3f}")
print(f"Status: {robustness['status']}")

# Performance evaluation
performance = evaluator.evaluate_performance(X_test, y_test)
print(f"Test accuracy: {performance['accuracy']:.3f}")
print(f"Samples per second: {performance['samples_per_second']:.0f}")
print(f"Total parameters: {performance['total_params']}")

# Stability evaluation
stability = evaluator.evaluate_stability(X_test, y_test, n_samples=100)
print(f"Prediction stability: {stability['overall_stability']:.3f}")

# Comprehensive evaluation report
evaluator.evaluate(X_test, y_test)
```

### Real-Time Training Monitoring

```python
from neuroscope.diagnostics import TrainingMonitor

# Configure monitoring
monitor = TrainingMonitor()

# Train with monitoring
history = model.fit(
    X_train, y_train,
    X_val=X_val, y_val=y_val,
    epochs=100,
    monitor=monitor,
    monitor_freq=1,
    early_stopping_patience=10
)

# Monitor provides real-time output during training:
# Epoch   1  Train loss: 0.652691, Train Accuracy: 0.6125 Val loss: 0.6239471, Val Accuracy: 0.64000
# ----------------------------------------------------------------------------------------------------
# SNR: 🟡 (0.70),     | Dead Neurons: 🟢 (0.00%)  | VGP:      🟢  | EGP:     🟢  | Weight Health: 🟢
# WUR: 🟢 (1.30e-03)  | Saturation:   🟢 (0.00)   | Progress: 🟢  | Plateau: 🟢  | Overfitting:   🟡
# ----------------------------------------------------------------------------------------------------


# Ultra-fast training - 10-100x speedup!
history = model.fit_fast(
    X_train, y_train, X_val, y_val,
    epochs=100, 
    batch_size=256,
    eval_freq=5 
)

```


## Advanced Visualization

### Comprehensive Training Analysis

```python
from neuroscope.viz import Visualizer

viz = Visualizer(history)

# Learning curves with confidence intervals
viz.plot_learning_curves(
    figsize=(9, 4),
    ci=True,                      # Confidence intervals
    markers=True,                 # Show epoch markers
    save_path="learning_curves.png"
)

# Dynamic metric labeling examples:
# For R² regression: Shows "R²" instead of "Accuracy"
# For F1 classification: Shows "F1" instead of "Accuracy" 
# For precision: Shows "Precision" instead of "Accuracy"
# The system automatically detects your training metric and updates all labels

# Network internals evolution over epochs
viz.plot_activation_stats(
    figsize=(12, 4),
    reference_lines=True          # Show healthy ranges
)

viz.plot_gradient_stats(
    figsize=(12, 4),
    reference_lines=True
)

viz.plot_weight_stats(
    figsize=(12, 4),
    reference_lines=True
)

# Distribution plots for specific epochs
viz.plot_activation_hist(epoch=25, figsize=(9, 4), kde=True)
viz.plot_gradient_hist(epoch=25, figsize=(9, 4), kde=True)
viz.plot_weight_hist(epoch=25, figsize=(9, 4), kde=True)

# Gradient norms and weight update ratios
viz.plot_gradient_norms(figsize=(12, 4), reference_lines=True)
viz.plot_update_ratios(figsize=(12, 4), reference_lines=True)
```

### Training Animation with Dynamic Labels

```python
# Create animated training visualization with correct metric labels
viz.plot_training_animation(bg="dark", save_path="training_animation.gif")

# The animation automatically shows:
# - "Accuracy Evolution" for accuracy metrics
# - "R² Evolution" for R² regression
# - "F1 Evolution" for F1 score
# - "Precision Evolution" for precision
# - And updates all bar chart labels accordingly
```

## Best Practices

### Model Architecture Guidelines

```python
# For deep networks (>4 layers)
deep_model = MLP(
    [784, 512, 256, 128, 64, 32, 10],
    hidden_activation="selu",      # Self-normalizing
    init_method="selu_init",       # Proper initialization
    dropout_type="alpha",          # Maintains normalization
    dropout_rate=0.1
)
deep_model.compile(optimizer="adam", lr=1e-3, reg="l2", lamda=1e-4)

# For wide networks
wide_model = MLP(
    [784, 1024, 1024, 10],
    hidden_activation="relu",
    init_method="he",
    dropout_rate=0.5              # Higher dropout for wide networks
)
wide_model.compile(optimizer="adam", lr=1e-3, reg="l2", lamda=1e-5)

# For small datasets
small_data_model = MLP(
    [784, 64, 32, 10],
    hidden_activation="tanh",      # Less prone to overfitting
    dropout_rate=0.2
)
small_data_model.compile(optimizer="adam", lr=1e-3, reg="l2", lamda=1e-3)
```

## Troubleshooting

### Common Issues and Solutions

| Issue | Symptoms | Solutions |
|-------|----------|----------|
| **Vanishing Gradients** | Loss plateaus early, poor learning | Use ReLU/LeakyReLU, He initialization, lower learning rate |
| **Exploding Gradients** | Loss becomes NaN, unstable training | Gradient clipping, lower learning rate |
| **Dead Neurons** | Many zero activations | LeakyReLU, better initialization, check learning rate |
| **Overfitting** | Training accuracy >> validation accuracy | Dropout, L1/L2 regularization, more data |
| **Underfitting** | Both accuracies low | Larger network, lower regularization, higher learning rate |


## Diagnosis Book

### Quick Problem Solver

This comprehensive reference helps you quickly identify and solve common neural network training issues using NeuroScope's diagnostic tools.

#### **Dead Neurons (ReLU Death)**

**Symptoms:**
- 🔴 Dead Neurons > 30% in TrainingMonitor
- Activations stuck at zero
- Poor learning performance

**Causes:**
- Large learning rates
- Poor weight initialization
- Saturated ReLU activations

**Solutions:**
```python
# Use Leaky ReLU instead of ReLU
model = MLP([784, 128, 10], hidden_activation="leaky_relu")

# Better initialization
model = MLP([784, 128, 10], init_method="he")

# Lower learning rate
model.compile(optimizer="adam", lr=1e-4)

# Gradient clipping
model.compile(optimizer="adam", lr=1e-3, gradient_clip=1.0)
```

**NeuroScope Tools:**
- `TrainingMonitor`: Real-time dead neuron detection
- `PreTrainingAnalyzer`: Weight initialization validation
- `Visualizer.plot_activation_hist()`: Activation distribution analysis

---

#### **Vanishing Gradient Problem (VGP)**

**Symptoms:**
- 🔴 VGP status in TrainingMonitor
- Gradient Signal-to-Noise Ratio < 0.4
- Early layers learn slowly

**Causes:**
- Deep networks with sigmoid/tanh
- Poor weight initialization
- Inappropriate activation functions

**Solutions:**
```python
# Use ReLU-based activations
model = MLP([784, 256, 128, 64, 10], hidden_activation="relu")

# He initialization for ReLU
model = MLP([784, 256, 128, 64, 10], init_method="he")

# For very deep networks, use SELU
model = MLP([784, 512, 256, 128, 64, 32, 10], 
           hidden_activation="selu", init_method="selu_init")
```

**NeuroScope Tools:**
- `TrainingMonitor`: Real-time VGP detection
- `Visualizer.plot_gradient_stats()`: Gradient magnitude tracking
- `Visualizer.plot_gradient_hist()`: Gradient distribution analysis

---

#### **Exploding Gradient Problem (EGP)**

**Symptoms:**
- 🔴 EGP status in TrainingMonitor
- Loss becomes NaN or very large
- Unstable training dynamics

**Causes:**
- High learning rates
- Deep networks without normalization
- Poor weight initialization

**Solutions:**
```python
# Gradient clipping (most effective)
model.compile(optimizer="adam", lr=1e-3, gradient_clip=1.0)

# Lower learning rate
model.compile(optimizer="adam", lr=1e-4)

# Better initialization
model = MLP([784, 128, 10], init_method="xavier")
```

**NeuroScope Tools:**
- `TrainingMonitor`: Real-time EGP detection
- `Visualizer.plot_gradient_norms()`: Gradient norm monitoring

---

#### **Activation Saturation**

**Symptoms:**
- 🔴 Saturation > 25% in TrainingMonitor
- Sigmoid/Tanh outputs near 0 or 1
- Poor gradient flow

**Causes:**
- Sigmoid/Tanh with large inputs
- Poor weight initialization
- High learning rates

**Solutions:**
```python
# Use ReLU-based activations
model = MLP([784, 128, 10], hidden_activation="relu")

# For sigmoid/tanh, use proper initialization
model = MLP([784, 128, 10], hidden_activation="tanh", init_method="xavier")

# Lower learning rate
model.compile(optimizer="adam", lr=1e-4)
```

**NeuroScope Tools:**
- `TrainingMonitor`: Real-time saturation monitoring
- `Visualizer.plot_activation_hist()`: Activation distribution analysis

---

#### **Poor Weight Update Ratios**

**Symptoms:**
- 🔴 WUR outside 1e-4 to 1e-2 range
- Very slow or unstable learning

**Causes:**
- Inappropriate learning rate
- Poor weight initialization
- Gradient scaling issues

**Solutions:**
```python
# Adjust learning rate based on WUR
# If WUR < 1e-4: increase learning rate
model.compile(optimizer="adam", lr=1e-2)

# If WUR > 1e-2: decrease learning rate
model.compile(optimizer="adam", lr=1e-4)

# Use adaptive learning rate
model.compile(optimizer="adam", lr=1e-3)  # Adam adapts automatically
```

**NeuroScope Tools:**
- `TrainingMonitor`: Real-time WUR monitoring
- `Visualizer.plot_update_ratios()`: Update ratio tracking

---

#### **Training Plateau**

**Symptoms:**
- 🔴 Plateau status in TrainingMonitor
- Loss stops decreasing
- No learning progress

**Causes:**
- Learning rate too low
- Local minima
- Insufficient model capacity

**Solutions:**
```python
# Increase learning rate
model.compile(optimizer="adam", lr=1e-2)

# Add regularization to escape local minima
model.compile(optimizer="adam", lr=1e-3, reg="l2", lamda=1e-4)

# Increase model capacity
model = MLP([784, 256, 128, 10])  # Larger hidden layers
```

**NeuroScope Tools:**
- `TrainingMonitor`: Real-time plateau detection
- `Visualizer.plot_learning_curves()`: Loss progression analysis

---

#### **Overfitting**

**Symptoms:**
- 🔴 Overfitting status in TrainingMonitor
- Validation loss increases while training loss decreases
- Large gap between train/validation accuracy

**Causes:**
- Model too complex
- Insufficient regularization
- Too many training epochs

**Solutions:**
```python
# Add L2 regularization
model.compile(optimizer="adam", lr=1e-3, reg="l2", lamda=1e-3)

# Add dropout
model = MLP([784, 128, 10], dropout_rate=0.3)

# Early stopping
history = model.fit(X_train, y_train, X_val=X_val, y_val=y_val,
                   early_stopping_patience=10)

# Reduce model complexity
model = MLP([784, 64, 10])  # Smaller hidden layers
```

**NeuroScope Tools:**
- `TrainingMonitor`: Real-time overfitting detection
- `PostTrainingEvaluator`: Comprehensive overfitting analysis
- `Visualizer.plot_learning_curves()`: Train/validation gap visualization

---

#### **Poor Initial Loss**

**Symptoms:**
- PreTrainingAnalyzer shows FAIL status
- Initial loss much higher than expected
- Training starts poorly

**Causes:**
- Poor weight initialization
- Data preprocessing issues
- Wrong output activation

**Solutions:**
```python
# Check and fix weight initialization
analyzer = PreTrainingAnalyzer(model)
analyzer.analyze()
model = MLP([784, 128, 10], init_method="smart")

# Verify output activation
# For binary classification:
model = MLP([784, 128, 1], out_activation="sigmoid")
# For multiclass:
model = MLP([784, 128, 10], out_activation="softmax")
# For regression:
model = MLP([784, 128, 1], out_activation=None)
```

**NeuroScope Tools:**
- `PreTrainingAnalyzer`: Comprehensive pre-training validation
- `PreTrainingAnalyzer.analyze()`:  Pre training analysis

---

#### **Poor Model Robustness**

**Symptoms:**
- PostTrainingEvaluator shows poor robustness
- Model fails on noisy inputs
- Unstable predictions

**Causes:**
- Overfitting to training data
- Insufficient regularization
- Poor generalization

**Solutions:**
```python
# Add noise-based regularization
model = MLP([784, 128, 10], dropout_rate=0.2)

# Increase regularization
model.compile(optimizer="adam", lr=1e-3, reg="l2", lamda=1e-3)

# Data augmentation (add noise during training)
# Note: Implement in your training loop
```

**NeuroScope Tools:**
- `PostTrainingEvaluator.evaluate_robustness()`: Noise robustness testing
- `PostTrainingEvaluator.evaluate_stability()`: Prediction stability analysis

---

### Quick Reference Table

| Problem | TrainingMonitor Status | Primary Solution | NeuroScope Tool |
|---------|----------------------|------------------|-----------------|
| Dead Neurons | 🔴 Dead Neurons > 30% | Use LeakyReLU, He init | `plot_activation_hist()` |
| Vanishing Gradients | 🔴 VGP, SNR < 0.4 | ReLU + He init | `plot_gradient_stats()` |
| Exploding Gradients | 🔴 EGP | Gradient clipping | `plot_gradient_norms()` |
| Saturation | 🔴 Saturation > 25% | Avoid sigmoid/tanh | `plot_activation_hist()` |
| Poor WUR | 🔴 WUR outside range | Adjust learning rate | `plot_update_ratios()` |
| Plateau | 🔴 Plateau | Increase LR or capacity | `plot_learning_curves()` |
| Overfitting | 🔴 Overfitting | Add regularization | `evaluate_robustness()` |

## Next Steps

1. **[Technical Deep Dive](technical_deep_dive.md)**: Research-backed explanations with mathematical foundations and literature citations
2. **[API Reference](reference.md)**: Explore all available functions and classes
3. **[Examples](https://github.com/ahmadrazacdx/neuro-scope/tree/main/examples)**: Jupyter notebooks with real-world examples
4. **[Quickstart Guide](quickstart.md)**: Quick introduction to NeuroScope
5. **[Contributing](https://github.com/ahmadrazacdx/neuro-scope/blob/main/CONTRIBUTING.md)**: Help improve NeuroScope
