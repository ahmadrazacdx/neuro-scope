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
    layer_dims=[784, 128, 64, 10],           # Input -> Hidden -> Hidden ->Output
    hidden_activation="relu",                 # ReLU for hidden layers
    out_activation="softmax",                 # Softmax for classification
    init_method="he",                        # He initialization for ReLU
    dropout_rate=0.2                         # 20% dropout fo regularization
)

# Step 2: Compile the model with optimizer selection
model.compile(
    optimizer="adam",                        # Choose: "adam", "sgd", "sgdm", "sgdnm", "rmsprop"
    lr=1e-3,                                # Learning rate
    reg="l2",                               # L2 regularization (use "l2" or None)
    lamda=1e-3                              # Regularization strength
)

# Optimizer Quick Reference:
# - "adam"   : Best default choice for most tasks (recommended)
# - "sgd"    : Simple SGD, good for simple problems
# - "sgdm"   : SGD with momentum, faster convergence
# - "sgdnm"  : SGD with Nesterov momentum, best momentum variant
# - "rmsprop": Adaptive learning, good for RNNs and non-stationary objectives

# Step 3: Pre-training analysis (optional but recommended)
analyzer = PreTrainingAnalyzer(model)
analyzer.analyze(X_train, y_train)

# Step 4: 
# Train with monitoring
monitor = TrainingMonitor()
history = model.fit(X_train, y_train, 
                   X_val=X_val, y_val=y_val,
                   epochs=100, 
                   monitor=monitor,
                   verbose=True
)

# Fast training - ~5-10× speedup! without monitors
history = model.fit_fast(
    X_train, y_train, 
    X_val=X_val, y_val=y_val,
    epochs=100, 
    batch_size=256,
    eval_freq=5                        
)

# Step 5: Post-training evaluation
evaluator = PostTrainingEvaluator(model)
analyzer.evaluate(X_train, y_train)

# Step 6: Visualize results
viz = Visualizer(history)
viz.plot_learning_curves()
viz.plot_activation_hist()

# Step 7: Save your trained model
model.save("my_trained_model.ns")              # Basic save (weights + architecture)
model.save("with_optimizer.ns", save_optimizer=True)  # Save with optimizer state

# Step 8: Load model later
loaded_model = MLP.load("my_trained_model.ns")
predictions = loaded_model.predict(X_test)

# Load with optimizer state to continue training
model_resume = MLP.load("with_optimizer.ns", load_optimizer=True)
history_continued = model_resume.fit(X_train, y_train, epochs=50)  # Seamless continuation
```

## Saving and Loading Models

### Save Trained Models

```python
# After training your model
history = model.fit(X_train, y_train, epochs=100)

# Save just weights and architecture (for inference)
model.save("trained_model.ns")

# Save with optimizer state (for resuming training)
model.save("checkpoint.ns", save_optimizer=True)
```

### Load Models

```python
# Load for inference
model = MLP.load("trained_model.ns")
predictions = model.predict(X_new)

# Load and continue training
model = MLP.load("checkpoint.ns", load_optimizer=True)
# Model is already compiled with saved optimizer!
history = model.fit(X_train, y_train, epochs=50)  # Training continues smoothly
```

### Benefits of Optimizer State Persistence

- **Seamless Training Continuation**: No loss spike when resuming
- **Preserves Momentum**: Velocity buffers and adaptive rates maintained
- **Exact State Recovery**: Training continues as if never interrupted
- **Example**: Loss at epoch 100: 0.234 → Load → Loss at epoch 101: 0.230 (smooth!)

## Choosing the Right Optimizer

### Quick Decision Guide

```python
# General-purpose deep learning (RECOMMENDED)
model.compile(optimizer="adam", lr=1e-3)

# Faster convergence on smooth objectives
model.compile(optimizer="sgdnm", lr=1e-2)  # Nesterov momentum

# Adaptive learning for non-stationary problems
model.compile(optimizer="rmsprop", lr=1e-3)

# Simple problems or full control needed
model.compile(optimizer="sgd", lr=1e-2)

# Faster than SGD, smoother convergence
model.compile(optimizer="sgdm", lr=1e-2)  # Standard momentum
```

### Optimizer Comparison

| Optimizer | Best For | Learning Rate | Convergence Speed |
|-----------|----------|---------------|-------------------|
| `adam` | Most tasks (default) | 1e-3 to 1e-4 | Fast |
| `sgdnm` | Non-convex optimization | 1e-2 to 1e-3 | Very Fast |
| `rmsprop` | RNNs, non-stationary | 1e-3 to 1e-4 | Fast |
| `sgdm` | General acceleration | 1e-2 to 1e-3 | Medium |
| `sgd` | Simple problems | 1e-2 to 1e-1 | Slow |

## Understanding the Output

### Compilation Summary
```
===============================================================
                    MLP ARCHITECTURE SUMMARY
===============================================================
                    MLP ARCHITECTURE SUMMARY
===============================================================
Layer        Type               Output Shape    Params    
---------------------------------------------------------------
Layer 1      Input → Hidden     (128,)          100480    
Layer 2      Hidden → Hidden    (64,)           8256      
Layer 3      Hidden → Output    (10,)           650       
---------------------------------------------------------------
TOTAL                                           109386    
===============================================================
Hidden Activation                               relu
Output Activation                               softmax
Optimizer                                       Adam
Learning Rate                                   0.001
Dropout                                         20.0% (normal)
L2 Regularization                               λ = 0.001
===============================================================
```
### Pre-training Analysis Results
```
==========================================================================================
                         NEUROSCOPE PRE-TRAINING ANALYSIS
==========================================================================================
DIAGNOSTIC TOOL             STATUS       RESULT         NOTE                                      
------------------------------------------------------------------------------------------
Initial Loss Check          PASS         0.8107         Perfect loss init                         
Initialization Validation   PASS         3 layers       Good weight init                          
Layer Capacity Analysis     PASS         861 params     No bottlenecks                            
Architecture Sanity Check   PASS         0I/0W          Architecture is fine                      
Capacity vs Data Ratio      PASS         861 params     Excellent model size                      
Convergence Feasibility     EXCELLENT    100.0%         Excellent convergence setup               
------------------------------------------------------------------------------------------
OVERALL STATUS: ALL SYSTEMS READY
TESTS PASSED: 6/6
==========================================================================================
```

### Training Progress
```
Epoch   1  Train loss: 0.652691, Train Accuracy: 0.6125 Val loss: 0.6239471, Val Accuracy: 0.64000
Epoch   2  Train loss: 0.555231, Train Accuracy: 0.7325 Val loss: 0.5320609, Val Accuracy: 0.73000
Epoch   3  Train loss: 0.483989, Train Accuracy: 0.8100 Val loss: 0.4652307, Val Accuracy: 0.80250
Epoch   4  Train loss: 0.423608, Train Accuracy: 0.8400 Val loss: 0.4062951, Val Accuracy: 0.84500
Epoch   5  Train loss: 0.377493, Train Accuracy: 0.8581 Val loss: 0.3613807, Val Accuracy: 0.86500
```
### Training with Monitors
```
Epoch   1  Train loss: 0.652691, Train Accuracy: 0.6125 Val loss: 0.6239471, Val Accuracy: 0.64000
Epoch   2  Train loss: 0.555231, Train Accuracy: 0.7325 Val loss: 0.5320609, Val Accuracy: 0.73000
Epoch   3  Train loss: 0.483989, Train Accuracy: 0.8100 Val loss: 0.4652307, Val Accuracy: 0.80250
Epoch   4  Train loss: 0.423608, Train Accuracy: 0.8400 Val loss: 0.4062951, Val Accuracy: 0.84500
----------------------------------------------------------------------------------------------------
SNR: 🟡 (0.70),     | Dead Neurons: 🟢 (0.00%)  | VGP:      🟢  | EGP:     🟢  | Weight Health: 🟢
WUR: 🟢 (1.30e-03)  | Saturation:   🟢 (0.00)   | Progress: 🟢  | Plateau: 🟢  | Overfitting:   🟡
----------------------------------------------------------------------------------------------------
```
### Post Training Evaluation
```
================================================================================
                  NEUROSCOPE POST-TRAINING EVALUATION
================================================================================
EVALUATION      STATUS       SCORE        NOTE                                         
--------------------------------------------------------------------------------
Robustness      EXCELLENT    0.993        Highly robust to noise                       
Performance     EXCELLENT    0.907        High accuracy and fast inference             
Stability       EXCELLENT    0.800        Highly stable predictions                    
--------------------------------------------------------------------------------
OVERALL STATUS: EVALUATION COMPLETE
EVALUATIONS PASSED: 3/3
================================================================================
                     CLASSIFICATION METRICS
================================================================================
METRIC               STATUS       SCORE        NOTE                                    
--------------------------------------------------------------------------------
Accuracy             PASS         0.9075       Good performance                        
Precision            PASS         0.9084       Good performance                        
Recall               PASS         0.9075       Good performance                        
F1-Score             PASS         0.9075       Good performance                        
--------------------------------------------------------------------------------
METRICS STATUS: METRICS EVALUATION COMPLETE
METRICS PASSED: 4/4
================================================================================
```

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
### 3. Multi Class Classification

```python
# Binary classification setup
model = MLP([784, 64, 32, 10], 
           hidden_activation="relu", 
           out_activation="softmax")
model.compile(optimizer="adam", lr=1e-3)

# For binary targets (0/1)
y_binary = np.random.randint(0, 2, (1000, 1))
history = model.fit(X_train, y_binary, epochs=30)
```
### 3. Regression

```python
# Regression setup
model = MLP([784, 128, 64, 1], 
           hidden_activation="relu", 
           out_activation=None)  # No output activation (linear) for regression
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
accuracy = accuracy_binary(y_true, predictions, thresh=0.5)

# Initialize weights manually
weights, biases = he_init([784, 128, 10])
```

### Regularization Options

```python
# L2 regularization
model.compile(optimizer="adam", lr=1e-3, reg="l2", lamda=1e-4)

# Dropout (Inverted-default)
model = MLP([784, 128, 10], 
           hidden_activation="relu",
           dropout_rate=0.3)      # 30% normal dropout

# Dropout (alpha- if hidden_activation is Selu)
model = MLP([784, 128, 10], 
           hidden_activation="relu",
           dropout_type='alpha'
           dropout_rate=0.3)      # 30% alpha dropout

# Gradient clipping
model.compile(optimizer="adam", lr=1e-3, gradient_clip=5.0)
```

### Optimizer Options

```python
# Adam optimizer (default)
model.compile(optimizer="adam", lr=1e-3)

# SGD optimizer
model.compile(optimizer="sgd", lr=1e-2)

# SGD with Momentum (faster convergence)
model.compile(optimizer="sgdm", lr=1e-2)

# SGD with Nesterov Momentum (even better convergence)
model.compile(optimizer="sgdnm", lr=1e-2)

# RMSprop (great for RNNs and non-stationary objectives)
model.compile(optimizer="rmsprop", lr=1e-3)

# Note: NeuroScope supports 5 optimizers: sgd, sgdm, sgdnm, rmsprop, adam
```

## Diagnostic Tools

### Monitor Training Health

```python
# Real-time monitoring
monitor = TrainingMonitor(history_size=50)
history = model.fit(X_train, y_train, X_val=X_val, y_val=y_val, 
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

### Learning Curves

```python
viz = Visualizer(history)

# Basic learning curves
viz.plot_learning_curves()

# Advanced learning curves with confidence intervals 
# Only availabe for use with fit()
viz.plot_learning_curves(figsize=(9, 4), ci=True, markers=True)

# for fit_fast()
viz.plot_curves_fast()
# The plot titles and labels automatically sync with your training metric:
# - metric="accuracy" → shows "Accuracy" 
# - metric="r2" → shows "R²"
# - metric="f1" → shows "F1"
# - metric="precision" → shows "Precision"
# - And more...
```

### Network Internals

```python
# Activation distributions
viz.plot_activation_hist(epoch=25, kde=True)

# Gradient flow analysis
viz.plot_gradient_hist(epoch=25, last_layer=False)

# Weight histogram at epoch=25 (default last)
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

---

Ready to dive deeper? Check out our [comprehensive examples](https://github.com/ahmadrazacdx/neuro-scope/tree/main/examples) or explore the [full API reference](reference.md)!
