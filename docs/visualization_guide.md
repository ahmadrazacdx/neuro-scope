# NeuroScope Visualization Guide

Complete guide to NeuroScope's dynamic visualization system with automatic metric detection and professional plotting capabilities.

## Overview

NeuroScope's visualization system automatically detects and displays the correct metric labels based on your training configuration. No more hardcoded "Accuracy" labels - the plots dynamically adapt to show R², F1, Precision, or any other metric you're using.

## Dynamic Metric Detection

### Supported Metrics and Display Names

| Training Metric | Display Name | Use Case |
|----------------|--------------|----------|
| `"accuracy"` | Accuracy | Classification |
| `"r2"` | R² | Regression |
| `"f1"` | F1 | Classification |
| `"precision"` | Precision | Classification |
| `"recall"` | Recall | Classification |
| `"mse"` | MSE | Regression |
| `"rmse"` | RMSE | Regression |
| `"mae"` | MAE | Regression |
| `"smart"` | Context-aware | Auto-detection |

### Smart Metric Detection

The `"smart"` metric automatically selects the appropriate display name based on your model configuration:

```python
# Regression (no output activation)
model = MLP([784, 128, 1], out_activation=None)
history = model.fit(X_train, y_train, X_val, y_val, metric="smart")
# Displays: "MSE"

# Binary classification (sigmoid)
model = MLP([784, 128, 1], out_activation="sigmoid") 
history = model.fit(X_train, y_train, X_val, y_val, metric="smart")
# Displays: "Accuracy"

# Multi-class classification (softmax)
model = MLP([784, 128, 10], out_activation="softmax")
history = model.fit(X_train, y_train, X_val, y_val, metric="smart")
# Displays: "Accuracy"
```

## Learning Curves with Dynamic Labels

### Basic Usage

```python
from neuroscope import MLP, Visualizer

# Train with different metrics
model = MLP([784, 128, 10])
model.compile(optimizer="adam", lr=1e-3)

# R² Regression
history_r2 = model.fit(X_train, y_train, X_val, y_val, metric="r2")
viz_r2 = Visualizer(history_r2)
viz_r2.plot_learning_curves()  # Shows "R²" in title and labels

# F1 Classification  
history_f1 = model.fit(X_train, y_train, X_val, y_val, metric="f1")
viz_f1 = Visualizer(history_f1)
viz_f1.plot_learning_curves()  # Shows "F1" in title and labels

# Precision Classification
history_prec = model.fit(X_train, y_train, X_val, y_val, metric="precision")
viz_prec = Visualizer(history_prec)
viz_prec.plot_learning_curves()  # Shows "Precision" in title and labels
```

### Advanced Styling

```python
# Professional publication-quality plots
viz.plot_learning_curves(
    figsize=(10, 5),              # Larger figure
    ci=True,                      # Confidence intervals
    markers=True,                 # Epoch markers
    save_path="r2_curves.png"     # Save to file
)
```

## Training Animation with Dynamic Labels

The training animation automatically adapts all labels and titles based on your metric:

```python
# Create animated GIF with correct metric labels
viz.plot_training_animation(
    bg="dark",                    # Dark theme
    save_path="f1_training.gif"   # Save animation
)

# Animation features:
# - Dynamic title: "F1 Evolution" (not "Accuracy Evolution")
# - Dynamic y-axis: "F1" (not "Accuracy") 
# - Dynamic legend: "Train F1", "Val F1"
# - Dynamic bar chart: "Train F1", "Val F1" labels
```

## High-Performance Training Visualization

Works seamlessly with both `fit()` and `fit_fast()` methods:

```python
# Ultra-fast training with R² metric
history = model.fit_fast(
    X_train, y_train, X_val, y_val,
    epochs=100,
    batch_size=256,
    metric="r2",                  # R² regression
    eval_freq=5
)

viz = Visualizer(history)
viz.plot_learning_curves()       # Automatically shows "R²" labels
```

## Complete Workflow Examples

### Regression Workflow

```python
import numpy as np
from neuroscope import MLP, Visualizer

# Regression data
X_train = np.random.randn(1000, 20)
y_train = np.random.randn(1000, 1)
X_val = np.random.randn(200, 20)  
y_val = np.random.randn(200, 1)

# Regression model
model = MLP([20, 64, 32, 1], out_activation=None)
model.compile(optimizer="adam", lr=1e-3)

# Train with R² metric
history = model.fit(
    X_train, y_train, X_val, y_val,
    epochs=50,
    metric="r2"                   # R² for regression
)

# Visualize with automatic R² labels
viz = Visualizer(history)
viz.plot_learning_curves()       # Shows "R²" everywhere
viz.plot_training_animation()     # "R² Evolution" animation
```

### Classification Workflow

```python
# Binary classification data
X_train = np.random.randn(1000, 20)
y_train = np.random.randint(0, 2, (1000, 1))
X_val = np.random.randn(200, 20)
y_val = np.random.randint(0, 2, (200, 1))

# Binary classification model
model = MLP([20, 64, 32, 1], out_activation="sigmoid")
model.compile(optimizer="adam", lr=1e-3)

# Train with F1 metric
history = model.fit(
    X_train, y_train, X_val, y_val,
    epochs=50,
    metric="f1"                   # F1 for imbalanced classification
)

# Visualize with automatic F1 labels
viz = Visualizer(history)
viz.plot_learning_curves()       # Shows "F1" everywhere
viz.plot_training_animation()     # "F1 Evolution" animation
```

### Multi-Metric Comparison

```python
# Compare different metrics on same data
metrics = ["accuracy", "f1", "precision", "recall"]
histories = {}

for metric in metrics:
    model.reset_all()
    history = model.fit(
        X_train, y_train, X_val, y_val,
        epochs=30,
        metric=metric,
        verbose=False
    )
    histories[metric] = history

# Create separate visualizations for each metric
for metric, history in histories.items():
    viz = Visualizer(history)
    viz.plot_learning_curves(save_path=f"{metric}_curves.png")
    # Each plot shows correct labels: "Accuracy", "F1", "Precision", "Recall"
```

## Advanced Features

### Custom Metric Display Names

The system uses the MLP's built-in `_get_metric_display_name()` method, which you can extend:

```python
# The mapping is handled automatically:
# "r2" → "R²" (with proper superscript)
# "f1" → "F1" 
# "precision" → "Precision"
# "recall" → "Recall"
# "mse" → "MSE"
# "rmse" → "RMSE"
# "mae" → "MAE"
# Custom metrics → UPPERCASE
```

### Backward Compatibility

Existing code continues to work unchanged:

```python
# Old code still works
history = model.fit(X_train, y_train, X_val, y_val)  # Uses default "smart" metric
viz = Visualizer(history)
viz.plot_learning_curves()  # Shows appropriate labels based on model type
```

## Best Practices

### 1. Choose Appropriate Metrics

```python
# Regression tasks
history = model.fit(..., metric="r2")        # R² for explained variance
history = model.fit(..., metric="mse")       # MSE for loss-based evaluation
history = model.fit(..., metric="mae")       # MAE for robust evaluation

# Balanced classification
history = model.fit(..., metric="accuracy")  # Accuracy for balanced datasets

# Imbalanced classification  
history = model.fit(..., metric="f1")        # F1 for imbalanced datasets
history = model.fit(..., metric="precision") # Precision when false positives costly
history = model.fit(..., metric="recall")    # Recall when false negatives costly
```

### 2. Consistent Visualization Workflow

```python
# Standard workflow
history = model.fit(X_train, y_train, X_val, y_val, metric="your_metric")
viz = Visualizer(history)

# Always create learning curves first
viz.plot_learning_curves(figsize=(10, 5), ci=True, save_path="curves.png")

# Then create animation for presentations
viz.plot_training_animation(bg="dark", save_path="training.gif")
```

### 3. Publication-Quality Plots

```python
# High-quality settings for papers
viz.plot_learning_curves(
    figsize=(9, 4),               # Standard academic figure size
    ci=True,                      # Show confidence intervals
    markers=True,                 # Clear epoch markers
    save_path="paper_figure.png"  # High-DPI output
)
```

## Troubleshooting

### Common Issues

1. **Wrong labels showing**: Ensure you're passing the correct `metric` parameter to `fit()` or `fit_fast()`
2. **Old "Accuracy" labels**: Make sure you're using the updated NeuroScope version with dynamic labeling
3. **Animation not updating**: The animation system automatically uses the same metric detection as static plots

### Debug Information

```python
# Check what metric information is stored
print(f"Metric: {viz.metric}")
print(f"Display name: {viz.metric_display_name}")

# Verify training history contains metric info
print(f"History keys: {list(viz.hist.keys())}")
```

---

**The NeuroScope visualization system now provides professional, publication-quality plots with automatic metric detection - no more manual label management required!** 🎉
