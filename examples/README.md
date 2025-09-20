# NeuroScope Examples

This directory contains comprehensive examples demonstrating NeuroScope's capabilities across different use cases and domains.

## 📁 Directory Structure

```
examples/
├── README.md                    # This file
├── basic/                       # Basic usage examples
│   ├── 01_first_neural_network.py
│   ├── 02_binary_classification.py
│   └── 03_regression_example.py
├── intermediate/                # Intermediate examples
│   ├── 04_hyperparameter_tuning.py
│   ├── 05_diagnostic_analysis.py
│   └── 06_visualization_gallery.py
├── advanced/                    # Advanced usage patterns
│   ├── 07_custom_components.py
│   ├── 08_training_strategies.py
│   └── 09_production_pipeline.py
├── notebooks/                   # Jupyter notebooks
│   ├── mnist_classification.ipynb
│   ├── iris_classification.ipynb
│   ├── boston_housing_regression.ipynb
│   └── diagnostic_deep_dive.ipynb
└── datasets/                    # Sample datasets
    ├── load_datasets.py
    └── synthetic_data.py
```

## 🚀 Quick Start

### Prerequisites

```bash
# Install NeuroScope with examples dependencies
pip install neuroscope[examples]

# Or install individual dependencies
pip install jupyter matplotlib seaborn pandas scikit-learn
```

### Running Examples

```bash
# Run basic examples
cd examples/basic
python 01_first_neural_network.py

# Launch Jupyter notebooks
cd examples/notebooks
jupyter notebook mnist_classification.ipynb
```

## 📚 Example Categories

### 🔰 Basic Examples

Perfect for beginners learning NeuroScope fundamentals:

- **[First Neural Network](basic/01_first_neural_network.py)**: Complete walkthrough from data to predictions
- **[Binary Classification](basic/02_binary_classification.py)**: Two-class classification with diagnostics
- **[Regression Example](basic/03_regression_example.py)**: Continuous target prediction

### 🎯 Intermediate Examples

For users comfortable with basics, exploring advanced features:

- **[Hyperparameter Tuning](intermediate/04_hyperparameter_tuning.py)**: Systematic optimization strategies
- **[Diagnostic Analysis](intermediate/05_diagnostic_analysis.py)**: Deep dive into training diagnostics
- **[Visualization Gallery](intermediate/06_visualization_gallery.py)**: Comprehensive plotting examples

### 🚀 Advanced Examples

Production-ready patterns and customization:

- **[Custom Components](advanced/07_custom_components.py)**: Building custom loss functions, metrics, and activations
- **[Training Strategies](advanced/08_training_strategies.py)**: Advanced training techniques and optimization
- **[Production Pipeline](advanced/09_production_pipeline.py)**: End-to-end ML pipeline with NeuroScope

### 📓 Jupyter Notebooks

Interactive tutorials with detailed explanations:

- **[MNIST Classification](notebooks/mnist_classification.ipynb)**: Classic handwritten digit recognition
- **[Iris Classification](notebooks/iris_classification.ipynb)**: Multi-class flower classification
- **[Boston Housing Regression](notebooks/boston_housing_regression.ipynb)**: House price prediction
- **[Diagnostic Deep Dive](notebooks/diagnostic_deep_dive.ipynb)**: Comprehensive diagnostic analysis

## 🎓 Learning Path

### For Beginners
1. Start with `01_first_neural_network.py`
2. Try `02_binary_classification.py` 
3. Explore `mnist_classification.ipynb`
4. Move to intermediate examples

### For Experienced Users
1. Review `05_diagnostic_analysis.py`
2. Explore `06_visualization_gallery.py`
3. Study `07_custom_components.py`
4. Implement `09_production_pipeline.py`

## 🔧 Common Patterns

### Model Creation Pattern
```python
from neuroscope import MLP, PreTrainingAnalyzer, TrainingMonitor, Visualizer

# 1. Create model
model = MLP([input_size, hidden_size, output_size], 
           hidden_activation="relu", out_activation="softmax")

# 2. Compile
model.compile(optimizer="adam", lr=1e-3, loss="cce", metrics=["accuracy"])

# 3. Pre-training analysis
analyzer = PreTrainingAnalyzer(model)
pre_results = analyzer.analyze(X_train, y_train)

# 4. Train with monitoring
monitor = TrainingMonitor()
history = model.fit(X_train, y_train, validation_data=(X_val, y_val),
                   epochs=100, monitor=monitor)

# 5. Visualize
viz = Visualizer(history)
viz.plot_learning_curves()
```

### Diagnostic Pattern
```python
from neuroscope.diagnostics import PostTrainingEvaluator

# Post-training evaluation
evaluator = PostTrainingEvaluator(model)
performance = evaluator.evaluate_performance(X_test, y_test)
robustness = evaluator.evaluate_robustness(X_test, y_test, noise_levels=[0.1, 0.2])

print(f"Test Accuracy: {performance['accuracy']:.3f}")
print(f"Robustness: {robustness['mean_accuracy']:.3f}")
```

## 🛠️ Troubleshooting

### Common Issues

1. **Import Errors**: Ensure NeuroScope is installed: `pip install neuroscope`
2. **Missing Dependencies**: Install example dependencies: `pip install matplotlib pandas scikit-learn`
3. **Jupyter Issues**: Install Jupyter: `pip install jupyter`
4. **Data Loading**: Check dataset paths and permissions

### Getting Help

- **Documentation**: [https://ahmadrazacdx.github.io/neuroscope/](https://ahmadrazacdx.github.io/neuroscope/)
- **Issues**: [GitHub Issues](https://github.com/ahmadrazacdx/neuro-scope/issues)
- **Discussions**: [GitHub Discussions](https://github.com/ahmadrazacdx/neuro-scope/discussions)

## 🤝 Contributing Examples

We welcome new examples! Please:

1. Follow the existing code style and structure
2. Include comprehensive comments and docstrings
3. Add a brief description in this README
4. Test your example thoroughly
5. Submit a pull request

### Example Template

```python
"""
NeuroScope Example: [Title]

Description: [Brief description of what this example demonstrates]

Author: [Your name]
Date: [Date]
"""

import numpy as np
from neuroscope import MLP

def main():
    """Main example function."""
    # Your example code here
    pass

if __name__ == "__main__":
    main()
```

---

Happy learning with NeuroScope! 🧠✨
