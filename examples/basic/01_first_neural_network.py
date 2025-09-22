"""
NeuroScope Example: Your First Neural Network

Description: Complete walkthrough from data preparation to model evaluation,
demonstrating the core NeuroScope workflow with synthetic data.

This example covers:
- Data preparation and preprocessing
- Model creation and compilation
- Pre-training analysis
- Training with monitoring
- Visualization and evaluation

Author: NeuroScope Team
Date: 2025
"""

import matplotlib.pyplot as plt
import numpy as np

from neuroscope import MLP, PreTrainingAnalyzer, TrainingMonitor, Visualizer


def generate_synthetic_data(n_samples=1000, n_features=20, n_classes=3, noise=0.1):
    """Generate synthetic classification data for demonstration."""
    np.random.seed(42)

    # Generate random data
    X = np.random.randn(n_samples, n_features)

    # Create class-specific patterns
    for i in range(n_classes):
        class_mask = np.arange(n_samples) % n_classes == i
        X[class_mask] += np.random.randn(n_features) * 2  # Class-specific offset

    # Add noise
    X += np.random.randn(n_samples, n_features) * noise

    # Create labels (one-hot encoded)
    y = np.zeros((n_samples, n_classes))
    for i in range(n_samples):
        y[i, i % n_classes] = 1

    return X, y


def split_data(X, y, train_ratio=0.7, val_ratio=0.15):
    """Split data into train, validation, and test sets."""
    n_samples = X.shape[0]
    n_train = int(n_samples * train_ratio)
    n_val = int(n_samples * val_ratio)

    # Shuffle indices
    indices = np.random.permutation(n_samples)

    # Split indices
    train_idx = indices[:n_train]
    val_idx = indices[n_train : n_train + n_val]
    test_idx = indices[n_train + n_val :]

    return (
        X[train_idx],
        y[train_idx],
        X[val_idx],
        y[val_idx],
        X[test_idx],
        y[test_idx],
    )


def main():
    """Main example demonstrating NeuroScope workflow."""
    print("NeuroScope Example: Your First Neural Network")
    print("=" * 50)

    # Step 1: Generate and prepare data
    print("\nStep 1: Preparing Data")
    print("-" * 25)

    X, y = generate_synthetic_data(n_samples=1000, n_features=20, n_classes=3)
    X_train, y_train, X_val, y_val, X_test, y_test = split_data(X, y)

    print(f"Training samples: {X_train.shape[0]}")
    print(f"Validation samples: {X_val.shape[0]}")
    print(f"Test samples: {X_test.shape[0]}")
    print(f"Features: {X_train.shape[1]}")
    print(f"Classes: {y_train.shape[1]}")

    # Step 2: Create and configure model
    print("\nStep 2: Creating Model")
    print("-" * 25)

    model = MLP(
        layer_dims=[20, 64, 32, 3],  # Input -> Hidden -> Hidden -> Output
        hidden_activation="relu",  # ReLU for hidden layers
        out_activation="softmax",  # Softmax for multi-class classification
        init_method="he",  # He initialization for ReLU
        dropout_rate=0.2,  # 20% dropout for regularization
    )

    # Compile the model
    model.compile(
        optimizer="adam",  # Adam optimizer
        lr=1e-3,  # Learning rate
        reg="l2",  # L2 regularization
        lamda=1e-4,  # Regularization strength
    )

    print("Model architecture:")
    print(f"  Layers: {model.layer_dims}")
    print(f"  Hidden activation: {model.hidden_activation}")
    print(f"  Output activation: {model.out_activation}")
    print(
        f"  Parameters: {sum(w.size for w in model.weights) + sum(b.size for b in model.biases)}"
    )

    # Step 3: Pre-training analysis
    print("\nStep 3: Pre-training Analysis")
    print("-" * 30)

    analyzer = PreTrainingAnalyzer(model)
    pre_results = analyzer.analyze(X_train, y_train)

    print("Pre-training diagnostics:")
    print(f"  Initial loss: {pre_results['initial_loss']:.4f}")
    print(f"  Expected loss: {pre_results['expected_loss']:.4f}")
    print(f"  Loss ratio: {pre_results['loss_ratio']:.2f}")
    print(f"  Weight init quality: {pre_results['weight_init_quality']}")

    if pre_results["loss_ratio"] > 1.5:
        print("Warning: Initial loss is high - consider checking data preprocessing")
    else:
        print("Info: Initial loss looks good")

    # Step 4: Train with monitoring
    print("\nStep 4: Training Model")
    print("-" * 25)

    monitor = TrainingMonitor()
    print("\n🚀 Training the neural network...")
    history = model.fit(
        X_train,
        y_train,
        X_val=X_val,
        y_val=y_val,
        epochs=100,
        batch_size=32,
        monitor=monitor,
        verbose=True,
        early_stopping_patience=10,
        metric="f1",
    )

    print(f"\nTraining completed in {len(history['history']['loss'])} epochs")

    # Check for training issues
    if monitor.has_dead_neurons():
        print("Warning: Dead neurons detected during training")
    if monitor.has_vanishing_gradients():
        print("Warning: Vanishing gradients detected")
    if monitor.has_exploding_gradients():
        print("Warning: Exploding gradients detected")

    # Step 5: Evaluate model
    print("\nStep 5: Model Evaluation")
    print("-" * 25)

    # Training metrics
    final_train_loss = history["history"]["loss"][-1]
    final_train_acc = history["history"]["accuracy"][-1]
    final_val_loss = history["history"]["val_loss"][-1]
    final_val_acc = history["history"]["val_accuracy"][-1]

    print("Final training metrics:")
    print(f"  Train loss: {final_train_loss:.4f}")
    print(f"  Train accuracy: {final_train_acc:.4f}")
    print(f"  Validation loss: {final_val_loss:.4f}")
    print(f"  Validation accuracy: {final_val_acc:.4f}")

    # Test evaluation
    test_predictions = model.predict(X_test)
    test_loss = model.evaluate(X_test, y_test)
    test_accuracy = np.mean(
        np.argmax(test_predictions, axis=1) == np.argmax(y_test, axis=1)
    )

    print("\nTest metrics:")
    print(f"  Test loss: {test_loss:.4f}")
    print(f"  Test accuracy: {test_accuracy:.4f}")

    # Step 5: Visualize Results with Dynamic Metric Labels
    print("\nStep 5: Visualization with Dynamic Labels")
    print("-" * 40)

    viz = Visualizer(history)

    # Learning curves - automatically shows "F1" labels instead of "Accuracy"
    print("📊 Creating learning curves with automatic F1 labels...")
    viz.plot_learning_curves(figsize=(9, 4), ci=True, markers=True)

    # Training animation - shows "F1 Evolution" instead of "Accuracy Evolution"
    print("🎬 Creating training animation with F1 labels...")
    viz.plot_training_animation(bg="light")

    print(
        "✨ Notice: All plots automatically show 'F1' labels based on your training metric!"
    )

    # Step 6: Visualization
    print("\nStep 6: Visualization")
    print("-" * 23)

    # Create a figure with multiple subplots
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle("NeuroScope Training Analysis", fontsize=16)

    # Learning curves
    plt.subplot(2, 2, 1)
    epochs = range(1, len(history["history"]["loss"]) + 1)
    plt.plot(epochs, history["history"]["loss"], "b-", label="Training Loss")
    plt.plot(epochs, history["history"]["val_loss"], "r-", label="Validation Loss")
    plt.title("Learning Curves - Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.subplot(2, 2, 2)
    plt.plot(epochs, history["history"]["accuracy"], "b-", label="Training Accuracy")
    plt.plot(
        epochs, history["history"]["val_accuracy"], "r-", label="Validation Accuracy"
    )
    plt.title("Learning Curves - Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Prediction distribution
    plt.subplot(2, 2, 3)
    test_pred_classes = np.argmax(test_predictions, axis=1)
    test_true_classes = np.argmax(y_test, axis=1)

    plt.hist(test_pred_classes, bins=3, alpha=0.7, label="Predictions", density=True)
    plt.hist(test_true_classes, bins=3, alpha=0.7, label="True Labels", density=True)
    plt.title("Class Distribution")
    plt.xlabel("Class")
    plt.ylabel("Density")
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Confusion matrix (simple)
    plt.subplot(2, 2, 4)
    confusion = np.zeros((3, 3))
    for i in range(len(test_true_classes)):
        confusion[test_true_classes[i], test_pred_classes[i]] += 1

    plt.imshow(confusion, cmap="Blues", interpolation="nearest")
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.colorbar()

    # Add text annotations
    for i in range(3):
        for j in range(3):
            plt.text(j, i, int(confusion[i, j]), ha="center", va="center")

    plt.tight_layout()
    plt.show()

    # Summary
    print("\nExample Complete!")
    print("=" * 20)
    print(
        f"Successfully trained a neural network with {test_accuracy:.1%} test accuracy"
    )
    print(f"Model converged in {len(history['history']['loss'])} epochs")
    print("No major training issues detected")
    print("\nNext steps:")
    print("   - Try different architectures or hyperparameters")
    print("   - Explore the diagnostic tools in more detail")
    print("   - Check out more advanced examples")


if __name__ == "__main__":
    main()
