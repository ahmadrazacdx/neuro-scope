"""
NeuroScope Example: Binary Classification

Description: Binary classification example using NeuroScope with comprehensive
diagnostic analysis. Demonstrates sigmoid output activation, binary crossentropy
loss, and specialized binary metrics.

This example covers:
- Binary classification setup
- Sigmoid activation and BCE loss
- Binary-specific metrics and diagnostics
- Imbalanced dataset handling
- ROC curve analysis

Author: NeuroScope Team
Date: 2025
"""

import matplotlib.pyplot as plt
import numpy as np

from neuroscope import (
    MLP,
    PreTrainingAnalyzer,
    TrainingMonitor,
    Visualizer,
    accuracy_binary,
    f1_score,
    precision,
    recall,
)


def generate_binary_data(n_samples=1000, n_features=10, class_ratio=0.3, noise=0.2):
    """Generate synthetic binary classification data."""
    np.random.seed(42)

    # Generate features
    X = np.random.randn(n_samples, n_features)

    # Create separable classes with some overlap
    n_positive = int(n_samples * class_ratio)
    n_negative = n_samples - n_positive

    # Positive class: shift features in positive direction
    X[:n_positive] += np.array([1.5, 1.0, 0.5, -0.5, 1.2, 0.8, -0.3, 1.1, 0.6, -0.2])

    # Negative class: shift features in negative direction
    X[n_positive:] += np.array(
        [-1.2, -0.8, -1.0, 0.3, -1.5, -0.6, 0.4, -0.9, -1.1, 0.1]
    )

    # Add noise
    X += np.random.randn(n_samples, n_features) * noise

    # Create binary labels
    y = np.zeros((n_samples, 1))
    y[:n_positive] = 1

    # Shuffle data
    indices = np.random.permutation(n_samples)
    X = X[indices]
    y = y[indices]

    return X, y


def calculate_roc_curve(y_true, y_scores, n_thresholds=100):
    """Calculate ROC curve points."""
    thresholds = np.linspace(0, 1, n_thresholds)
    tpr_values = []
    fpr_values = []

    for threshold in thresholds:
        y_pred = (y_scores >= threshold).astype(int)

        # Calculate confusion matrix elements
        tp = np.sum((y_true == 1) & (y_pred == 1))
        tn = np.sum((y_true == 0) & (y_pred == 0))
        fp = np.sum((y_true == 0) & (y_pred == 1))
        fn = np.sum((y_true == 1) & (y_pred == 0))

        # Calculate TPR and FPR
        tpr = tp / (tp + fn) if (tp + fn) > 0 else 0
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0

        tpr_values.append(tpr)
        fpr_values.append(fpr)

    return np.array(fpr_values), np.array(tpr_values), thresholds


def calculate_auc(fpr, tpr):
    """Calculate Area Under Curve using trapezoidal rule."""
    return np.trapz(tpr, fpr)


def main():
    """Main binary classification example."""
    print("🎯 NeuroScope Example: Binary Classification")
    print("=" * 45)

    # Step 1: Generate imbalanced binary data
    print("\n📊 Step 1: Preparing Binary Data")
    print("-" * 30)

    X, y = generate_binary_data(n_samples=1000, n_features=10, class_ratio=0.3)

    # Split data
    n_train = int(0.7 * len(X))
    n_val = int(0.15 * len(X))

    X_train, y_train = X[:n_train], y[:n_train]
    X_val, y_val = X[n_train : n_train + n_val], y[n_train : n_train + n_val]
    X_test, y_test = X[n_train + n_val :], y[n_train + n_val :]

    # Analyze class distribution
    train_positive_ratio = np.mean(y_train)
    val_positive_ratio = np.mean(y_val)
    test_positive_ratio = np.mean(y_test)

    print("Dataset statistics:")
    print(f"  Total samples: {len(X)}")
    print(f"  Features: {X.shape[1]}")
    print(f"  Training samples: {len(X_train)} ({train_positive_ratio:.1%} positive)")
    print(f"  Validation samples: {len(X_val)} ({val_positive_ratio:.1%} positive)")
    print(f"  Test samples: {len(X_test)} ({test_positive_ratio:.1%} positive)")

    # Step 2: Create binary classification model
    print("\n🏗️ Step 2: Creating Binary Classifier")
    print("-" * 35)

    model = MLP(
        layer_dims=[10, 32, 16, 1],  # Input -> Hidden -> Hidden -> Output (1 neuron)
        hidden_activation="relu",  # ReLU for hidden layers
        out_activation="sigmoid",  # Sigmoid for binary classification
        init_method="he",  # He initialization
        dropout_rate=0.3,  # Higher dropout for small dataset
        l2_reg=1e-3,  # L2 regularization
    )

    # Compile with binary-specific settings
    model.compile(
        optimizer="adam",
        lr=1e-3,
        loss="bce",  # Binary crossentropy
        metrics=["accuracy"],
    )

    print("Binary classifier configuration:")
    print(f"  Architecture: {model.layer_dims}")
    print(f"  Output activation: {model.out_activation}")
    print("  Loss function: Binary Crossentropy")
    print(
        f"  Total parameters: {sum(w.size for w in model.weights) + sum(b.size for b in model.biases)}"
    )

    # Step 3: Pre-training analysis
    print("\n🔍 Step 3: Pre-training Analysis")
    print("-" * 30)

    analyzer = PreTrainingAnalyzer(model)
    pre_results = analyzer.analyze(X_train, y_train)

    print("Pre-training diagnostics:")
    print(f"  Initial loss: {pre_results['initial_loss']:.4f}")
    print("  Expected loss (random): ~0.693")
    print(f"  Weight initialization: {pre_results['weight_init_quality']}")

    # For binary classification, random performance should be ~0.693 (log(2))
    if pre_results["initial_loss"] > 1.0:
        print("  ⚠️  Initial loss is high - check data preprocessing")
    else:
        print("  ✅ Initial loss looks reasonable")

    # Step 4: Train with monitoring
    print("\n🚀 Step 4: Training Binary Classifier")
    print("-" * 35)

    monitor = TrainingMonitor(
        check_dead_neurons=True,
        check_gradients=True,
        dead_neuron_threshold=0.05,  # Train the model with precision metric for imbalanced data
    )
    print("\n🚀 Training the neural network...")
    history = model.fit(
        X_train,
        y_train,
        X_val=X_val,
        y_val=y_val,
        epochs=150,
        batch_size=32,
        monitor=monitor,
        verbose=True,
        early_stopping=True,
        patience=15,
        metric="precision",  # Precision metric - plots will show "Precision" labels
    )

    print(f"Training completed in {len(history['history']['loss'])} epochs")

    # Step 5: Comprehensive evaluation
    print("\n📈 Step 5: Binary Classification Metrics")
    print("-" * 40)

    # Get predictions
    train_pred_proba = model.predict(X_train)
    val_pred_proba = model.predict(X_val)
    test_pred_proba = model.predict(X_test)

    # Convert to binary predictions (threshold = 0.5)
    train_pred_binary = (train_pred_proba > 0.5).astype(int)
    val_pred_binary = (val_pred_proba > 0.5).astype(int)
    test_pred_binary = (test_pred_proba > 0.5).astype(int)

    # Calculate comprehensive metrics
    def calculate_binary_metrics(y_true, y_pred_proba, y_pred_binary, dataset_name):
        """Calculate and display binary classification metrics."""
        acc = accuracy_binary(y_true, y_pred_binary)
        prec = precision(y_true, y_pred_binary)
        rec = recall(y_true, y_pred_binary)
        f1 = f1_score(y_true, y_pred_binary)

        print(f"\n{dataset_name} Metrics:")
        print(f"  Accuracy: {acc:.4f}")
        print(f"  Precision: {prec:.4f}")
        print(f"  Recall: {rec:.4f}")
        print(f"  F1-Score: {f1:.4f}")

        return acc, prec, rec, f1

    # Calculate metrics for all sets
    train_metrics = calculate_binary_metrics(
        y_train, train_pred_proba, train_pred_binary, "Training"
    )
    val_metrics = calculate_binary_metrics(
        y_val, val_pred_proba, val_pred_binary, "Validation"
    )
    test_metrics = calculate_binary_metrics(
        y_test, test_pred_proba, test_pred_binary, "Test"
    )

    # Calculate ROC curve for test set
    fpr, tpr, thresholds = calculate_roc_curve(
        y_test.flatten(), test_pred_proba.flatten()
    )
    auc_score = calculate_auc(fpr, tpr)
    print(f"\nROC AUC Score: {auc_score:.4f}")

    # Step 6: Visualize with Dynamic Precision Labels
    print("\n📊 Step 6: Visualization with Dynamic Precision Labels")
    print("-" * 50)

    viz = Visualizer(history)

    # Learning curves - automatically shows "Precision" labels
    print("Creating learning curves with automatic Precision labels...")
    viz.plot_learning_curves(figsize=(10, 5), ci=True, markers=True)

    # Training animation - shows "Precision Evolution"
    print("Creating training animation with Precision labels...")
    viz.plot_training_animation(bg="dark")

    print(
        "✨ All plots automatically show 'Precision' labels based on your training metric!"
    )

    # Step 7: Advanced visualization
    print("\n📊 Step 6: Binary Classification Visualization")
    print("-" * 45)

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle("Binary Classification Analysis", fontsize=16)

    # Learning curves
    epochs = range(1, len(history["history"]["loss"]) + 1)

    axes[0, 0].plot(epochs, history["history"]["loss"], "b-", label="Training")
    axes[0, 0].plot(epochs, history["history"]["val_loss"], "r-", label="Validation")
    axes[0, 0].set_title("Loss Curves")
    axes[0, 0].set_xlabel("Epoch")
    axes[0, 0].set_ylabel("Binary Crossentropy")
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    axes[0, 1].plot(epochs, history["history"]["accuracy"], "b-", label="Training")
    axes[0, 1].plot(
        epochs, history["history"]["val_accuracy"], "r-", label="Validation"
    )
    axes[0, 1].set_title("Accuracy Curves")
    axes[0, 1].set_xlabel("Epoch")
    axes[0, 1].set_ylabel("Accuracy")
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # ROC Curve
    axes[0, 2].plot(fpr, tpr, "b-", linewidth=2, label=f"ROC (AUC = {auc_score:.3f})")
    axes[0, 2].plot([0, 1], [0, 1], "r--", alpha=0.5, label="Random")
    axes[0, 2].set_title("ROC Curve")
    axes[0, 2].set_xlabel("False Positive Rate")
    axes[0, 2].set_ylabel("True Positive Rate")
    axes[0, 2].legend()
    axes[0, 2].grid(True, alpha=0.3)

    # Prediction distribution
    axes[1, 0].hist(
        test_pred_proba[y_test.flatten() == 0],
        bins=20,
        alpha=0.7,
        label="Negative Class",
        density=True,
        color="red",
    )
    axes[1, 0].hist(
        test_pred_proba[y_test.flatten() == 1],
        bins=20,
        alpha=0.7,
        label="Positive Class",
        density=True,
        color="blue",
    )
    axes[1, 0].axvline(
        x=0.5, color="black", linestyle="--", alpha=0.7, label="Threshold"
    )
    axes[1, 0].set_title("Prediction Distribution")
    axes[1, 0].set_xlabel("Predicted Probability")
    axes[1, 0].set_ylabel("Density")
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    # Confusion Matrix
    tp = np.sum((y_test == 1) & (test_pred_binary == 1))
    tn = np.sum((y_test == 0) & (test_pred_binary == 0))
    fp = np.sum((y_test == 0) & (test_pred_binary == 1))
    fn = np.sum((y_test == 1) & (test_pred_binary == 0))

    confusion = np.array([[tn, fp], [fn, tp]])
    im = axes[1, 1].imshow(confusion, cmap="Blues", interpolation="nearest")
    axes[1, 1].set_title("Confusion Matrix")
    axes[1, 1].set_xlabel("Predicted")
    axes[1, 1].set_ylabel("Actual")
    axes[1, 1].set_xticks([0, 1])
    axes[1, 1].set_yticks([0, 1])
    axes[1, 1].set_xticklabels(["Negative", "Positive"])
    axes[1, 1].set_yticklabels(["Negative", "Positive"])

    # Add text annotations
    for i in range(2):
        for j in range(2):
            axes[1, 1].text(
                j,
                i,
                confusion[i, j],
                ha="center",
                va="center",
                color="white" if confusion[i, j] > confusion.max() / 2 else "black",
            )

    # Metrics comparison
    metrics_names = ["Accuracy", "Precision", "Recall", "F1-Score"]
    train_vals = list(train_metrics)
    val_vals = list(val_metrics)
    test_vals = list(test_metrics)

    x = np.arange(len(metrics_names))
    width = 0.25

    axes[1, 2].bar(x - width, train_vals, width, label="Train", alpha=0.8)
    axes[1, 2].bar(x, val_vals, width, label="Validation", alpha=0.8)
    axes[1, 2].bar(x + width, test_vals, width, label="Test", alpha=0.8)
    axes[1, 2].set_title("Metrics Comparison")
    axes[1, 2].set_ylabel("Score")
    axes[1, 2].set_xticks(x)
    axes[1, 2].set_xticklabels(metrics_names, rotation=45)
    axes[1, 2].legend()
    axes[1, 2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

    # Step 7: Threshold analysis
    print("\n🎚️ Step 7: Threshold Analysis")
    print("-" * 28)

    # Find optimal threshold
    f1_scores = []
    test_thresholds = np.linspace(0.1, 0.9, 17)

    for threshold in test_thresholds:
        pred_binary = (test_pred_proba > threshold).astype(int)
        f1 = f1_score(y_test, pred_binary)
        f1_scores.append(f1)

    optimal_idx = np.argmax(f1_scores)
    optimal_threshold = test_thresholds[optimal_idx]
    optimal_f1 = f1_scores[optimal_idx]

    print("Threshold analysis:")
    print(f"  Default threshold (0.5): F1 = {f1_score(y_test, test_pred_binary):.4f}")
    print(f"  Optimal threshold ({optimal_threshold:.2f}): F1 = {optimal_f1:.4f}")

    # Final evaluation with optimal threshold
    optimal_pred_binary = (test_pred_proba > optimal_threshold).astype(int)
    optimal_acc = accuracy_binary(y_test, optimal_pred_binary)
    optimal_prec = precision(y_test, optimal_pred_binary)
    optimal_rec = recall(y_test, optimal_pred_binary)

    print("\nOptimal threshold performance:")
    print(f"  Accuracy: {optimal_acc:.4f}")
    print(f"  Precision: {optimal_prec:.4f}")
    print(f"  Recall: {optimal_rec:.4f}")
    print(f"  F1-Score: {optimal_f1:.4f}")

    # Summary
    print("\n🎉 Binary Classification Complete!")
    print("=" * 35)
    print(f"✅ Model achieved {test_metrics[0]:.1%} test accuracy")
    print(f"✅ ROC AUC score: {auc_score:.3f}")
    print(f"✅ Optimal F1-score: {optimal_f1:.3f} at threshold {optimal_threshold:.2f}")

    if auc_score > 0.8:
        print("🌟 Excellent classification performance!")
    elif auc_score > 0.7:
        print("👍 Good classification performance!")
    else:
        print("⚠️  Consider improving the model or getting more data")

    print("\n💡 Key insights:")
    print("   - Binary classification requires sigmoid output and BCE loss")
    print("   - Threshold tuning can significantly improve performance")
    print("   - ROC AUC is threshold-independent and robust for imbalanced data")
    print("   - Monitor precision/recall trade-off for business requirements")


if __name__ == "__main__":
    main()
