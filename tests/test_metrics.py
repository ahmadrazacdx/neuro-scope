"""
Test suite for Metrics module.
Tests all evaluation metrics for regression and classification.
"""

import numpy as np
import pytest

from neuroscope.mlp.metrics import Metrics


class TestMetrics:
    """Test suite for evaluation metrics."""

    def test_accuracy_multiclass_basic(self):
        """Test multi-class accuracy with sparse labels."""
        y_true = np.array([0, 1, 2, 0, 1])
        y_pred = np.array(
            [
                [0.9, 0.05, 0.05],  # Correct: class 0
                [0.1, 0.8, 0.1],  # Correct: class 1
                [0.1, 0.1, 0.8],  # Correct: class 2
                [0.8, 0.1, 0.1],  # Correct: class 0
                [0.2, 0.7, 0.1],  # Correct: class 1
            ]
        )

        accuracy = Metrics.accuracy_multiclass(y_true, y_pred)

        assert isinstance(accuracy, float)
        assert 0.0 <= accuracy <= 1.0
        assert accuracy == 1.0  # All predictions correct

    def test_accuracy_multiclass_with_onehot(self):
        """Test multi-class accuracy with one-hot labels."""
        y_true_onehot = np.array(
            [[1, 0, 0], [0, 1, 0], [0, 0, 1]]  # class 0  # class 1  # class 2
        )
        y_pred = np.array([[0.9, 0.05, 0.05], [0.1, 0.8, 0.1], [0.1, 0.1, 0.8]])

        accuracy = Metrics.accuracy_multiclass(y_true_onehot, y_pred)

        assert isinstance(accuracy, float)
        assert 0.0 <= accuracy <= 1.0
        assert accuracy == 1.0

    def test_accuracy_multiclass_partial_correct(self):
        """Test multi-class accuracy with some incorrect predictions."""
        y_true = np.array([0, 1, 2, 0])
        y_pred = np.array(
            [
                [0.9, 0.05, 0.05],  # Correct: class 0
                [0.1, 0.8, 0.1],  # Correct: class 1
                [0.8, 0.1, 0.1],  # Incorrect: predicts 0, true is 2
                [0.1, 0.8, 0.1],  # Incorrect: predicts 1, true is 0
            ]
        )

        accuracy = Metrics.accuracy_multiclass(y_true, y_pred)
        assert accuracy == 0.5  # 2 out of 4 correct

    def test_accuracy_binary_basic(self):
        """Test binary classification accuracy."""
        y_true = np.array([0, 1, 1, 0, 1])
        y_pred = np.array([0.1, 0.8, 0.9, 0.2, 0.7])  # probabilities

        accuracy = Metrics.accuracy_binary(y_true, y_pred, thresh=0.5)

        assert isinstance(accuracy, float)
        assert 0.0 <= accuracy <= 1.0
        assert accuracy == 1.0  # All predictions correct with default threshold

    def test_accuracy_binary_different_threshold(self):
        """Test binary accuracy with different threshold."""
        y_true = np.array([0, 1, 1, 0])
        y_pred = np.array([0.3, 0.7, 0.6, 0.4])

        # With threshold 0.5
        acc_05 = Metrics.accuracy_binary(y_true, y_pred, thresh=0.5)

        # With threshold 0.6
        acc_06 = Metrics.accuracy_binary(y_true, y_pred, thresh=0.6)

        assert isinstance(acc_05, float)
        assert isinstance(acc_06, float)
        assert 0.0 <= acc_05 <= 1.0
        assert 0.0 <= acc_06 <= 1.0

        # They might be different due to different thresholds

    def test_accuracy_binary_perfect_predictions(self):
        """Test binary accuracy with perfect predictions."""
        y_true = np.array([0, 1, 0, 1])
        y_pred = np.array([0.0, 1.0, 0.0, 1.0])

        accuracy = Metrics.accuracy_binary(y_true, y_pred)
        assert accuracy == 1.0

    def test_accuracy_binary_worst_predictions(self):
        """Test binary accuracy with completely wrong predictions."""
        y_true = np.array([0, 1, 0, 1])
        y_pred = np.array([1.0, 0.0, 1.0, 0.0])  # All wrong

        accuracy = Metrics.accuracy_binary(y_true, y_pred)
        assert accuracy == 0.0

    def test_metrics_with_single_sample(self):
        """Test metrics with single sample."""
        # Binary accuracy
        acc_binary = Metrics.accuracy_binary(np.array([1]), np.array([0.8]))
        assert acc_binary == 1.0

        # Multi-class accuracy
        acc_multi = Metrics.accuracy_multiclass(
            np.array([1]), np.array([[0.1, 0.8, 0.1]])
        )
        assert acc_multi == 1.0

    def test_metrics_edge_cases(self):
        """Test metrics with edge cases."""
        # All same class
        y_true_same = np.array([1, 1, 1, 1])
        y_pred_same = np.array([0.9, 0.8, 0.7, 0.6])

        acc = Metrics.accuracy_binary(y_true_same, y_pred_same)
        assert isinstance(acc, float)
        assert 0.0 <= acc <= 1.0

    def test_multiclass_with_ties(self):
        """Test multi-class accuracy when predictions have ties."""
        y_true = np.array([0, 1])
        y_pred = np.array(
            [
                [0.5, 0.5, 0.0],  # Tie between class 0 and 1, argmax will pick 0
                [0.3, 0.3, 0.4],  # Class 2 wins
            ]
        )

        accuracy = Metrics.accuracy_multiclass(y_true, y_pred)
        # First prediction: true=0, pred=0 (correct due to argmax)
        # Second prediction: true=1, pred=2 (incorrect)
        assert accuracy == 0.5

    def test_consistency_across_calls(self):
        """Test that metrics are consistent across multiple calls."""
        y_true = np.array([0, 1, 0, 1, 1])
        y_pred = np.array([0.1, 0.9, 0.2, 0.8, 0.7])

        acc1 = Metrics.accuracy_binary(y_true, y_pred)
        acc2 = Metrics.accuracy_binary(y_true, y_pred)

        assert acc1 == acc2

    def test_available_metrics(self):
        """Test what metrics are actually available."""
        available_metrics = [
            attr
            for attr in dir(Metrics)
            if callable(getattr(Metrics, attr)) and not attr.startswith("_")
        ]

        # Should have at least accuracy functions
        assert len(available_metrics) > 0
        assert "accuracy_multiclass" in available_metrics
        assert "accuracy_binary" in available_metrics

        print(f"Available metrics: {available_metrics}")

    def test_other_metrics_if_available(self):
        """Test other metrics if they exist."""
        y_true = np.array([1.0, 2.0, 3.0, 4.0])
        y_pred = np.array([1.1, 2.1, 2.9, 4.1])

        # Test regression metrics if available
        if hasattr(Metrics, "mse"):
            mse = Metrics.mse(y_true, y_pred)
            assert isinstance(mse, (int, float))
            assert mse >= 0

        if hasattr(Metrics, "mae"):
            mae = Metrics.mae(y_true, y_pred)
            assert isinstance(mae, (int, float))
            assert mae >= 0

        if hasattr(Metrics, "r2_score"):
            r2 = Metrics.r2_score(y_true, y_pred)
            assert isinstance(r2, (int, float))

        # Test classification metrics
        y_true_class = np.array([0, 1, 1, 0])
        y_pred_class = np.array([0.1, 0.9, 0.8, 0.2])

        if hasattr(Metrics, "precision"):
            precision = Metrics.precision(y_true_class, y_pred_class > 0.5)
            assert isinstance(precision, (int, float))
            assert 0.0 <= precision <= 1.0

        if hasattr(Metrics, "recall"):
            recall = Metrics.recall(y_true_class, y_pred_class > 0.5)
            assert isinstance(recall, (int, float))
            assert 0.0 <= recall <= 1.0

        if hasattr(Metrics, "f1_score"):
            f1 = Metrics.f1_score(y_true_class, y_pred_class > 0.5)
            assert isinstance(f1, (int, float))
            assert 0.0 <= f1 <= 1.0

    @pytest.mark.parametrize("threshold", [0.1, 0.3, 0.5, 0.7, 0.9])
    def test_binary_accuracy_different_thresholds(self, threshold):
        """Test binary accuracy with various thresholds."""
        y_true = np.array([0, 1, 0, 1, 1, 0])
        y_pred = np.array([0.2, 0.8, 0.3, 0.9, 0.6, 0.4])

        accuracy = Metrics.accuracy_binary(y_true, y_pred, thresh=threshold)

        assert isinstance(accuracy, float)
        assert 0.0 <= accuracy <= 1.0

    def test_multiclass_accuracy_many_classes(self):
        """Test multi-class accuracy with many classes."""
        num_classes = 10
        num_samples = 50

        # Create random but valid predictions
        np.random.seed(42)
        y_true = np.random.randint(0, num_classes, num_samples)

        # Create predictions that favor the true class
        y_pred = np.random.rand(num_samples, num_classes)
        for i, true_class in enumerate(y_true):
            y_pred[i, true_class] += 0.5  # Boost true class probability

        # Normalize to get proper probabilities
        y_pred = y_pred / np.sum(y_pred, axis=1, keepdims=True)

        accuracy = Metrics.accuracy_multiclass(y_true, y_pred)

        assert isinstance(accuracy, float)
        assert 0.0 <= accuracy <= 1.0
        # Should be reasonably high due to boosting true class
        assert accuracy > 0.5

    def test_zero_accuracy_cases(self):
        """Test cases that should give zero accuracy."""
        # Binary case where all predictions are wrong
        y_true_binary = np.array([0, 1, 0, 1])
        y_pred_binary = np.array([0.9, 0.1, 0.8, 0.2])  # All wrong

        acc_binary = Metrics.accuracy_binary(y_true_binary, y_pred_binary)
        assert acc_binary == 0.0

        # Multi-class case where all predictions are wrong
        y_true_multi = np.array([0, 1, 2])
        y_pred_multi = np.array(
            [
                [0.1, 0.8, 0.1],  # Predicts 1, true is 0
                [0.8, 0.1, 0.1],  # Predicts 0, true is 1
                [0.1, 0.8, 0.1],  # Predicts 1, true is 2
            ]
        )

        acc_multi = Metrics.accuracy_multiclass(y_true_multi, y_pred_multi)
        assert acc_multi == 0.0

    def test_accuracy_handles_nan_predictions(self):
        """Ensure accuracy functions handle NaN predictions gracefully."""
        y_true = np.array([0, 1, 0, 1])
        y_pred = np.array([0.9, np.nan, 0.2, 0.8])

        # Binary accuracy should treat NaN as incorrect prediction
        acc = Metrics.accuracy_binary(y_true, y_pred)
        assert isinstance(acc, float)
        assert 0.0 <= acc <= 1.0

    def test_f1_score_zero_support(self):
        """Test F1 score when total support is zero."""
        # Edge case where no samples exist for any class
        y_true = np.array([])
        y_pred = np.array([]).reshape(0, 3)  # 3 classes but 0 samples

        # With empty arrays, should handle gracefully
        # We'll test with actual data but zero support scenario
        y_true = np.array([0, 0, 0])
        y_pred_probs = np.array([[0.9, 0.05, 0.05], [0.8, 0.1, 0.1], [0.85, 0.1, 0.05]])

        # All predictions are class 0, all true labels are class 0
        f1 = Metrics.f1_score(y_true, y_pred_probs, average="weighted")
        assert isinstance(f1, float)
        assert 0.0 <= f1 <= 1.0

    def test_precision_macro_average(self):
        """Test precision with macro averaging."""
        y_true = np.array([0, 1, 2, 0, 1, 2])
        y_pred = np.array(
            [
                [0.9, 0.05, 0.05],  # Correct: 0
                [0.1, 0.8, 0.1],  # Correct: 1
                [0.1, 0.1, 0.8],  # Correct: 2
                [0.9, 0.05, 0.05],  # Correct: 0
                [0.1, 0.8, 0.1],  # Correct: 1
                [0.1, 0.1, 0.8],  # Correct: 2
            ]
        )

        precision = Metrics.precision(y_true, y_pred, average="macro")
        assert isinstance(precision, float)
        assert 0.0 <= precision <= 1.0
        assert precision == 1.0  # All predictions correct

    def test_precision_no_average(self):
        """Test precision without averaging (returns per-class)."""
        y_true = np.array([0, 1, 2, 0, 1])
        y_pred = np.array(
            [
                [0.9, 0.05, 0.05],  # Correct: 0
                [0.1, 0.8, 0.1],  # Correct: 1
                [0.1, 0.1, 0.8],  # Correct: 2
                [0.9, 0.05, 0.05],  # Correct: 0
                [0.1, 0.8, 0.1],  # Correct: 1
            ]
        )

        precision = Metrics.precision(y_true, y_pred, average=None)
        assert isinstance(precision, np.ndarray)
        assert precision.shape == (3,)  # 3 classes
        assert np.all(precision >= 0.0)
        assert np.all(precision <= 1.0)

    def test_recall_macro_average(self):
        """Test recall with macro averaging."""
        y_true = np.array([0, 1, 2, 0, 1, 2])
        y_pred = np.array(
            [
                [0.9, 0.05, 0.05],  # Correct: 0
                [0.1, 0.8, 0.1],  # Correct: 1
                [0.1, 0.1, 0.8],  # Correct: 2
                [0.9, 0.05, 0.05],  # Correct: 0
                [0.1, 0.8, 0.1],  # Correct: 1
                [0.1, 0.1, 0.8],  # Correct: 2
            ]
        )

        recall = Metrics.recall(y_true, y_pred, average="macro")
        assert isinstance(recall, float)
        assert 0.0 <= recall <= 1.0
        assert recall == 1.0  # All predictions correct

    def test_recall_no_average(self):
        """Test recall without averaging (returns per-class)."""
        y_true = np.array([0, 1, 2, 0, 1])
        y_pred = np.array(
            [
                [0.9, 0.05, 0.05],  # Correct: 0
                [0.1, 0.8, 0.1],  # Correct: 1
                [0.1, 0.1, 0.8],  # Correct: 2
                [0.9, 0.05, 0.05],  # Correct: 0
                [0.1, 0.8, 0.1],  # Correct: 1
            ]
        )

        recall = Metrics.recall(y_true, y_pred, average=None)
        assert isinstance(recall, np.ndarray)
        assert recall.shape == (3,)  # 3 classes
        assert np.all(recall >= 0.0)
        assert np.all(recall <= 1.0)

    def test_r2_score_perfect_fit(self):
        """Test R² score with perfect predictions."""
        y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y_pred = np.array([1.0, 2.0, 3.0, 4.0, 5.0])

        r2 = Metrics.r2_score(y_true, y_pred)
        assert r2 == 1.0

    def test_r2_score_zero_variance(self):
        """Test R² score when true values have zero variance."""
        y_true = np.array([5.0, 5.0, 5.0, 5.0])
        y_pred = np.array([5.0, 5.0, 5.0, 5.0])

        # When ss_tot == 0 and ss_res == 0, should return 1.0
        r2 = Metrics.r2_score(y_true, y_pred)
        assert r2 == 1.0

        # When ss_tot == 0 but ss_res != 0
        y_pred_bad = np.array([4.0, 6.0, 5.0, 5.0])
        r2 = Metrics.r2_score(y_true, y_pred_bad)
        assert r2 == 0.0

    def test_apply_averaging_weighted_imbalanced(self):
        """Test weighted averaging with imbalanced class distribution."""
        # Create imbalanced dataset: many class 0, few class 1
        y_true = np.array([0, 0, 0, 0, 0, 0, 0, 0, 1, 1])  # 8 class 0, 2 class 1
        y_pred = np.array(
            [
                [0.9, 0.1],  # Correct: 0
                [0.9, 0.1],  # Correct: 0
                [0.9, 0.1],  # Correct: 0
                [0.9, 0.1],  # Correct: 0
                [0.9, 0.1],  # Correct: 0
                [0.9, 0.1],  # Correct: 0
                [0.9, 0.1],  # Correct: 0
                [0.9, 0.1],  # Correct: 0
                [0.1, 0.9],  # Correct: 1
                [0.1, 0.9],  # Correct: 1
            ]
        )

        # Weighted average should give more importance to class 0
        precision_weighted = Metrics.precision(y_true, y_pred, average="weighted")
        precision_macro = Metrics.precision(y_true, y_pred, average="macro")

        assert isinstance(precision_weighted, float)
        assert isinstance(precision_macro, float)
        # Both should be 1.0 since all predictions are correct
        assert precision_weighted == 1.0
        assert precision_macro == 1.0
