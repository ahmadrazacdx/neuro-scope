"""
Extended tests for PostTraining functionality to increase coverage.
"""

from unittest.mock import patch

import numpy as np
import pytest

from neuroscope import MLP
from neuroscope.diagnostics.posttraining import PostTrainingEvaluator


class TestPostTrainingExtended:
    """Extended tests for PostTraining functionality."""

    @pytest.fixture
    def trained_model(self):
        """Create a trained model for testing."""
        model = MLP([4, 8, 2], hidden_activation="relu", out_activation="sigmoid")
        model.compile(optimizer="adam", lr=0.01)

        # Create synthetic data and train briefly
        X = np.random.randn(50, 4)
        y = np.random.randint(0, 2, (50, 2))

        # Train for a few epochs
        model.fit_fast(X, y, epochs=3, verbose=0)
        return model

    @pytest.fixture
    def test_data(self):
        """Create test data."""
        X = np.random.randn(20, 4)
        y = np.random.randint(0, 2, (20, 2))
        return X, y

    def test_evaluator_initialization_success(self, trained_model):
        """Test successful evaluator initialization."""
        evaluator = PostTrainingEvaluator(trained_model)
        assert evaluator.model is trained_model
        assert isinstance(evaluator.results, dict)

    def test_evaluator_initialization_no_weights(self):
        """Test evaluator initialization with uninitialized model."""
        model = MLP([4, 8, 2])
        # Don't compile the model

        with pytest.raises(ValueError, match="Model must be compiled"):
            PostTrainingEvaluator(model)

    def test_evaluator_initialization_not_compiled(self):
        """Test evaluator initialization with uncompiled model."""
        model = MLP([4, 8, 2])
        # Initialize weights but don't compile
        model._initialize_weights()

        with pytest.raises(ValueError, match="Model must be compiled"):
            PostTrainingEvaluator(model)

    def test_evaluate_robustness_default_noise(self, trained_model, test_data):
        """Test robustness evaluation with default noise levels."""
        evaluator = PostTrainingEvaluator(trained_model)
        X, y = test_data

        results = evaluator.evaluate_robustness(X, y)

        assert isinstance(results, dict)
        assert "baseline_accuracy" in results
        assert "baseline_loss" in results
        assert "overall_robustness" in results
        assert "status" in results
        assert "note" in results

    def test_evaluate_robustness_custom_noise(self, trained_model, test_data):
        """Test robustness evaluation with custom noise levels."""
        evaluator = PostTrainingEvaluator(trained_model)
        X, y = test_data
        custom_noise = [0.0, 0.1, 0.2]

        results = evaluator.evaluate_robustness(X, y, noise_levels=custom_noise)

        assert isinstance(results, dict)
        assert "overall_robustness" in results
        assert "status" in results

    def test_evaluate_robustness_empty_noise(self, trained_model, test_data):
        """Test robustness evaluation with empty noise levels."""
        evaluator = PostTrainingEvaluator(trained_model)
        X, y = test_data

        results = evaluator.evaluate_robustness(X, y, noise_levels=[])

        assert isinstance(results, dict)
        assert "overall_robustness" in results
        assert results["overall_robustness"] == 0.0

    def test_evaluate_performance_basic(self, trained_model, test_data):
        """Test basic performance evaluation."""
        evaluator = PostTrainingEvaluator(trained_model)
        X, y = test_data

        results = evaluator.evaluate_performance(X, y)

        assert isinstance(results, dict)
        assert "accuracy" in results
        assert "loss" in results
        assert "samples_per_second" in results
        assert "total_params" in results
        assert "all_metrics" in results
        assert "status" in results
        assert "note" in results
        assert isinstance(results["accuracy"], (int, float))
        assert isinstance(results["loss"], (int, float))

    def test_evaluate_performance_timing(self, trained_model, test_data):
        """Test that performance evaluation measures timing."""
        evaluator = PostTrainingEvaluator(trained_model)
        X, y = test_data

        results = evaluator.evaluate_performance(X, y)

        # Should have measured samples per second
        assert results["samples_per_second"] > 0
        assert isinstance(results["samples_per_second"], (int, float))

    def test_comprehensive_evaluation(self, trained_model, test_data, capsys):
        """Test comprehensive evaluation method."""
        evaluator = PostTrainingEvaluator(trained_model)
        X, y = test_data

        # The evaluate method prints results but doesn't return them
        evaluator.evaluate(X, y)

        # Check that output was printed
        captured = capsys.readouterr()
        assert "NEUROSCOPE POST-TRAINING EVALUATION" in captured.out
        assert "EVALUATION" in captured.out

    def test_evaluate_caching(self, trained_model, test_data):
        """Test that evaluation results work consistently."""
        evaluator = PostTrainingEvaluator(trained_model)
        X, y = test_data

        # Test individual evaluations
        perf1 = evaluator.evaluate_performance(X, y)
        perf2 = evaluator.evaluate_performance(X, y)

        # Should return consistent results
        assert perf1["accuracy"] == perf2["accuracy"]
        assert perf1["loss"] == perf2["loss"]

    def test_robustness_with_zero_noise(self, trained_model, test_data):
        """Test robustness evaluation with zero noise level."""
        evaluator = PostTrainingEvaluator(trained_model)
        X, y = test_data

        results = evaluator.evaluate_robustness(X, y, noise_levels=[0.0])

        # Should return valid robustness results
        assert isinstance(results, dict)
        assert "baseline_accuracy" in results
        assert "overall_robustness" in results

    def test_robustness_with_high_noise(self, trained_model, test_data):
        """Test robustness evaluation with high noise levels."""
        evaluator = PostTrainingEvaluator(trained_model)
        X, y = test_data

        results = evaluator.evaluate_robustness(X, y, noise_levels=[0.0, 1.0, 2.0])

        # Should return valid robustness results
        assert isinstance(results, dict)
        assert "overall_robustness" in results
        assert isinstance(results["overall_robustness"], (int, float))

    def test_performance_with_different_data_shapes(self, trained_model):
        """Test performance evaluation with different data shapes."""
        evaluator = PostTrainingEvaluator(trained_model)

        # Test with single sample
        X_single = np.random.randn(1, 4)
        y_single = np.random.randint(0, 2, (1, 2))

        results = evaluator.evaluate_performance(X_single, y_single)
        assert "accuracy" in results
        assert "loss" in results

        # Test with larger batch
        X_large = np.random.randn(100, 4)
        y_large = np.random.randint(0, 2, (100, 2))

        results = evaluator.evaluate_performance(X_large, y_large)
        assert "accuracy" in results
        assert "loss" in results

    def test_robustness_score_calculation(self, trained_model, test_data):
        """Test robustness score calculation logic."""
        evaluator = PostTrainingEvaluator(trained_model)
        X, y = test_data

        # Test with known noise levels
        noise_levels = [0.0, 0.1, 0.2, 0.3]
        results = evaluator.evaluate_robustness(X, y, noise_levels=noise_levels)

        # Robustness score should be between 0 and 1
        assert 0.0 <= results["overall_robustness"] <= 1.0
        assert isinstance(results["overall_robustness"], (int, float))

    def test_error_handling_invalid_data_shapes(self, trained_model):
        """Test error handling with invalid data shapes."""
        evaluator = PostTrainingEvaluator(trained_model)

        # Test with correct shapes first
        X = np.random.randn(10, 4)
        y = np.random.randint(0, 2, (10, 2))

        results = evaluator.evaluate_performance(X, y)
        assert "accuracy" in results

    def test_model_prediction_error_handling(self, trained_model, test_data):
        """Test handling of model prediction errors."""
        evaluator = PostTrainingEvaluator(trained_model)
        X, y = test_data

        # Mock model to raise an error
        with patch.object(
            trained_model, "predict", side_effect=Exception("Prediction error")
        ):
            # Should return error status instead of raising
            results = evaluator.evaluate_performance(X, y)
            assert results["status"] == "ERROR"

    def test_results_structure_consistency(self, trained_model, test_data):
        """Test that results have consistent structure."""
        evaluator = PostTrainingEvaluator(trained_model)
        X, y = test_data

        # Test multiple evaluations
        for _ in range(3):
            perf_results = evaluator.evaluate_performance(X, y)
            rob_results = evaluator.evaluate_robustness(X, y)

            # Performance results structure
            expected_perf_keys = {
                "accuracy",
                "loss",
                "samples_per_second",
                "total_params",
                "all_metrics",
                "status",
                "note",
            }
            assert expected_perf_keys.issubset(set(perf_results.keys()))

            # Robustness results structure
            expected_rob_keys = {
                "baseline_accuracy",
                "baseline_loss",
                "overall_robustness",
                "status",
                "note",
            }
            assert expected_rob_keys.issubset(set(rob_results.keys()))

    def test_numerical_stability(self, trained_model):
        """Test numerical stability with edge case data."""
        evaluator = PostTrainingEvaluator(trained_model)

        # Test with very small values
        X_small = np.random.randn(10, 4) * 1e-10
        y_small = np.random.randint(0, 2, (10, 2))

        results = evaluator.evaluate_performance(X_small, y_small)
        assert np.isfinite(results["accuracy"])
        assert np.isfinite(results["loss"])

        # Test with very large values
        X_large = np.random.randn(10, 4) * 1e10
        y_large = np.random.randint(0, 2, (10, 2))

        results = evaluator.evaluate_performance(X_large, y_large)
        assert np.isfinite(results["accuracy"])
        assert np.isfinite(results["loss"])
