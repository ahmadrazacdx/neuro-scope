"""
Test suite for Loss Functions module.
Tests all loss functions and their mathematical properties.
"""

import numpy as np
import pytest

from neuroscope.mlp.losses import LossFunctions


class TestLossFunctions:
    """Test suite for loss functions."""

    def test_mse_basic_functionality(self):
        """Test Mean Squared Error basic functionality."""
        y_true = np.array([1.0, 2.0, 3.0, 4.0])
        y_pred = np.array([1.1, 2.1, 2.9, 4.1])

        loss = LossFunctions.mse(y_true, y_pred)

        # Basic properties
        assert isinstance(loss, (int, float))
        assert loss >= 0
        assert np.isfinite(loss)

        # Manual calculation
        expected = np.mean((y_true - y_pred) ** 2)
        assert abs(loss - expected) < 1e-10

    def test_mse_perfect_predictions(self):
        """Test MSE with perfect predictions."""
        y_true = np.array([1.0, 2.0, 3.0])

        loss = LossFunctions.mse(y_true, y_true)
        assert loss == 0.0

    def test_mse_with_different_shapes(self):
        """Test MSE with different input shapes."""
        # 1D arrays
        loss1 = LossFunctions.mse([1.0, 2.0], [1.1, 2.1])
        assert isinstance(loss1, (int, float))

        # 2D arrays (flattened internally)
        y_true_2d = np.array([[1.0], [2.0]])
        y_pred_2d = np.array([[1.1], [2.1]])
        loss2 = LossFunctions.mse(y_true_2d, y_pred_2d)
        assert isinstance(loss2, (int, float))

    def test_bce_basic_functionality(self):
        """Test Binary Cross Entropy basic functionality."""
        y_true = np.array([0.0, 1.0, 1.0, 0.0])
        y_pred = np.array([0.1, 0.9, 0.8, 0.2])

        loss = LossFunctions.bce(y_true, y_pred)

        # Basic properties
        assert isinstance(loss, (int, float))
        assert loss >= 0
        assert np.isfinite(loss)

    def test_bce_perfect_predictions(self):
        """Test BCE with perfect predictions."""
        y_true = np.array([0.0, 1.0, 0.0, 1.0])
        y_pred = np.array([0.0, 1.0, 0.0, 1.0])

        loss = LossFunctions.bce(y_true, y_pred)
        assert loss < 1e-10  # Should be very close to 0

    def test_bce_numerical_stability(self):
        """Test BCE numerical stability with extreme values."""
        y_true = np.array([0.0, 1.0])

        # Test with predictions very close to 0 and 1
        y_pred = np.array([1e-15, 1.0 - 1e-15])
        loss = LossFunctions.bce(y_true, y_pred)
        assert np.isfinite(loss)

        # Test with exact 0 and 1 (should be clipped)
        y_pred_extreme = np.array([0.0, 1.0])
        loss_extreme = LossFunctions.bce(y_true, y_pred_extreme)
        assert np.isfinite(loss_extreme)

    def test_cce_basic_functionality(self):
        """Test Categorical Cross Entropy basic functionality."""
        # Test with sparse labels
        y_true_sparse = np.array([0, 1, 2])
        y_pred = np.array([[0.9, 0.05, 0.05], [0.1, 0.8, 0.1], [0.1, 0.1, 0.8]])

        loss = LossFunctions.cce(y_true_sparse, y_pred)

        # Basic properties
        assert isinstance(loss, (int, float))
        assert loss >= 0
        assert np.isfinite(loss)

    def test_cce_with_onehot_labels(self):
        """Test CCE with one-hot encoded labels."""
        y_true_onehot = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
        y_pred = np.array([[0.9, 0.05, 0.05], [0.1, 0.8, 0.1], [0.1, 0.1, 0.8]])

        loss = LossFunctions.cce(y_true_onehot, y_pred)

        assert isinstance(loss, (int, float))
        assert loss >= 0
        assert np.isfinite(loss)

    def test_cce_perfect_predictions(self):
        """Test CCE with perfect predictions."""
        y_true = np.array([0, 1, 2])
        y_pred = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])

        loss = LossFunctions.cce(y_true, y_pred)
        assert loss < 1e-10  # Should be very close to 0

    def test_cce_numerical_stability(self):
        """Test CCE numerical stability."""
        y_true = np.array([0, 1])

        # Test with very small probabilities
        y_pred = np.array([[1e-15, 1.0 - 1e-15], [1.0 - 1e-15, 1e-15]])

        loss = LossFunctions.cce(y_true, y_pred)
        assert np.isfinite(loss)

    def test_mse_with_regularization(self):
        """Test MSE with L2 regularization."""
        y_true = np.array([1.0, 2.0, 3.0])
        y_pred = np.array([1.1, 2.1, 2.9])

        # Create some dummy weights
        weights = [np.array([[0.5, 0.3], [0.2, 0.4]]), np.array([[0.1], [0.2]])]

        loss_reg = LossFunctions.mse_with_reg(y_true, y_pred, weights, lamda=0.01)
        loss_no_reg = LossFunctions.mse(y_true, y_pred)

        # Regularized loss should be higher
        assert loss_reg > loss_no_reg
        assert np.isfinite(loss_reg)

    def test_cce_with_regularization(self):
        """Test CCE with L2 regularization."""
        y_true = np.array([0, 1])
        y_pred = np.array([[0.9, 0.1], [0.2, 0.8]])

        # Create some dummy weights
        weights = [np.array([[0.5, 0.3]]), np.array([[0.1, 0.2]])]

        loss_reg = LossFunctions.cce_with_reg(y_true, y_pred, weights, lamda=0.01)
        loss_no_reg = LossFunctions.cce(y_true, y_pred)

        # Regularized loss should be higher
        assert loss_reg > loss_no_reg
        assert np.isfinite(loss_reg)

    def test_loss_function_consistency(self):
        """Test consistency between different loss calculations."""
        # MSE should be symmetric for perfect predictions
        y = np.array([1.0, 2.0, 3.0])
        loss1 = LossFunctions.mse(y, y)
        loss2 = LossFunctions.mse(y, y)
        assert loss1 == loss2

        # BCE should be consistent
        y_binary = np.array([0.0, 1.0])
        y_pred_binary = np.array([0.1, 0.9])
        bce1 = LossFunctions.bce(y_binary, y_pred_binary)
        bce2 = LossFunctions.bce(y_binary, y_pred_binary)
        assert bce1 == bce2

    def test_loss_increases_with_error(self):
        """Test that loss increases as predictions get worse."""
        y_true = np.array([1.0, 2.0, 3.0])

        # Small error
        y_pred_small = np.array([1.01, 2.01, 3.01])
        loss_small = LossFunctions.mse(y_true, y_pred_small)

        # Large error
        y_pred_large = np.array([1.1, 2.1, 3.1])
        loss_large = LossFunctions.mse(y_true, y_pred_large)

        assert loss_large > loss_small

    def test_edge_cases(self):
        """Test edge cases for loss functions."""
        # Empty arrays (if supported)
        try:
            import warnings

            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=RuntimeWarning)
                y_empty = np.array([])
                loss = LossFunctions.mse(y_empty, y_empty)
                # If it works, should be 0 or NaN
                assert np.isnan(loss) or loss == 0
        except (ValueError, ZeroDivisionError):
            # This is acceptable behavior
            pass

        # Single value
        y_single = np.array([1.0])
        y_pred_single = np.array([1.1])
        loss_single = LossFunctions.mse(y_single, y_pred_single)
        assert np.isfinite(loss_single)
        assert loss_single >= 0

    def test_negative_values(self):
        """Test loss functions with negative values."""
        y_true = np.array([-1.0, -2.0, 0.0, 1.0, 2.0])
        y_pred = np.array([-0.9, -2.1, 0.1, 1.1, 1.9])

        loss = LossFunctions.mse(y_true, y_pred)
        assert np.isfinite(loss)
        assert loss >= 0

    def test_regularization_lambda_effect(self):
        """Test that regularization lambda affects loss magnitude."""
        y_true = np.array([1.0, 2.0])
        y_pred = np.array([1.1, 2.1])
        weights = [np.array([[1.0, 0.5], [0.3, 0.8]])]

        # Different lambda values
        loss_small_lambda = LossFunctions.mse_with_reg(
            y_true, y_pred, weights, lamda=0.001
        )
        loss_large_lambda = LossFunctions.mse_with_reg(
            y_true, y_pred, weights, lamda=0.1
        )

        # Higher lambda should give higher total loss
        assert loss_large_lambda > loss_small_lambda

    @pytest.mark.parametrize("eps", [1e-12, 1e-10, 1e-8])
    def test_bce_epsilon_parameter(self, eps):
        """Test BCE with different epsilon values."""
        y_true = np.array([0.0, 1.0])
        y_pred = np.array([0.0, 1.0])  # Extreme values

        loss = LossFunctions.bce(y_true, y_pred, eps=eps)
        assert np.isfinite(loss)
        assert loss >= 0

    def test_bce_with_reg_basic(self):
        """Test Binary Cross Entropy with L2 regularization."""
        y_true = np.array([0.0, 1.0, 1.0, 0.0])
        y_pred = np.array([0.1, 0.9, 0.8, 0.2])
        weights = [np.array([[1.0, 0.5], [0.3, 0.8]])]
        lamda = 0.01

        loss = LossFunctions.bce_with_reg(y_true, y_pred, weights, lamda)

        # Basic properties
        assert isinstance(loss, (int, float))
        assert loss >= 0
        assert np.isfinite(loss)

        # Should be greater than BCE alone
        bce_loss = LossFunctions.bce(y_true, y_pred)
        assert loss > bce_loss

    def test_bce_with_reg_zero_lambda(self):
        """Test BCE with reg when lambda=0 equals plain BCE."""
        y_true = np.array([0.0, 1.0, 1.0, 0.0])
        y_pred = np.array([0.1, 0.9, 0.8, 0.2])
        weights = [np.array([[1.0, 0.5], [0.3, 0.8]])]

        loss_with_reg = LossFunctions.bce_with_reg(y_true, y_pred, weights, lamda=0.0)
        loss_plain = LossFunctions.bce(y_true, y_pred)

        assert abs(loss_with_reg - loss_plain) < 1e-10

    def test_bce_with_reg_large_weights(self):
        """Test BCE with reg with larger weights increases regularization term."""
        y_true = np.array([0.0, 1.0])
        y_pred = np.array([0.1, 0.9])
        weights_small = [np.array([[0.1, 0.1]])]
        weights_large = [np.array([[10.0, 10.0]])]
        lamda = 0.1

        loss_small = LossFunctions.bce_with_reg(y_true, y_pred, weights_small, lamda)
        loss_large = LossFunctions.bce_with_reg(y_true, y_pred, weights_large, lamda)

        assert loss_large > loss_small

    def test_mse_extreme_and_zero_targets(self):
        """MSE should handle zero targets and very large target values robustly."""
        # Zero targets
        y_zero = np.zeros(5)
        y_pred_small = np.zeros(5) + 1e-6
        loss_zero = LossFunctions.mse(y_zero, y_pred_small)
        assert np.isfinite(loss_zero) and loss_zero >= 0

        # Very large target values
        y_large = np.ones(5) * 1e8
        y_pred_large = y_large + np.random.randn(5) * 1e4
        loss_large = LossFunctions.mse(y_large, y_pred_large)
        assert np.isfinite(loss_large) and loss_large > 0
