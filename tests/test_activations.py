"""
Test suite for Activation Functions module.
Tests all activation functions and their derivatives.
"""

import numpy as np
import pytest

from neuroscope.mlp.activations import ActivationFunctions


class TestActivationFunctions:
    """Test suite for activation functions."""

    def test_sigmoid_basic(self):
        """Test sigmoid activation function."""
        x = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])
        result = ActivationFunctions.sigmoid(x)

        # Check output range (0, 1)
        assert np.all(result > 0)
        assert np.all(result < 1)
        assert result.shape == x.shape

        # Check specific values
        assert abs(result[2] - 0.5) < 1e-10  # sigmoid(0) = 0.5
        assert result[4] > result[3] > result[2] > result[1] > result[0]  # monotonic

    def test_sigmoid_extreme_values(self):
        """Test sigmoid with extreme values (numerical stability)."""
        # Large positive values
        x_large = np.array([100.0, 500.0, 1000.0])
        result_large = ActivationFunctions.sigmoid(x_large)

        assert np.all(np.isfinite(result_large))
        assert np.all(result_large > 0.99)  # Should be close to 1

        # Large negative values
        x_small = np.array([-100.0, -500.0, -1000.0])
        result_small = ActivationFunctions.sigmoid(x_small)

        assert np.all(np.isfinite(result_small))
        assert np.all(result_small < 0.01)  # Should be close to 0

    def test_sigmoid_derivative(self):
        """Test sigmoid derivative function."""
        x = np.array([-1.0, 0.0, 1.0])

        if hasattr(ActivationFunctions, "sigmoid_derivative"):
            derivative = ActivationFunctions.sigmoid_derivative(x)

            assert derivative.shape == x.shape
            assert np.all(derivative >= 0)  # Derivative is always non-negative
            assert np.all(derivative <= 0.25)  # Max value is 0.25 at x=0

            # Check that derivative at 0 is 0.25
            assert abs(derivative[1] - 0.25) < 1e-10

    def test_tanh_basic(self):
        """Test tanh activation function."""
        x = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])
        result = ActivationFunctions.tanh(x)

        # Check output range (-1, 1)
        assert np.all(result > -1)
        assert np.all(result < 1)
        assert result.shape == x.shape

        # Check specific values
        assert abs(result[2] - 0.0) < 1e-10  # tanh(0) = 0
        assert result[4] > result[3] > result[2] > result[1] > result[0]  # monotonic

    def test_tanh_derivative(self):
        """Test tanh derivative if available."""
        if hasattr(ActivationFunctions, "tanh_derivative"):
            x = np.array([-1.0, 0.0, 1.0])
            derivative = ActivationFunctions.tanh_derivative(x)

            assert derivative.shape == x.shape
            assert np.all(derivative >= 0)  # Derivative is always non-negative
            assert np.all(derivative <= 1.0)  # Max value is 1 at x=0

            # Check that derivative at 0 is 1
            assert abs(derivative[1] - 1.0) < 1e-10

    def test_relu_basic(self):
        """Test ReLU activation function."""
        if hasattr(ActivationFunctions, "relu"):
            x = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])
            result = ActivationFunctions.relu(x)

            # Check ReLU properties
            assert result.shape == x.shape
            assert np.all(result >= 0)  # Always non-negative

            # Check specific values
            assert result[0] == 0.0  # relu(-2) = 0
            assert result[1] == 0.0  # relu(-1) = 0
            assert result[2] == 0.0  # relu(0) = 0
            assert result[3] == 1.0  # relu(1) = 1
            assert result[4] == 2.0  # relu(2) = 2

    def test_leaky_relu_basic(self):
        """Test Leaky ReLU activation function."""
        if hasattr(ActivationFunctions, "leaky_relu"):
            x = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])
            result = ActivationFunctions.leaky_relu(x)

            assert result.shape == x.shape

            # Positive values should be unchanged
            assert result[3] == 1.0
            assert result[4] == 2.0

            # Negative values should be small but non-zero
            assert result[0] < 0 and result[0] > -2.0
            assert result[1] < 0 and result[1] > -1.0

            # Zero should remain zero
            assert result[2] == 0.0

    def test_selu_basic(self):
        """Test SELU activation function."""
        if hasattr(ActivationFunctions, "selu"):
            x = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])
            result = ActivationFunctions.selu(x)

            assert result.shape == x.shape
            assert np.all(np.isfinite(result))

            # SELU should be identity-like for positive values
            assert result[3] > 0
            assert result[4] > result[3]

            # SELU has specific behavior for negative values
            assert result[0] < 0
            assert result[1] < 0

    def test_softmax_basic(self):
        """Test softmax activation function."""
        # 2D input (batch of samples)
        x = np.array([[1.0, 2.0, 3.0], [1.0, 1.0, 1.0], [0.0, 0.0, 0.0]])

        result = ActivationFunctions.softmax(x)

        # Check output properties
        assert result.shape == x.shape
        assert np.all(result >= 0)  # All values non-negative
        assert np.all(result <= 1)  # All values <= 1

        # Check that each row sums to 1
        row_sums = np.sum(result, axis=1)
        np.testing.assert_allclose(row_sums, 1.0, rtol=1e-10)

        # Check that larger input values get larger probabilities
        assert result[0, 2] > result[0, 1] > result[0, 0]  # 3 > 2 > 1

        # Equal inputs should give equal probabilities
        np.testing.assert_allclose(result[1], [1 / 3, 1 / 3, 1 / 3], rtol=1e-10)

    def test_softmax_numerical_stability(self):
        """Test softmax numerical stability with large values."""
        # Large values that could cause overflow
        x = np.array([[100.0, 200.0, 300.0]])
        result = ActivationFunctions.softmax(x)

        assert np.all(np.isfinite(result))
        assert np.all(result >= 0)
        np.testing.assert_allclose(np.sum(result, axis=1), 1.0, rtol=1e-10)

    def test_available_activations(self):
        """Test what activation functions are available."""
        activations = [
            attr
            for attr in dir(ActivationFunctions)
            if callable(getattr(ActivationFunctions, attr)) and not attr.startswith("_")
        ]

        # Should have basic activations
        assert len(activations) > 0
        assert "sigmoid" in activations
        assert "tanh" in activations
        assert "softmax" in activations

        print(f"Available activations: {activations}")

    @pytest.mark.parametrize("activation", ["sigmoid", "tanh"])
    def test_activation_derivatives_numerical(self, activation):
        """Test activation derivatives numerically."""
        if not hasattr(ActivationFunctions, f"{activation}_derivative"):
            pytest.skip(f"{activation}_derivative not available")

        activation_fn = getattr(ActivationFunctions, activation)
        derivative_fn = getattr(ActivationFunctions, f"{activation}_derivative")

        x = np.array([0.0, 1.0, -1.0])
        h = 1e-7

        # Compute numerical derivative
        numerical_grad = np.zeros_like(x)
        for i in range(len(x)):
            x_plus = x.copy()
            x_minus = x.copy()
            x_plus[i] += h
            x_minus[i] -= h

            numerical_grad[i] = (
                activation_fn(x_plus[i]) - activation_fn(x_minus[i])
            ) / (2 * h)

        # Compute analytical derivative
        analytical_grad = derivative_fn(x)

        # Compare
        np.testing.assert_allclose(analytical_grad, numerical_grad, rtol=1e-5)

    def test_activation_output_shapes(self):
        """Test that activations preserve input shapes."""
        shapes_to_test = [
            (5,),  # 1D
            (3, 4),  # 2D
            (2, 3, 4),  # 3D
        ]

        for shape in shapes_to_test:
            x = np.random.randn(*shape)

            # Test sigmoid
            result_sigmoid = ActivationFunctions.sigmoid(x)
            assert result_sigmoid.shape == shape

            # Test tanh
            result_tanh = ActivationFunctions.tanh(x)
            assert result_tanh.shape == shape

            # Test softmax (only for 2D)
            if len(shape) == 2:
                result_softmax = ActivationFunctions.softmax(x)
                assert result_softmax.shape == shape

    def test_activation_edge_cases(self):
        """Test activations with edge cases."""
        # Test with zeros
        x_zeros = np.zeros(5)

        sigmoid_zeros = ActivationFunctions.sigmoid(x_zeros)
        np.testing.assert_allclose(sigmoid_zeros, 0.5)

        tanh_zeros = ActivationFunctions.tanh(x_zeros)
        np.testing.assert_allclose(tanh_zeros, 0.0)

        # Test with single value
        single_val = np.array([1.0])

        sigmoid_single = ActivationFunctions.sigmoid(single_val)
        assert sigmoid_single.shape == (1,)
        assert 0 < sigmoid_single[0] < 1

        tanh_single = ActivationFunctions.tanh(single_val)
        assert tanh_single.shape == (1,)
        assert -1 < tanh_single[0] < 1

    def test_monotonicity(self):
        """Test that activation functions are monotonic where expected."""
        x = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])

        # Sigmoid should be monotonically increasing
        sigmoid_result = ActivationFunctions.sigmoid(x)
        assert np.all(np.diff(sigmoid_result) > 0)

        # Tanh should be monotonically increasing
        tanh_result = ActivationFunctions.tanh(x)
        assert np.all(np.diff(tanh_result) > 0)

        # ReLU should be monotonically non-decreasing
        if hasattr(ActivationFunctions, "relu"):
            relu_result = ActivationFunctions.relu(x)
            assert np.all(np.diff(relu_result) >= 0)

    def test_selu_derivative(self):
        """Test SELU derivative function."""
        z = np.array([-2.0, -0.5, 0.0, 0.5, 2.0])
        activation = ActivationFunctions.selu(z)
        derivative = ActivationFunctions.selu_derivative(activation)

        assert derivative.shape == z.shape
        assert np.all(derivative > 0)  # SELU derivative is always positive

    def test_inverted_dropout_training_false(self):
        """Test inverted dropout when training=False returns input unchanged."""
        A = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        result = ActivationFunctions.inverted_dropout(A, rate=0.5, training=False)

        np.testing.assert_array_equal(result, A)

    def test_inverted_dropout_rate_zero(self):
        """Test inverted dropout with rate=0 returns input unchanged."""
        A = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        result = ActivationFunctions.inverted_dropout(A, rate=0.0, training=True)

        np.testing.assert_array_equal(result, A)

    def test_inverted_dropout_training_true(self):
        """Test inverted dropout applies dropout during training."""
        np.random.seed(42)
        A = np.ones((100, 100)) * 2.0
        rate = 0.5
        result = ActivationFunctions.inverted_dropout(A, rate=rate, training=True)

        # Some values should be zeroed out
        assert np.any(result == 0.0)
        # Non-zero values should be scaled up
        non_zero_mask = result != 0.0
        assert np.all(result[non_zero_mask] > A[non_zero_mask])

    def test_alpha_dropout_not_training(self):
        """Test alpha dropout when training=False returns input unchanged."""
        A = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        result = ActivationFunctions.alpha_dropout(A, rate=0.5, training=False)

        np.testing.assert_array_equal(result, A)

    def test_alpha_dropout_rate_zero(self):
        """Test alpha dropout with rate=0 returns input unchanged."""
        A = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        result = ActivationFunctions.alpha_dropout(A, rate=0.0, training=True)

        np.testing.assert_array_equal(result, A)

    def test_alpha_dropout_applies_during_training(self):
        """Test alpha dropout applies transformations during training."""
        np.random.seed(42)
        A = np.ones((100, 100))
        rate = 0.5
        result = ActivationFunctions.alpha_dropout(A, rate=rate, training=True)

        # Output should differ from input when training
        assert not np.array_equal(result, A)
        # Shape should be preserved
        assert result.shape == A.shape
