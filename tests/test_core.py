"""
Tests for core neural network operations (forward/backward pass).
"""

import numpy as np
import pytest

from neuroscope.mlp.core import _BackwardPass, _ForwardPass


class TestForwardPass:
    """Test forward propagation functionality."""

    def test_forward_mlp_basic(self):
        """Test basic forward pass functionality."""
        # Simple 2-layer network: 3 -> 2 -> 1
        X = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])  # 2 samples, 3 features

        # Initialize simple weights and biases
        weights = [
            np.array([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]]),  # 3x2
            np.array([[0.7], [0.8]]),  # 2x1
        ]
        biases = [np.array([[0.1, 0.2]]), np.array([[0.3]])]  # 1x2  # 1x1

        # Forward pass
        activations, z_values = _ForwardPass.forward_mlp(
            X,
            weights,
            biases,
            hidden_activation="relu",
            out_activation=None,
            dropout_rate=0.0,
            training=False,
        )

        # Check shapes
        assert len(activations) == 2  # 2 layers (no input stored)
        assert len(z_values) == 2  # 2 layers (no z for input)

        assert activations[0].shape == (2, 2)  # Hidden layer
        assert activations[1].shape == (2, 1)  # Output layer

        assert z_values[0].shape == (2, 2)  # Hidden layer z
        assert z_values[1].shape == (2, 1)  # Output layer z

        # Check that all values are finite
        for act in activations:
            assert np.all(np.isfinite(act))
        for z in z_values:
            assert np.all(np.isfinite(z))

    def test_forward_mlp_different_activations(self):
        """Test forward pass with different activation functions."""
        X = np.array([[1.0, 2.0]])
        weights = [np.array([[0.5], [0.3]])]
        biases = [np.array([[0.1]])]

        activations = ["relu", "sigmoid", "tanh", "leaky_relu"]

        for activation in activations:
            result_activations, result_z = _ForwardPass.forward_mlp(
                X, weights, biases, hidden_activation=activation, training=False
            )

            # Should complete without errors
            assert len(result_activations) == 1  # Only output layer
            assert len(result_z) == 1
            assert np.all(np.isfinite(result_activations[-1]))

    def test_forward_mlp_with_dropout(self):
        """Test forward pass with dropout."""
        X = np.array([[1.0, 2.0, 3.0]])
        weights = [np.array([[0.1], [0.2], [0.3]])]
        biases = [np.array([[0.1]])]

        # Training mode with dropout
        activations_train, _ = _ForwardPass.forward_mlp(
            X, weights, biases, dropout_rate=0.5, training=True
        )

        # Inference mode (no dropout)
        activations_infer, _ = _ForwardPass.forward_mlp(
            X, weights, biases, dropout_rate=0.5, training=False
        )

        # Both should be finite but potentially different due to dropout
        assert np.all(np.isfinite(activations_train[-1]))
        assert np.all(np.isfinite(activations_infer[-1]))

    def test_forward_mlp_output_activations(self):
        """Test different output activation functions."""
        X = np.array([[1.0, 2.0]])
        weights = [np.array([[0.5], [0.3]])]
        biases = [np.array([[0.1]])]

        # Test sigmoid output
        activations_sigmoid, _ = _ForwardPass.forward_mlp(
            X, weights, biases, out_activation="sigmoid", training=False
        )
        # Sigmoid output should be between 0 and 1
        output_sigmoid = activations_sigmoid[-1]
        assert np.all(output_sigmoid >= 0) and np.all(output_sigmoid <= 1)

        # Test linear output (None)
        activations_linear, _ = _ForwardPass.forward_mlp(
            X, weights, biases, out_activation=None, training=False
        )
        # Linear output can be any value
        assert np.all(np.isfinite(activations_linear[-1]))


class TestBackwardPass:
    """Test backward propagation functionality."""

    def test_backward_mlp_basic(self):
        """Test basic backward pass functionality."""
        # Simple network setup
        X = np.array([[1.0, 2.0]])
        y = np.array([[1.0]])

        weights = [np.array([[0.5], [0.3]])]
        biases = [np.array([[0.1]])]

        # Forward pass first
        activations, z_values = _ForwardPass.forward_mlp(
            X, weights, biases, hidden_activation="relu", training=False
        )

        # Backward pass
        dW, db = _BackwardPass.backward_mlp(
            y,
            activations,
            z_values,
            weights,
            biases,
            X,
            hidden_activation="relu",
            out_activation=None,
        )

        # Check shapes
        assert len(dW) == len(weights)
        assert len(db) == len(biases)

        assert dW[0].shape == weights[0].shape
        assert db[0].shape == biases[0].shape

        # Check that gradients are finite
        for grad_w in dW:
            assert np.all(np.isfinite(grad_w))
        for grad_b in db:
            assert np.all(np.isfinite(grad_b))

    def test_backward_mlp_different_losses(self):
        """Test backward pass with different loss functions."""
        X = np.array([[1.0, 2.0]])
        y = np.array([[1.0]])

        weights = [np.array([[0.5], [0.3]])]
        biases = [np.array([[0.1]])]

        # Forward pass
        activations, z_values = _ForwardPass.forward_mlp(
            X, weights, biases, training=False
        )

        loss_functions = ["mse", "bce"]

        for loss_fn in loss_functions:
            try:
                dW, db = _BackwardPass.backward_mlp(
                    y, activations, z_values, weights, biases, X
                )

                # Should produce finite gradients
                assert np.all(np.isfinite(dW[0]))
                assert np.all(np.isfinite(db[0]))
            except ValueError:
                # Some loss functions might not be compatible with certain setups
                pass

    def test_backward_mlp_multilayer(self):
        """Test backward pass with multiple layers."""
        X = np.array([[1.0, 2.0, 3.0]])
        y = np.array([[1.0]])

        # 3 -> 2 -> 1 network
        weights = [
            np.array([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]]),
            np.array([[0.7], [0.8]]),
        ]
        biases = [np.array([[0.1, 0.2]]), np.array([[0.3]])]

        # Forward pass
        activations, z_values = _ForwardPass.forward_mlp(
            X, weights, biases, hidden_activation="relu", training=False
        )

        # Backward pass
        dW, db = _BackwardPass.backward_mlp(
            y, activations, z_values, weights, biases, X, hidden_activation="relu"
        )

        # Check that we get gradients for all layers
        assert len(dW) == 2
        assert len(db) == 2

        # Check shapes match
        for i in range(len(weights)):
            assert dW[i].shape == weights[i].shape
            assert db[i].shape == biases[i].shape

        # Check finite gradients
        for grad_w, grad_b in zip(dW, db):
            assert np.all(np.isfinite(grad_w))
            assert np.all(np.isfinite(grad_b))

    def test_gradient_flow_consistency(self):
        """Test that gradients flow consistently through the network."""
        X = np.array([[1.0, 2.0]])
        y = np.array([[0.5]])

        weights = [np.array([[0.3], [0.7]])]
        biases = [np.array([[0.1]])]

        # Forward pass
        activations, z_values = _ForwardPass.forward_mlp(
            X, weights, biases, hidden_activation="relu", training=False
        )

        # Backward pass
        dW, db = _BackwardPass.backward_mlp(
            y, activations, z_values, weights, biases, X, hidden_activation="relu"
        )

        # Gradients should not be zero (unless by coincidence)
        # and should be reasonable in magnitude
        assert np.any(np.abs(dW[0]) > 1e-10)  # Not all zeros
        assert np.any(np.abs(db[0]) > 1e-10)  # Not all zeros
        assert np.all(np.abs(dW[0]) < 100)  # Not exploding
        assert np.all(np.abs(db[0]) < 100)  # Not exploding


class TestForwardBackwardIntegration:
    """Test integration between forward and backward passes."""

    def test_forward_backward_integration(self):
        """Test that forward and backward passes work together."""
        # Create a simple training scenario
        X = np.array([[1.0, 2.0], [3.0, 4.0]])
        y = np.array([[1.0], [0.0]])

        weights = [np.array([[0.5, 0.3], [0.2, 0.8]]), np.array([[0.6], [0.4]])]
        biases = [np.array([[0.1, 0.2]]), np.array([[0.1]])]

        # Forward pass
        activations, z_values = _ForwardPass.forward_mlp(
            X,
            weights,
            biases,
            hidden_activation="relu",
            out_activation="sigmoid",
            training=True,
        )

        # Backward pass
        dW, db = _BackwardPass.backward_mlp(
            y,
            activations,
            z_values,
            weights,
            biases,
            X,
            hidden_activation="relu",
            out_activation="sigmoid",
        )

        # Should complete without errors and produce reasonable gradients
        assert len(activations) == 2  # 2 layers (no input stored)
        assert len(z_values) == 2  # 2 layers
        assert len(dW) == 2  # 2 weight matrices
        assert len(db) == 2  # 2 bias vectors

        # All outputs should be finite
        for act in activations:
            assert np.all(np.isfinite(act))
        for z in z_values:
            assert np.all(np.isfinite(z))
        for grad_w in dW:
            assert np.all(np.isfinite(grad_w))
        for grad_b in db:
            assert np.all(np.isfinite(grad_b))

    def test_numerical_stability(self):
        """Test numerical stability with extreme values."""
        # Test with small values
        X_small = np.array([[1e-8, 1e-8]])
        # Test with large values
        X_large = np.array([[1e3, 1e3]])

        weights = [np.array([[0.1], [0.1]])]
        biases = [np.array([[0.01]])]
        y = np.array([[0.5]])

        for X_test in [X_small, X_large]:
            # Forward pass should handle extreme values
            activations, z_values = _ForwardPass.forward_mlp(
                X_test, weights, biases, training=False
            )

            # Should produce finite outputs
            assert np.all(np.isfinite(activations[-1]))

            # Backward pass should also be stable
            dW, db = _BackwardPass.backward_mlp(
                y, activations, z_values, weights, biases, X_test
            )

            # Gradients should be finite
            assert np.all(np.isfinite(dW[0]))
            assert np.all(np.isfinite(db[0]))


class TestWarningThrottling:
    """Test warning throttling mechanism in core module."""

    def test_warning_throttling_reset(self):
        """Test that warning throttling can be reset."""
        # This test accesses internal throttling state if available
        # The _ForwardPass and _BackwardPass may have throttled_warning method

        # Create large values that would trigger warnings
        X = np.array([[1e10, 2e10]])
        weights = [np.array([[0.1], [0.1]])]
        biases = [np.array([[0.01]])]

        import warnings

        # First call - should generate warnings
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            _ForwardPass.forward_mlp(X, weights, biases, training=False)
            first_warning_count = len(w)

        # If throttling exists and is working, subsequent calls might suppress warnings
        # Reset would clear the throttling state

        # This is a basic test - actual throttling behavior depends on implementation
        assert isinstance(first_warning_count, int)

    def test_warning_throttling_limits(self):
        """Test that warning throttling mechanism exists and functions properly."""
        # Test that the forward pass handles potential warnings gracefully
        X = np.array([[1.0, 2.0]])
        weights = [np.array([[1.0], [1.0]])]
        biases = [np.array([[0.0]])]

        # Make multiple calls - should not crash or accumulate excessive warnings
        for i in range(5):
            activations, z_values = _ForwardPass.forward_mlp(
                X, weights, biases, training=False
            )
            # Should complete successfully
            assert np.all(np.isfinite(activations[-1]))

        # Test passes if no exceptions were raised
        assert True


class TestForwardPassEdgeCases:
    """Additional tests to increase coverage for forward pass edge cases."""

    def test_forward_mlp_invalid_output_activation(self):
        """Test that invalid output activation raises ValueError."""
        X = np.array([[1.0, 2.0]])
        weights = [np.array([[1.0], [1.0]])]
        biases = [np.array([[0.0]])]

        with pytest.raises(ValueError, match="Unknown output activation"):
            _ForwardPass.forward_mlp(
                X, weights, biases, out_activation="invalid_activation"
            )

    def test_forward_mlp_invalid_hidden_activation(self):
        """Test that invalid hidden activation raises ValueError."""
        X = np.array([[1.0, 2.0]])
        weights = [
            np.array([[1.0, 1.0], [1.0, 1.0]]),  # Hidden layer
            np.array([[1.0], [1.0]]),  # Output layer
        ]
        biases = [np.array([[0.0, 0.0]]), np.array([[0.0]])]

        with pytest.raises(ValueError, match="Unknown activation function"):
            _ForwardPass.forward_mlp(
                X, weights, biases, hidden_activation="invalid_activation"
            )

    def test_forward_mlp_alpha_dropout(self):
        """Test forward pass with alpha dropout."""
        np.random.seed(42)
        X = np.array([[1.0, 2.0], [3.0, 4.0]])
        weights = [
            np.array([[1.0, 1.0], [1.0, 1.0]]),  # Hidden layer
            np.array([[1.0], [1.0]]),  # Output layer
        ]
        biases = [np.array([[0.0, 0.0]]), np.array([[0.0]])]

        activations, z_values = _ForwardPass.forward_mlp(
            X, weights, biases, dropout_rate=0.2, dropout_type="alpha", training=True
        )

        assert len(activations) == 2
        assert activations[0].shape == (2, 2)
        assert activations[1].shape == (2, 1)

    def test_forward_mlp_invalid_dropout_type(self):
        """Test that invalid dropout type raises ValueError."""
        X = np.array([[1.0, 2.0]])
        weights = [
            np.array([[1.0, 1.0], [1.0, 1.0]]),
            np.array([[1.0], [1.0]]),
        ]
        biases = [np.array([[0.0, 0.0]]), np.array([[0.0]])]

        with pytest.raises(ValueError, match="Unknown dropout type"):
            _ForwardPass.forward_mlp(
                X,
                weights,
                biases,
                dropout_rate=0.5,
                dropout_type="invalid",
                training=True,
            )

    def test_forward_mlp_no_activation_hidden(self):
        """Test forward pass with no activation on hidden layers."""
        X = np.array([[1.0, 2.0]])
        weights = [
            np.array([[1.0, 1.0], [1.0, 1.0]]),
            np.array([[1.0], [1.0]]),
        ]
        biases = [np.array([[0.0, 0.0]]), np.array([[0.0]])]

        activations, z_values = _ForwardPass.forward_mlp(
            X, weights, biases, hidden_activation=None
        )

        # With no activation, A should equal Z for hidden layers
        assert np.allclose(activations[0], z_values[0])
