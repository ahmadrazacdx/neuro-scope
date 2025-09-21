"""
Tests for weight initialization strategies.
"""

import numpy as np

from neuroscope.mlp.initializers import WeightInits


class TestWeightInits:
    """Test weight initialization methods."""

    def test_he_init_basic(self):
        """Test He initialization basic functionality."""
        layer_dims = [10, 5, 1]
        weights, biases = WeightInits.he_init(layer_dims, seed=42)

        # Check correct number of layers
        assert len(weights) == 2  # 2 weight matrices
        assert len(biases) == 2  # 2 bias vectors

        # Check shapes
        assert weights[0].shape == (10, 5)  # Input to hidden
        assert weights[1].shape == (5, 1)  # Hidden to output
        assert biases[0].shape == (1, 5)  # Hidden bias
        assert biases[1].shape == (1, 1)  # Output bias

        # Check that weights are not zero
        assert not np.allclose(weights[0], 0)
        assert not np.allclose(weights[1], 0)

        # Check that biases are zero (He init zeros biases)
        assert np.allclose(biases[0], 0)
        assert np.allclose(biases[1], 0)

    def test_he_init_variance(self):
        """Test that He initialization produces correct variance."""
        layer_dims = [100, 50, 10]
        weights, biases = WeightInits.he_init(layer_dims, seed=42)

        # He init should have variance ≈ 2/fan_in
        for i, weight_matrix in enumerate(weights):
            fan_in = layer_dims[i]
            expected_std = np.sqrt(2.0 / fan_in)
            actual_std = np.std(weight_matrix)

            # Allow some tolerance due to finite sampling
            assert abs(actual_std - expected_std) < 0.1

    def test_xavier_init_basic(self):
        """Test Xavier initialization basic functionality."""
        layer_dims = [8, 4, 2]
        weights, biases = WeightInits.xavier_init(layer_dims, seed=42)

        # Check correct structure
        assert len(weights) == 2
        assert len(biases) == 2

        # Check shapes
        assert weights[0].shape == (8, 4)
        assert weights[1].shape == (4, 2)
        assert biases[0].shape == (1, 4)
        assert biases[1].shape == (1, 2)

        # Check non-zero weights
        assert not np.allclose(weights[0], 0)
        assert not np.allclose(weights[1], 0)

    def test_xavier_init_variance(self):
        """Test Xavier initialization variance properties."""
        layer_dims = [100, 50, 10]
        weights, biases = WeightInits.xavier_init(layer_dims, seed=42)

        # Xavier init should have variance ≈ 2/(fan_in + fan_out)
        for i, weight_matrix in enumerate(weights):
            fan_in = layer_dims[i]
            fan_out = layer_dims[i + 1]
            expected_std = np.sqrt(2.0 / (fan_in + fan_out))
            actual_std = np.std(weight_matrix)

            # Allow tolerance for finite sampling
            assert abs(actual_std - expected_std) < 0.1

    def test_random_init_basic(self):
        """Test random initialization basic functionality."""
        layer_dims = [6, 3, 1]
        weights, biases = WeightInits.random_init(layer_dims, seed=42)

        # Check structure
        assert len(weights) == 2
        assert len(biases) == 2

        # Check shapes
        assert weights[0].shape == (6, 3)
        assert weights[1].shape == (3, 1)

        # Check non-zero weights
        assert not np.allclose(weights[0], 0)
        assert not np.allclose(weights[1], 0)

    def test_selu_init_basic(self):
        """Test SELU initialization basic functionality."""
        layer_dims = [5, 3, 2]
        weights, biases = WeightInits.selu_init(layer_dims, seed=42)

        # Check structure
        assert len(weights) == 2
        assert len(biases) == 2

        # Check shapes
        assert weights[0].shape == (5, 3)
        assert weights[1].shape == (3, 2)

        # Check non-zero weights
        assert not np.allclose(weights[0], 0)
        assert not np.allclose(weights[1], 0)

    def test_smart_init_basic(self):
        """Test smart initialization basic functionality."""
        layer_dims = [4, 6, 1]

        # Test with different activations
        activations = ["relu", "sigmoid", "tanh", "leaky_relu"]

        for activation in activations:
            weights, biases = WeightInits.smart_init(layer_dims, activation, seed=42)

            # Check structure
            assert len(weights) == 2
            assert len(biases) == 2

            # Check shapes
            assert weights[0].shape == (4, 6)
            assert weights[1].shape == (6, 1)

            # Check non-zero weights
            assert not np.allclose(weights[0], 0)
            assert not np.allclose(weights[1], 0)

    def test_reproducibility_with_seed(self):
        """Test that same seed produces same initialization."""
        layer_dims = [10, 5, 1]

        # Test each initialization method
        init_methods = [
            WeightInits.he_init,
            WeightInits.xavier_init,
            WeightInits.random_init,
            WeightInits.selu_init,
        ]

        for init_method in init_methods:
            # Initialize twice with same seed
            weights1, biases1 = init_method(layer_dims, seed=123)
            weights2, biases2 = init_method(layer_dims, seed=123)

            # Should be identical
            for w1, w2 in zip(weights1, weights2):
                np.testing.assert_array_equal(w1, w2)
            for b1, b2 in zip(biases1, biases2):
                np.testing.assert_array_equal(b1, b2)

    def test_different_seeds_produce_different_weights(self):
        """Test that different seeds produce different initializations."""
        layer_dims = [8, 4, 2]

        init_methods = [
            WeightInits.he_init,
            WeightInits.xavier_init,
            WeightInits.random_init,
            WeightInits.selu_init,
        ]

        for init_method in init_methods:
            # Initialize with different seeds
            weights1, _ = init_method(layer_dims, seed=42)
            weights2, _ = init_method(layer_dims, seed=123)

            # Should be different
            weights_different = False
            for w1, w2 in zip(weights1, weights2):
                if not np.allclose(w1, w2):
                    weights_different = True
                    break

            assert (
                weights_different
            ), f"{init_method.__name__} should produce different weights with different seeds"

    def test_single_layer_network(self):
        """Test initialization with minimal network (single layer)."""
        layer_dims = [5, 1]  # Input directly to output

        init_methods = [
            WeightInits.he_init,
            WeightInits.xavier_init,
            WeightInits.random_init,
            WeightInits.selu_init,
        ]

        for init_method in init_methods:
            weights, biases = init_method(layer_dims, seed=42)

            # Should have exactly one weight matrix and bias vector
            assert len(weights) == 1
            assert len(biases) == 1

            # Check shape
            assert weights[0].shape == (5, 1)
            assert biases[0].shape == (1, 1)

    def test_deep_network_initialization(self):
        """Test initialization with deep network."""
        layer_dims = [100, 64, 32, 16, 8, 1]  # 5-layer network

        init_methods = [
            WeightInits.he_init,
            WeightInits.xavier_init,
            WeightInits.random_init,
            WeightInits.selu_init,
        ]

        for init_method in init_methods:
            weights, biases = init_method(layer_dims, seed=42)

            # Should have 5 weight matrices and bias vectors
            assert len(weights) == 5
            assert len(biases) == 5

            # Check all shapes
            expected_shapes = [(100, 64), (64, 32), (32, 16), (16, 8), (8, 1)]
            for i, (w, expected_shape) in enumerate(zip(weights, expected_shapes)):
                assert w.shape == expected_shape, f"Layer {i} weight shape mismatch"

            # Check that all weights are finite
            for w in weights:
                assert np.all(np.isfinite(w))
            for b in biases:
                assert np.all(np.isfinite(b))

    def test_initialization_statistics(self):
        """Test statistical properties of initializations."""
        layer_dims = [50, 25, 10]

        # Test He initialization statistics
        weights_he, _ = WeightInits.he_init(layer_dims, seed=42)

        # He init should have approximately zero mean
        for w in weights_he:
            assert abs(np.mean(w)) < 0.1  # Should be close to zero

        # Test Xavier initialization statistics
        weights_xavier, _ = WeightInits.xavier_init(layer_dims, seed=42)

        # Xavier init should also have approximately zero mean
        for w in weights_xavier:
            assert abs(np.mean(w)) < 0.1  # Should be close to zero

    def test_smart_init_activation_specific(self):
        """Test that smart init chooses appropriate method for activation."""
        layer_dims = [10, 5, 1]

        # Smart init should handle all supported activations
        activations = ["relu", "leaky_relu", "sigmoid", "tanh", "selu"]

        for activation in activations:
            weights, biases = WeightInits.smart_init(layer_dims, activation, seed=42)

            # Should produce valid initialization
            assert len(weights) == 2
            assert len(biases) == 2

            # All values should be finite
            for w in weights:
                assert np.all(np.isfinite(w))
            for b in biases:
                assert np.all(np.isfinite(b))

            # Weights should not be all zeros
            assert not all(np.allclose(w, 0) for w in weights)

    def test_bias_initialization(self):
        """Test that biases are initialized correctly."""
        layer_dims = [8, 4, 2]

        # Most initialization methods should zero-initialize biases
        methods_with_zero_bias = [
            WeightInits.he_init,
            WeightInits.xavier_init,
            WeightInits.selu_init,
        ]

        for init_method in methods_with_zero_bias:
            _, biases = init_method(layer_dims, seed=42)

            # Biases should be zero or very close to zero
            for b in biases:
                assert np.allclose(b, 0, atol=1e-10)

    def test_large_network_initialization(self):
        """Test initialization performance with large networks."""
        # Test with a reasonably large network
        layer_dims = [1000, 500, 250, 100, 10]

        # Should complete without memory issues or excessive time
        weights, biases = WeightInits.he_init(layer_dims, seed=42)

        # Check that initialization completed successfully
        assert len(weights) == 4
        assert len(biases) == 4

        # Check that large matrices are properly initialized
        assert weights[0].shape == (1000, 500)
        assert not np.allclose(weights[0], 0)

        # Should maintain proper variance even for large matrices
        expected_std = np.sqrt(2.0 / 1000)  # He init for first layer
        actual_std = np.std(weights[0])
        assert (
            abs(actual_std - expected_std) < 0.05
        )  # Tighter tolerance for large matrices
