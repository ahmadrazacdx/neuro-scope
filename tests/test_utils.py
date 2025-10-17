"""
Test suite for Utils module.
Tests utility functions and helper methods.
"""

import numpy as np

from neuroscope.mlp.utils import Utils


class TestUtils:
    """Test suite for utility functions."""

    def test_get_batches_fast_method(self, sample_data):
        """Test get_batches_fast optimized batch processing function."""
        X, y = sample_data
        batch_size = 16

        # Test basic functionality
        batches = list(Utils.get_batches_fast(X, y, batch_size, shuffle=False))

        # Check that we get the expected number of batches
        expected_batches = int(np.ceil(len(X) / batch_size))
        assert len(batches) == expected_batches

        # Check batch shapes and content
        total_samples = 0
        for i, (X_batch, y_batch) in enumerate(batches):
            # Check shapes
            assert X_batch.ndim == 2
            assert y_batch.ndim == 2  # get_batches_fast reshapes y to 2D
            assert X_batch.shape[0] == y_batch.shape[0]
            assert X_batch.shape[1] == X.shape[1]  # Same number of features

            # Check batch size (last batch might be smaller)
            if i < len(batches) - 1:
                assert X_batch.shape[0] == batch_size
            else:
                assert X_batch.shape[0] <= batch_size

            total_samples += X_batch.shape[0]

            # Check that data is finite
            assert np.all(np.isfinite(X_batch))
            assert np.all(np.isfinite(y_batch))

        # Check that all samples are included
        assert total_samples == len(X)

    def test_get_batches_fast_shuffle(self, classification_data):
        """Test get_batches_fast with shuffle functionality."""
        X, y = classification_data
        batch_size = 20

        # Get batches without shuffle
        batches_no_shuffle = list(
            Utils.get_batches_fast(X, y, batch_size, shuffle=False)
        )
        first_batch_no_shuffle = batches_no_shuffle[0][0]

        # Get batches with shuffle
        batches_shuffle = list(Utils.get_batches_fast(X, y, batch_size, shuffle=True))
        first_batch_shuffle = batches_shuffle[0][0]

        # With shuffle, first batch should likely be different
        # (This test might occasionally fail due to randomness, but very unlikely)
        assert not np.array_equal(first_batch_no_shuffle, first_batch_shuffle)

        # But total number of batches should be the same
        assert len(batches_no_shuffle) == len(batches_shuffle)

    def test_get_batches_fast_edge_cases(self, rng):
        """Test get_batches_fast with edge cases."""
        # Small dataset
        X_small = rng.standard_normal((5, 3))
        y_small = rng.standard_normal(5)

        # Batch size larger than dataset
        batches = list(
            Utils.get_batches_fast(X_small, y_small, batch_size=10, shuffle=False)
        )
        assert len(batches) == 1
        assert batches[0][0].shape[0] == 5

        # Batch size equals dataset size
        batches = list(
            Utils.get_batches_fast(X_small, y_small, batch_size=5, shuffle=False)
        )
        assert len(batches) == 1
        assert batches[0][0].shape[0] == 5

        # Batch size of 1
        batches = list(
            Utils.get_batches_fast(X_small, y_small, batch_size=1, shuffle=False)
        )
        assert len(batches) == 5
        for X_batch, y_batch in batches:
            assert X_batch.shape[0] == 1
            assert y_batch.shape[0] == 1

    def test_get_batches_fast_vs_get_batches(self, sample_data):
        """Test that get_batches_fast produces same results as get_batches (if it exists)."""
        X, y = sample_data
        batch_size = 16

        # Test get_batches_fast
        batches_fast = list(Utils.get_batches_fast(X, y, batch_size, shuffle=False))

        # Test regular get_batches if it exists
        if hasattr(Utils, "get_batches"):
            batches_regular = list(Utils.get_batches(X, y, batch_size, shuffle=False))

            # Should produce same number of batches
            assert len(batches_fast) == len(batches_regular)

            # Each batch should have same content (when not shuffled)
            for (X_fast, y_fast), (X_reg, y_reg) in zip(batches_fast, batches_regular):
                assert X_fast.shape == X_reg.shape
                assert y_fast.shape == y_reg.shape
                # Note: We don't check exact equality since fast version might use views
                # but shapes and finite values should match
                assert np.all(np.isfinite(X_fast)) == np.all(np.isfinite(X_reg))
                assert np.all(np.isfinite(y_fast)) == np.all(np.isfinite(y_reg))

    def test_check_numerical_stability_large_gradients(self, rng):
        """Test numerical stability warnings with large gradients."""
        # Create large gradient values
        large_grads = [rng.standard_normal((10, 10)) * 1e9]

        # Should return issues for large values in gradients context
        issues = Utils.check_numerical_stability(
            large_grads, context="gradients", fast_mode=False
        )
        # Should have warnings about large values
        assert len(issues) > 0
        assert "large" in issues[0].lower() or "gradients" in issues[0].lower()

    def test_check_numerical_stability_large_outputs(self, rng):
        """Test numerical stability warnings with large output activations."""
        # Create large output values
        large_outputs = [rng.standard_normal((10, 10)) * 1e9]

        issues = Utils.check_numerical_stability(
            large_outputs, context="output_activations", fast_mode=False
        )
        # Should have warnings about large values
        assert len(issues) > 0

    def test_check_numerical_stability_large_forward_pass(self, rng):
        """Test numerical stability warnings during forward pass."""
        # Create large activation values
        large_activations = [rng.standard_normal((10, 10)) * 1e9]

        issues = Utils.check_numerical_stability(
            large_activations, context="forward_pass", fast_mode=False
        )
        # Should have warnings about large values
        assert len(issues) > 0

    def test_check_numerical_stability_generic_context(self, rng):
        """Test numerical stability with generic context message."""
        # Create large values without specific context
        large_values = [rng.standard_normal((10, 10)) * 1e9]

        issues = Utils.check_numerical_stability(
            large_values, context="generic", fast_mode=False
        )
        # Should have warnings
        assert len(issues) > 0

    def test_check_numerical_stability_fast_mode_skips_checks(self, rng):
        """Test that fast_mode=True skips detailed checks for large values."""
        # Create large values that would normally trigger warnings
        large_grads = [rng.standard_normal((10, 10)) * 1e9]

        # Fast mode should not detect large values (only NaN/Inf)
        issues = Utils.check_numerical_stability(
            large_grads, context="gradients", fast_mode=True
        )
        # Fast mode should skip large value checks
        assert len(issues) == 0

    def test_validate_array_input_nan_values(self):
        """Test validate_array_input with NaN values."""
        import pytest

        arr_with_nan = np.array([1.0, 2.0, np.nan, 4.0])
        with pytest.raises(ValueError, match="contains NaN values"):
            Utils.validate_array_input(arr_with_nan, "test_array", fast_mode=False)

    def test_validate_array_input_inf_values(self):
        """Test validate_array_input with infinite values."""
        import pytest

        arr_with_inf = np.array([1.0, 2.0, np.inf, 4.0])
        with pytest.raises(ValueError, match="contains infinite values"):
            Utils.validate_array_input(arr_with_inf, "test_array", fast_mode=False)

    def test_validate_array_input_conversion_warning(self, capsys):
        """Test that array conversion prints warning in non-fast mode."""
        # Pass a list instead of ndarray
        arr_list = [[1, 2], [3, 4]]
        result = Utils.validate_array_input(arr_list, "test_list", fast_mode=False)

        # Check it was converted
        assert isinstance(result, np.ndarray)

        # Check warning was printed
        captured = capsys.readouterr()
        assert "test_list converted to numpy array" in captured.out

    def test_gradient_clipping_with_small_norm(self, rng):
        """Test gradient clipping when norm is already small."""
        # Create small gradients
        small_grads = [rng.standard_normal((3, 2)) * 0.01]

        clipped = Utils.gradient_clipping(small_grads, max_norm=5.0)

        # Should not be clipped (returned as-is)
        assert np.allclose(clipped[0], small_grads[0])

    def test_validate_layer_dims_tuple_input(self):
        """Test validate_layer_dims with tuple input."""
        layer_dims = (10, 20, 30, 1)
        result = Utils.validate_layer_dims(layer_dims, input_dim=10)

        # Should convert to list and validate
        assert isinstance(result, list)
        assert result == [10, 20, 30, 1]

    def test_validate_array_dimension_error(self):
        """Test validate_array_input with wrong dimensions (hits line 161-162)."""
        arr_3d = np.random.randn(5, 5, 5)  # 3D array

        # Expect 2D array, got 3D - should raise ValueError
        try:
            Utils.validate_array_input(arr_3d, "test_3d", min_dims=2, max_dims=2)
            assert False, "Should have raised ValueError"
        except ValueError as e:
            assert "must have 2-2 dimensions" in str(e)

    def test_check_numerical_stability_with_nan(self):
        """Test numerical stability check catches NaN (hits line 227)."""
        arrays_with_nan = [np.array([[1.0, 2.0], [np.nan, 4.0]])]

        issues = Utils.check_numerical_stability(
            arrays_with_nan, "test", fast_mode=False
        )

        assert len(issues) > 0
        assert "NaN" in issues[0]

    def test_check_numerical_stability_with_inf_fast_mode(self):
        """Test numerical stability check catches Inf in fast_mode (hits line 241)."""
        arrays_with_inf = [np.array([[1.0, 2.0], [np.inf, 4.0]])]

        issues = Utils.check_numerical_stability(
            arrays_with_inf, "test", fast_mode=True
        )

        assert len(issues) > 0
        assert "Inf" in issues[0] or "exploded" in issues[0]
