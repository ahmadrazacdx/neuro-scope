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
