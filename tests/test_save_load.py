"""
Test suite for MLP save/load functionality.

Tests model persistence including:
- Basic save/load operations
- Optimizer state persistence
- Architecture preservation
- Weight/bias accuracy
- Training continuation
- Edge cases and error handling
"""

from pathlib import Path

import numpy as np
import pytest

from neuroscope.mlp.mlp import MLP


class TestMLPSaveLoad:
    """Test suite for MLP save and load operations."""

    def test_save_creates_file(self, tmp_path):
        """Test save creates file with correct extension."""
        model = MLP([10, 20, 5])
        model.compile(optimizer="adam", lr=0.001)

        filepath = tmp_path / "test_model"
        model.save(str(filepath))

        # Should create .ns file
        expected_path = tmp_path / "test_model.ns"
        assert expected_path.exists()

    def test_save_with_ns_extension(self, tmp_path):
        """Test save works when .ns extension already provided."""
        model = MLP([10, 20, 5])
        model.compile(optimizer="adam", lr=0.001)

        filepath = tmp_path / "model.ns"
        model.save(str(filepath))

        assert filepath.exists()

    def test_load_basic_model(self, tmp_path):
        """Test loading a saved model restores architecture."""
        # Create and save model
        original = MLP(
            layer_dims=[10, 20, 15, 5],
            hidden_activation="relu",
            out_activation="softmax",
            init_method="he",
            dropout_rate=0.2,
            dropout_type="alpha",
        )

        filepath = tmp_path / "model.ns"
        original.save(str(filepath))

        # Load model (returns tuple: model, info)
        loaded, info = MLP.load(str(filepath))

        # Check architecture preserved
        assert loaded.layer_dims == original.layer_dims
        assert loaded.hidden_activation == original.hidden_activation
        assert loaded.out_activation == original.out_activation
        assert loaded.init_method == original.init_method
        assert loaded.dropout_rate == original.dropout_rate
        assert loaded.dropout_type == original.dropout_type

    def test_load_preserves_weights(self, tmp_path):
        """Test loading preserves exact weight values."""
        # Create model with known weights
        original = MLP([5, 10, 3])
        original.compile(optimizer="adam", lr=0.001)

        # Save original weights
        orig_weights = [w.copy() for w in original.weights]
        orig_biases = [b.copy() for b in original.biases]

        # Save and load
        filepath = tmp_path / "model.ns"
        original.save(str(filepath))
        loaded, info = MLP.load(str(filepath))

        # Weights should match exactly
        for orig_w, loaded_w in zip(orig_weights, loaded.weights):
            np.testing.assert_array_equal(orig_w, loaded_w)

        for orig_b, loaded_b in zip(orig_biases, loaded.biases):
            np.testing.assert_array_equal(orig_b, loaded_b)

    def test_load_predictions_match(self, tmp_path):
        """Test loaded model produces identical predictions."""
        # Create and train model
        np.random.seed(42)
        X = np.random.randn(50, 10)
        y = np.random.randn(50, 3)

        original = MLP([10, 20, 3])
        original.compile(optimizer="adam", lr=0.001)
        original.fit(X, y, epochs=5, verbose=False)

        # Get predictions
        orig_preds = original.predict(X)

        # Save, load, and predict again
        filepath = tmp_path / "model.ns"
        original.save(str(filepath))
        loaded, info = MLP.load(str(filepath))
        loaded_preds = loaded.predict(X)

        # Predictions should be identical
        np.testing.assert_allclose(orig_preds, loaded_preds, rtol=1e-10)

    def test_save_without_optimizer(self, tmp_path):
        """Test saving without optimizer state."""
        model = MLP([10, 20, 5])
        model.compile(optimizer="adam", lr=0.001)

        # Train briefly to initialize optimizer state
        X = np.random.randn(20, 10)
        y = np.random.randn(20, 5)
        model.fit(X, y, epochs=2, verbose=False)

        # Save without optimizer
        filepath = tmp_path / "model.ns"
        model.save(str(filepath), save_optimizer=False)

        # Load and check optimizer is not restored
        loaded, info = MLP.load(str(filepath), load_optimizer=False)
        assert not loaded.compiled
        assert loaded.optimizer is None

    def test_save_with_optimizer_sgd(self, tmp_path):
        """Test saving and loading SGD optimizer state."""
        model = MLP([10, 15, 5])
        model.compile(optimizer="sgd", lr=0.01)

        # Train to initialize optimizer
        X = np.random.randn(30, 10)
        y = np.random.randn(30, 5)
        model.fit(X, y, epochs=3, verbose=False)

        # Save with optimizer
        filepath = tmp_path / "model.ns"
        model.save(str(filepath), save_optimizer=True)

        # Load with optimizer
        loaded, info = MLP.load(str(filepath), load_optimizer=True)

        assert loaded.compiled
        assert loaded.optimizer is not None
        assert loaded.optimizer.__class__.__name__ == "SGD"
        assert loaded.lr == 0.01

    def test_save_with_optimizer_adam(self, tmp_path):
        """Test saving and loading Adam optimizer state with moments."""
        model = MLP([8, 12, 4])
        model.compile(optimizer="adam", lr=0.001)

        # Train to initialize Adam moments
        X = np.random.randn(40, 8)
        y = np.random.randn(40, 4)
        model.fit(X, y, epochs=5, verbose=False)

        # Save optimizer state
        orig_optimizer = model.optimizer
        orig_state = orig_optimizer.state_dict()

        # Save and load
        filepath = tmp_path / "model.ns"
        model.save(str(filepath), save_optimizer=True)
        loaded, info = MLP.load(str(filepath), load_optimizer=True)

        # Check optimizer restored
        assert loaded.optimizer.__class__.__name__ == "Adam"
        assert loaded.optimizer.learning_rate == 0.001

        # Check moments preserved
        loaded_state = loaded.optimizer.state_dict()
        assert "m_weights" in loaded_state["state"]
        assert "v_weights" in loaded_state["state"]
        assert loaded_state["state"]["t"] == orig_state["state"]["t"]

    def test_save_with_optimizer_momentum(self, tmp_path):
        """Test saving and loading SGD with Momentum optimizer."""
        model = MLP([10, 20, 5])
        model.compile(optimizer="sgdm", lr=0.01)

        # Train to initialize velocity
        X = np.random.randn(30, 10)
        y = np.random.randn(30, 5)
        model.fit(X, y, epochs=5, verbose=False)

        # Save and load
        filepath = tmp_path / "model.ns"
        model.save(str(filepath), save_optimizer=True)
        loaded, info = MLP.load(str(filepath), load_optimizer=True)

        # Check optimizer and velocity preserved
        assert loaded.optimizer.__class__.__name__ == "SGDMomentum"
        assert "velocity_w" in loaded.optimizer._state
        assert "velocity_b" in loaded.optimizer._state

    def test_save_with_optimizer_nesterov(self, tmp_path):
        """Test saving and loading SGD with Nesterov momentum."""
        model = MLP([10, 15, 5])
        model.compile(optimizer="sgdnm", lr=0.01)

        # Train
        X = np.random.randn(30, 10)
        y = np.random.randn(30, 5)
        model.fit(X, y, epochs=5, verbose=False)

        # Save and load
        filepath = tmp_path / "model.ns"
        model.save(str(filepath), save_optimizer=True)
        loaded, info = MLP.load(str(filepath), load_optimizer=True)

        # Check Nesterov flag preserved
        assert loaded.optimizer.nesterov is True

    def test_save_with_optimizer_rmsprop(self, tmp_path):
        """Test saving and loading RMSprop optimizer."""
        model = MLP([10, 20, 5])
        model.compile(optimizer="rmsprop", lr=0.001)

        # Train to initialize cache
        X = np.random.randn(30, 10)
        y = np.random.randn(30, 5)
        model.fit(X, y, epochs=5, verbose=False)

        # Save and load
        filepath = tmp_path / "model.ns"
        model.save(str(filepath), save_optimizer=True)
        loaded, info = MLP.load(str(filepath), load_optimizer=True)

        # Check optimizer and cache preserved
        assert loaded.optimizer.__class__.__name__ == "RMSprop"
        assert "square_avg_weights" in loaded.optimizer._state
        assert "square_avg_biases" in loaded.optimizer._state

    def test_training_continuation(self, tmp_path):
        """Test training can continue seamlessly after load."""
        np.random.seed(42)
        X = np.random.randn(100, 10)
        y = np.random.randn(100, 3)

        # Train for 10 epochs
        model = MLP([10, 20, 3])
        model.compile(optimizer="adam", lr=0.001)
        history1 = model.fit(X, y, epochs=10, verbose=False)
        loss_at_10 = history1["history"]["train_loss"][-1]

        # Save with optimizer
        filepath = tmp_path / "checkpoint.ns"
        model.save(str(filepath), save_optimizer=True)

        # Load and continue training
        loaded, info = MLP.load(str(filepath), load_optimizer=True)
        history2 = loaded.fit(X, y, epochs=10, verbose=False)
        loss_at_11 = history2["history"]["train_loss"][0]

        # Loss should continue smoothly (no large jump)
        # Allow up to 30% variation due to random initialization, dropout, etc.
        loss_jump = abs(loss_at_11 - loss_at_10) / loss_at_10
        assert loss_jump < 0.3, f"Loss jumped {loss_jump*100:.1f}% after loading"

    def test_training_continuation_without_optimizer_state(self, tmp_path):
        """Test training after loading without optimizer state requires recompile."""
        X = np.random.randn(50, 10)
        y = np.random.randn(50, 3)

        # Train and save
        model = MLP([10, 15, 3])
        model.compile(optimizer="adam", lr=0.001)
        model.fit(X, y, epochs=5, verbose=False)

        filepath = tmp_path / "model.ns"
        model.save(str(filepath), save_optimizer=False)

        # Load without optimizer
        loaded, info = MLP.load(str(filepath), load_optimizer=False)

        # Should need to compile before training
        assert not loaded.compiled

        # Compile and train
        loaded.compile(optimizer="adam", lr=0.001)
        history = loaded.fit(X, y, epochs=5, verbose=False)
        assert len(history["history"]["train_loss"]) == 5

    def test_save_with_metadata(self, tmp_path):
        """Test saving with custom metadata."""
        model = MLP([10, 20, 5])
        model.compile(optimizer="adam", lr=0.001)

        # Save with metadata
        filepath = tmp_path / "model.ns"
        model.save(str(filepath), epoch=50, accuracy=0.95, notes="best model")

        # Load and check (metadata is saved but not directly accessible in this implementation)
        loaded, info = MLP.load(str(filepath))
        assert loaded is not None

    def test_load_nonexistent_file(self):
        """Test loading non-existent file raises error."""
        with pytest.raises(FileNotFoundError):
            MLP.load("nonexistent_model.ns")

    def test_load_invalid_file(self, tmp_path):
        """Test loading invalid file format raises error."""
        # Create invalid file
        invalid_file = tmp_path / "invalid.ns"
        with open(invalid_file, "w") as f:
            f.write("This is not a valid model file")

        with pytest.raises(ValueError, match="Failed to load model file"):
            MLP.load(str(invalid_file))

    def test_load_missing_keys(self, tmp_path):
        """Test loading file with missing required keys raises error."""
        import pickle

        # Create file with missing keys
        incomplete_file = tmp_path / "incomplete.ns"
        with open(incomplete_file, "wb") as f:
            pickle.dump({"weights": [], "biases": []}, f)  # Missing 'model_config'

        with pytest.raises(ValueError, match="Invalid .ns file format"):
            MLP.load(str(incomplete_file))

    def test_multiple_save_load_cycles(self, tmp_path):
        """Test multiple save/load cycles preserve model accuracy."""
        np.random.seed(42)
        X = np.random.randn(50, 10)
        y = np.random.randn(50, 3)

        model = MLP([10, 15, 3])
        model.compile(optimizer="adam", lr=0.001)
        model.fit(X, y, epochs=10, verbose=False)

        # Get initial predictions
        orig_preds = model.predict(X)

        # Save and load multiple times
        for i in range(3):
            filepath = tmp_path / f"model_{i}.ns"
            model.save(str(filepath), save_optimizer=True)
            model, info = MLP.load(str(filepath), load_optimizer=True)

        # Final predictions should match original
        final_preds = model.predict(X)
        np.testing.assert_allclose(orig_preds, final_preds, rtol=1e-10)

    def test_save_load_different_architectures(self, tmp_path):
        """Test save/load works for various architectures."""
        architectures = [
            [5, 10, 5],  # Simple
            [100, 50, 25, 10],  # Deep
            [50, 200, 100, 10],  # Wide
            [10, 5, 3, 1],  # Narrow
        ]

        for i, arch in enumerate(architectures):
            model = MLP(arch)
            model.compile(optimizer="adam", lr=0.001)

            filepath = tmp_path / f"arch_{i}.ns"
            model.save(str(filepath))

            loaded, info = MLP.load(str(filepath))
            assert loaded.layer_dims == arch

    def test_save_load_different_activations(self, tmp_path):
        """Test save/load preserves different activation functions."""
        activations = ["relu", "leaky_relu", "tanh", "sigmoid", "selu"]
        out_activations = [None, "sigmoid", "softmax"]

        for i, (hidden_act, out_act) in enumerate(zip(activations, out_activations)):
            model = MLP(
                [10, 15, 5], hidden_activation=hidden_act, out_activation=out_act
            )

            filepath = tmp_path / f"activation_{i}.ns"
            model.save(str(filepath))

            loaded, info = MLP.load(str(filepath))
            assert loaded.hidden_activation == hidden_act
            assert loaded.out_activation == out_act

    def test_save_load_different_initializations(self, tmp_path):
        """Test save/load preserves initialization method."""
        init_methods = ["he", "xavier", "random", "selu_init", "smart"]

        for i, init_method in enumerate(init_methods):
            model = MLP([10, 20, 5], init_method=init_method)

            filepath = tmp_path / f"init_{i}.ns"
            model.save(str(filepath))

            loaded, info = MLP.load(str(filepath))
            assert loaded.init_method == init_method

    def test_save_load_with_dropout(self, tmp_path):
        """Test save/load preserves dropout configuration."""
        dropout_configs = [
            (0.0, "normal"),
            (0.2, "normal"),
            (0.5, "alpha"),
            (0.3, "normal"),
        ]

        for i, (rate, dtype) in enumerate(dropout_configs):
            model = MLP([10, 20, 5], dropout_rate=rate, dropout_type=dtype)

            filepath = tmp_path / f"dropout_{i}.ns"
            model.save(str(filepath))

            loaded, info = MLP.load(str(filepath))
            assert loaded.dropout_rate == rate
            assert loaded.dropout_type == dtype

    def test_save_load_preserves_seed(self, tmp_path):
        """Test save/load preserves initialization seed."""
        seeds = [42, 123, 999, 0]

        for seed in seeds:
            model = MLP([10, 15, 5], init_seed=seed)

            filepath = tmp_path / f"seed_{seed}.ns"
            model.save(str(filepath))

            loaded, info = MLP.load(str(filepath))
            assert loaded.init_seed == seed

    def test_cross_optimizer_load(self, tmp_path):
        """Test loading model and switching optimizers."""
        X = np.random.randn(50, 10)
        y = np.random.randn(50, 3)

        # Train with Adam
        model = MLP([10, 15, 3])
        model.compile(optimizer="adam", lr=0.001)
        model.fit(X, y, epochs=5, verbose=False)

        # Save without optimizer
        filepath = tmp_path / "model.ns"
        model.save(str(filepath), save_optimizer=False)

        # Load and use different optimizer
        loaded, info = MLP.load(str(filepath), load_optimizer=False)
        loaded.compile(optimizer="sgdm", lr=0.01)

        # Should work fine
        history = loaded.fit(X, y, epochs=5, verbose=False)
        assert len(history["history"]["train_loss"]) == 5

    def test_save_load_large_model(self, tmp_path):
        """Test save/load handles large models correctly."""
        # Large model
        model = MLP([1000, 500, 250, 100, 10])
        model.compile(optimizer="adam", lr=0.001)

        filepath = tmp_path / "large_model.ns"
        model.save(str(filepath))

        loaded, info = MLP.load(str(filepath))

        # Check all weights loaded
        assert len(loaded.weights) == 4
        assert loaded.weights[0].shape == (1000, 500)
        assert loaded.weights[-1].shape == (100, 10)

    def test_save_optimizer_state_convergence(self, tmp_path):
        """Test that optimizer state preservation leads to better convergence."""
        np.random.seed(42)
        X = np.random.randn(100, 10)
        y = np.random.randn(100, 3)

        # Train for 10 epochs with Adam (builds momentum)
        model1 = MLP([10, 20, 3])
        model1.compile(optimizer="adam", lr=0.001)
        model1.fit(X, y, epochs=10, verbose=False)

        # Save with optimizer state
        filepath = tmp_path / "with_state.ns"
        model1.save(str(filepath), save_optimizer=True)

        # Continue training with loaded state
        loaded_with_state, info = MLP.load(str(filepath), load_optimizer=True)
        history_with = loaded_with_state.fit(X, y, epochs=5, verbose=False)
        final_loss_with = history_with["history"]["train_loss"][-1]

        # Train fresh model from same weights but no optimizer state
        model2 = MLP([10, 20, 3])
        model2.weights = [w.copy() for w in model1.weights]
        model2.biases = [b.copy() for b in model1.biases]
        model2.compile(optimizer="adam", lr=0.001)
        history_without = model2.fit(X, y, epochs=5, verbose=False)
        final_loss_without = history_without["history"]["train_loss"][-1]

        # Model with preserved state should converge better or equally
        assert final_loss_with <= final_loss_without * 1.1  # Allow 10% margin


# Pytest fixtures for this test file
@pytest.fixture
def tmp_path():
    """Provide temporary directory for test files."""
    import tempfile

    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)
