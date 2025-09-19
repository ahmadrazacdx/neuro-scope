"""
Test configuration and fixtures for NeuroScope test suite.

This module provides shared fixtures and utilities for testing all NeuroScope components.
"""
import numpy as np
import pytest
from typing import Tuple, List, Dict, Any

# Test data constants
RANDOM_SEED = 42
N_SAMPLES = 100
N_FEATURES = 20
N_CLASSES = 3
BATCH_SIZE = 32


@pytest.fixture
def rng() -> np.random.Generator:
    """Provide consistent random number generator for reproducible tests."""
    return np.random.default_rng(RANDOM_SEED)


@pytest.fixture
def sample_data(rng: np.random.Generator) -> Tuple[np.ndarray, np.ndarray]:
    """Generate sample regression data for testing.
    
    Returns:
        Tuple containing (X, y) where X is features and y is targets.
    """
    X = rng.standard_normal((N_SAMPLES, N_FEATURES))
    y = rng.standard_normal(N_SAMPLES)
    return X, y


@pytest.fixture
def classification_data(rng: np.random.Generator) -> Tuple[np.ndarray, np.ndarray]:
    """Generate sample classification data for testing.
    
    Returns:
        Tuple containing (X, y) where X is features and y is class labels.
    """
    X = rng.standard_normal((N_SAMPLES, N_FEATURES))
    y = rng.integers(0, N_CLASSES, N_SAMPLES)
    return X, y


@pytest.fixture
def batch_data(rng: np.random.Generator) -> Tuple[np.ndarray, np.ndarray]:
    """Generate batch data for testing training loops.
    
    Returns:
        Tuple containing (X_batch, y_batch).
    """
    X = rng.standard_normal((BATCH_SIZE, N_FEATURES))
    y = rng.standard_normal(BATCH_SIZE)
    return X, y


@pytest.fixture
def mlp_config() -> Dict[str, Any]:
    """Provide standard MLP configuration for testing."""
    return {
        'layer_dims': [N_FEATURES, 64, 32, 1],
        'hidden_activation': 'relu',
        'out_activation': None,
        'init_method': 'he',
        'dropout_rate': 0.1
    }


@pytest.fixture
def training_history() -> List[Dict[str, float]]:
    """Generate mock training history for testing diagnostics."""
    history = []
    for epoch in range(10):
        history.append({
            'epoch': epoch,
            'train_loss': 1.0 - epoch * 0.1 + np.random.normal(0, 0.05),
            'val_loss': 1.2 - epoch * 0.08 + np.random.normal(0, 0.08),
            'train_acc': epoch * 0.1 + np.random.normal(0, 0.02),
            'val_acc': epoch * 0.08 + np.random.normal(0, 0.03)
        })
    return history


@pytest.fixture
def predictions_true_pairs(rng: np.random.Generator) -> Tuple[np.ndarray, np.ndarray]:
    """Generate predictions and true values for metrics testing."""
    y_true = rng.standard_normal(N_SAMPLES)
    y_pred = y_true + rng.standard_normal(N_SAMPLES) * 0.1  # Add noise
    return y_pred, y_true


@pytest.fixture
def weight_matrices(rng: np.random.Generator) -> List[np.ndarray]:
    """Generate sample weight matrices for testing."""
    return [
        rng.standard_normal((N_FEATURES, 64)),
        rng.standard_normal((64, 32)), 
        rng.standard_normal((32, 1))
    ]


@pytest.fixture
def bias_vectors(rng: np.random.Generator) -> List[np.ndarray]:
    """Generate sample bias vectors for testing."""
    return [
        rng.standard_normal(64),
        rng.standard_normal(32),
        rng.standard_normal(1)
    ]


# Parametrized fixtures for testing multiple scenarios
@pytest.fixture(params=['relu', 'sigmoid', 'tanh', 'leaky_relu'])
def activation_function(request) -> str:
    """Parametrized activation function for comprehensive testing."""
    return request.param


@pytest.fixture(params=['xavier', 'he', 'zeros', 'ones'])
def initializer_type(request) -> str:
    """Parametrized initializer for comprehensive testing."""
    return request.param


@pytest.fixture(params=['mse', 'mae', 'huber'])
def loss_function(request) -> str:
    """Parametrized loss function for comprehensive testing."""
    return request.param


@pytest.fixture(params=[0.001, 0.01, 0.1])
def learning_rate(request) -> float:
    """Parametrized learning rates for testing."""
    return request.param


# Helper functions for assertions
def assert_array_properties(arr: np.ndarray, expected_shape: Tuple[int, ...], 
                           dtype: type = np.float64, finite: bool = True) -> None:
    """Assert basic properties of numpy arrays in tests."""
    assert isinstance(arr, np.ndarray), f"Expected ndarray, got {type(arr)}"
    assert arr.shape == expected_shape, f"Expected shape {expected_shape}, got {arr.shape}"
    assert arr.dtype == dtype, f"Expected dtype {dtype}, got {arr.dtype}"
    if finite:
        assert np.isfinite(arr).all(), "Array contains non-finite values"


def assert_loss_properties(loss_value: float) -> None:
    """Assert properties expected of loss values."""
    assert isinstance(loss_value, (int, float, np.number)), f"Loss must be numeric, got {type(loss_value)}"
    assert np.isfinite(loss_value), "Loss must be finite"
    assert loss_value >= 0, f"Loss must be non-negative, got {loss_value}"


def assert_metric_range(metric_value: float, min_val: float = 0.0, max_val: float = 1.0) -> None:
    """Assert metric values are within expected range."""
    assert isinstance(metric_value, (int, float, np.number)), f"Metric must be numeric, got {type(metric_value)}"
    assert np.isfinite(metric_value), "Metric must be finite"
    assert min_val <= metric_value <= max_val, f"Metric {metric_value} not in range [{min_val}, {max_val}]"