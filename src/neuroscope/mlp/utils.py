"""
Utilities Module
Helper functions for training, validation, and data processing.
"""

import warnings

import numpy as np


class Utils:
    """
    Utility functions for neural network training and data processing.

    Provides essential helper functions for batch processing, gradient clipping,
    input validation, and numerical stability checks. All methods are static
    and can be used independently throughout the framework.
    """

    @staticmethod
    def get_batches(X, y, batch_size=32, shuffle=True):
        """
        Generate mini-batches for training.

        Creates mini-batches from input data with optional shuffling for
        stochastic gradient descent training. Handles the last batch even
        if it contains fewer samples than batch_size.

        Args:
            X (NDArray[np.float64]): Input data of shape (N, input_dim).
            y (NDArray[np.float64]): Target data of shape (N,) or (N, output_dim).
            batch_size (int, optional): Size of each mini-batch. Defaults to 32.
            shuffle (bool, optional): Whether to shuffle data before batching. Defaults to True.

        Yields:
            tuple[NDArray, NDArray]: (X_batch, y_batch) for each mini-batch.

        Example:
            >>> for X_batch, y_batch in Utils.get_batches(X_train, y_train, batch_size=64):
            ...     # Process batch
            ...     pass
        """
        N = X.shape[0]
        idx = np.arange(N)
        if shuffle:
            np.random.shuffle(idx)
        for start in range(0, N, batch_size):
            batch_idx = idx[start : start + batch_size]
            yield X[batch_idx], y[batch_idx].reshape(-1, 1)

    @staticmethod
    def gradient_clipping(gradients, max_norm=5.0):
        """
        Apply gradient clipping to prevent exploding gradients.

        Clips gradients by global norm as described in Pascanu et al. (2013).
        If the global norm exceeds max_norm, all gradients are scaled down
        proportionally to maintain their relative magnitudes.

        Args:
            gradients (list[NDArray[np.float64]]): List of gradient arrays.
            max_norm (float, optional): Maximum allowed gradient norm. Defaults to 5.0.

        Returns:
            list[NDArray[np.float64]]: Clipped gradient arrays.

        Note:
            Based on "On the difficulty of training recurrent neural networks"
            (Pascanu et al. 2013) for gradient norm clipping.

        Example:
            >>> clipped_grads = Utils.gradient_clipping(gradients, max_norm=5.0)
        """
        total_norm = 0
        for grad in gradients:
            total_norm += np.sum(grad**2)
        total_norm = np.sqrt(total_norm)
        if total_norm > max_norm:
            clip_coef = max_norm / (total_norm + 1e-8)
            gradients = [grad * clip_coef for grad in gradients]
        return gradients

    @staticmethod
    def validate_array_input(arr, name, min_dims=1, max_dims=3):
        """
        Validate and sanitize input arrays for neural network operations.

        Performs comprehensive validation including type checking, dimension
        validation, and numerical stability checks. Automatically converts
        compatible inputs to numpy arrays when possible.

        Args:
            arr: Input array or array-like object to validate.
            name (str): Name of the array for error messages.
            min_dims (int, optional): Minimum allowed dimensions. Defaults to 1.
            max_dims (int, optional): Maximum allowed dimensions. Defaults to 3.

        Returns:
            NDArray[np.float64]: Validated numpy array.

        Raises:
            TypeError: If input cannot be converted to numpy array.
            ValueError: If dimensions, shape, or values are invalid.

        Example:
            >>> X_valid = Utils.validate_array_input(X, "training_data", min_dims=2, max_dims=2)
        """
        if not isinstance(arr, np.ndarray):
            try:
                arr = np.asarray(arr)
                warnings.warn(f"{name} converted to numpy array")
            except Exception as e:
                raise TypeError(f"{name} must be convertible to numpy array: {e}")
        if arr.ndim < min_dims or arr.ndim > max_dims:
            raise ValueError(
                f"{name} must have {min_dims}-{max_dims} dimensions, got {arr.ndim}"
            )
        if arr.size == 0:
            raise ValueError(f"{name} cannot be empty")
        if np.any(np.isnan(arr)):
            raise ValueError(f"{name} contains NaN values")
        if np.any(np.isinf(arr)):
            raise ValueError(f"{name} contains infinite values")
        return arr

    @staticmethod
    def validate_layer_dims(layer_dims, input_dim):
        if not isinstance(layer_dims, (list, tuple)):
            raise TypeError("layer_dims must be a list or tuple")
        if len(layer_dims) < 2:
            raise ValueError(
                "layer_dims must have at least 2 layers (input and output)"
            )
        layer_dims = list(layer_dims)

        for i, dim in enumerate(layer_dims):
            if not isinstance(dim, int) or dim <= 0:
                raise ValueError(
                    f"Layer {i} dimension must be positive integer, got {dim}"
                )
        if layer_dims[0] != input_dim:
            raise ValueError(
                f"First layer dimension {layer_dims[0]} must match input features {input_dim}"
            )
        return layer_dims

    @staticmethod
    def check_numerical_stability(arrays, context="computation"):
        issues = []
        for i, arr in enumerate(arrays):
            if arr is None:
                continue
            # Check for NaN
            nan_count = np.sum(np.isnan(arr))
            if nan_count > 0:
                issues.append(
                    f"Array {i} in {context}: {nan_count} NaN values detected"
                )
            # Check for Inf
            inf_count = np.sum(np.isinf(arr))
            if inf_count > 0:
                issues.append(
                    f"Array {i} in {context}: {inf_count} infinite values detected"
                )
            # Check for very large values
            max_val = np.max(np.abs(arr))
            if max_val > 1e10:
                issues.append(
                    f"Array {i} in {context}: very large values detected (max: {max_val:.2e})"
                )
            # Check for very small gradients
            if context == "gradients" and max_val < 1e-10:
                issues.append(
                    f"Array {i} in {context}: very small gradients detected (max: {max_val:.2e})"
                )
        return issues
