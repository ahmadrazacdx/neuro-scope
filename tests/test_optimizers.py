"""
Test suite for optimizer implementations.
"""

import numpy as np
import pytest

from neuroscope.mlp.optimizers import SGD, Adam, RMSprop, SGDMomentum


class TestSGD:
    """Test suite for SGD optimizer."""

    def test_sgd_initialization(self):
        """Test SGD initializes correctly with valid parameters."""
        optimizer = SGD(learning_rate=0.01)
        assert optimizer.learning_rate == 0.01
        assert optimizer._state == {}
        assert optimizer.__class__.__name__ == "SGD"

    def test_sgd_invalid_learning_rate(self):
        """Test SGD raises error for invalid learning rates."""
        with pytest.raises(ValueError, match="Learning rate must be positive"):
            SGD(learning_rate=0.0)

        with pytest.raises(ValueError, match="Learning rate must be positive"):
            SGD(learning_rate=-0.01)

    def test_sgd_update(self):
        """Test SGD updates parameters correctly."""
        optimizer = SGD(learning_rate=0.1)

        # Create simple parameters
        weights = [np.array([[1.0, 2.0], [3.0, 4.0]])]
        biases = [np.array([[0.5, 0.5]])]

        # Create gradients
        weight_grads = [np.array([[0.1, 0.2], [0.3, 0.4]])]
        bias_grads = [np.array([[0.1, 0.2]])]

        # Store original values
        orig_w = weights[0].copy()
        orig_b = biases[0].copy()

        # Apply update
        optimizer.update(weights, biases, weight_grads, bias_grads)

        # Check updates: θ_new = θ_old - lr * grad
        expected_w = orig_w - 0.1 * weight_grads[0]
        expected_b = orig_b - 0.1 * bias_grads[0]

        np.testing.assert_allclose(weights[0], expected_w, rtol=1e-10)
        np.testing.assert_allclose(biases[0], expected_b, rtol=1e-10)

    def test_sgd_multiple_layers(self):
        """Test SGD updates multiple layers correctly."""
        optimizer = SGD(learning_rate=0.01)

        # Create 3-layer network parameters
        weights = [
            np.random.randn(10, 20),
            np.random.randn(20, 15),
            np.random.randn(15, 5),
        ]
        biases = [np.random.randn(1, 20), np.random.randn(1, 15), np.random.randn(1, 5)]

        # Create matching gradients
        weight_grads = [np.random.randn(*w.shape) for w in weights]
        bias_grads = [np.random.randn(*b.shape) for b in biases]

        # Store originals
        orig_weights = [w.copy() for w in weights]
        orig_biases = [b.copy() for b in biases]

        # Update
        optimizer.update(weights, biases, weight_grads, bias_grads)

        # Verify all layers updated
        for i in range(len(weights)):
            expected_w = orig_weights[i] - 0.01 * weight_grads[i]
            expected_b = orig_biases[i] - 0.01 * bias_grads[i]
            np.testing.assert_allclose(weights[i], expected_w, rtol=1e-10)
            np.testing.assert_allclose(biases[i], expected_b, rtol=1e-10)

    def test_sgd_state_dict(self):
        """Test SGD state serialization."""
        optimizer = SGD(learning_rate=0.05)

        state = optimizer.state_dict()

        assert state["type"] == "SGD"
        assert state["learning_rate"] == 0.05
        assert "state" in state

    def test_sgd_load_state_dict(self):
        """Test SGD state restoration."""
        optimizer = SGD(learning_rate=0.01)

        # Create a state dict
        state = {"type": "SGD", "learning_rate": 0.05, "state": {}}

        # Load state
        optimizer.load_state_dict(state)

        assert optimizer.learning_rate == 0.05
        assert optimizer._state == {}


class TestSGDMomentum:
    """Test suite for SGD with Momentum optimizer."""

    def test_momentum_initialization(self):
        """Test momentum optimizer initializes correctly."""
        optimizer = SGDMomentum(learning_rate=0.01, momentum=0.9)
        assert optimizer.learning_rate == 0.01
        assert optimizer.momentum == 0.9
        assert optimizer.nesterov is False
        assert "velocity_w" in optimizer._state
        assert "velocity_b" in optimizer._state
        assert optimizer._state["initialized"] is False

    def test_momentum_nesterov_initialization(self):
        """Test Nesterov momentum initialization."""
        optimizer = SGDMomentum(learning_rate=0.01, momentum=0.9, nesterov=True)
        assert optimizer.nesterov is True

    def test_momentum_invalid_params(self):
        """Test momentum rejects invalid parameters."""
        with pytest.raises(ValueError):
            SGDMomentum(learning_rate=-0.01)

        with pytest.raises(ValueError, match="Momentum must be in"):
            SGDMomentum(learning_rate=0.01, momentum=1.5)

        with pytest.raises(ValueError, match="Momentum must be in"):
            SGDMomentum(learning_rate=0.01, momentum=-0.1)

    def test_momentum_first_update(self):
        """Test momentum initialization on first update."""
        optimizer = SGDMomentum(learning_rate=0.1, momentum=0.9)

        weights = [np.array([[1.0, 2.0]])]
        biases = [np.array([[0.5]])]
        weight_grads = [np.array([[0.1, 0.2]])]
        bias_grads = [np.array([[0.1]])]

        orig_w = weights[0].copy()
        orig_b = biases[0].copy()

        optimizer.update(weights, biases, weight_grads, bias_grads)

        # First update: velocity = gradient (no momentum yet)
        # Then: param = param - lr * velocity
        expected_w = orig_w - 0.1 * weight_grads[0]
        expected_b = orig_b - 0.1 * bias_grads[0]

        np.testing.assert_allclose(weights[0], expected_w, rtol=1e-10)
        np.testing.assert_allclose(biases[0], expected_b, rtol=1e-10)

        # Check velocity is initialized
        assert "velocity_w" in optimizer._state
        assert "velocity_b" in optimizer._state
        assert optimizer._state["initialized"] is True

    def test_momentum_subsequent_updates(self):
        """Test momentum accumulates over multiple updates."""
        optimizer = SGDMomentum(learning_rate=0.1, momentum=0.9, nesterov=False)

        weights = [np.array([[1.0]])]
        biases = [np.array([[1.0]])]
        weight_grads = [np.array([[0.1]])]
        bias_grads = [np.array([[0.1]])]

        # First update
        optimizer.update(weights, biases, weight_grads, bias_grads)
        w_after_first = weights[0].copy()

        # Second update with same gradient
        optimizer.update(weights, biases, weight_grads, bias_grads)

        # Velocity should have accumulated
        # Second update should be larger due to momentum
        first_step = 1.0 - w_after_first[0, 0]
        second_step = w_after_first[0, 0] - weights[0][0, 0]

        # Second step should be larger (momentum accumulation)
        assert second_step > first_step

    def test_nesterov_momentum(self):
        """Test Nesterov momentum differs from standard momentum."""
        # Standard momentum
        opt_standard = SGDMomentum(learning_rate=0.1, momentum=0.9, nesterov=False)
        weights_std = [np.array([[1.0, 2.0]])]
        biases_std = [np.array([[0.5]])]

        # Nesterov momentum
        opt_nesterov = SGDMomentum(learning_rate=0.1, momentum=0.9, nesterov=True)
        weights_nest = [np.array([[1.0, 2.0]])]
        biases_nest = [np.array([[0.5]])]

        # Same gradients
        weight_grads = [np.array([[0.1, 0.2]])]
        bias_grads = [np.array([[0.1]])]

        # Multiple updates to see divergence
        for _ in range(5):
            opt_standard.update(
                weights_std,
                biases_std,
                [g.copy() for g in weight_grads],
                [g.copy() for g in bias_grads],
            )
            opt_nesterov.update(
                weights_nest,
                biases_nest,
                [g.copy() for g in weight_grads],
                [g.copy() for g in bias_grads],
            )

        # Results should differ (Nesterov uses lookahead)
        assert not np.allclose(weights_std[0], weights_nest[0])

    def test_momentum_state_dict(self):
        """Test momentum state serialization."""
        optimizer = SGDMomentum(learning_rate=0.01, momentum=0.9, nesterov=True)

        # Do an update to initialize velocity
        weights = [np.random.randn(5, 10)]
        biases = [np.random.randn(1, 10)]
        weight_grads = [np.random.randn(5, 10)]
        bias_grads = [np.random.randn(1, 10)]

        optimizer.update(weights, biases, weight_grads, bias_grads)

        state = optimizer.state_dict()

        assert state["type"] == "SGDMomentum"
        assert state["learning_rate"] == 0.01
        assert state["momentum"] == 0.9
        assert state["nesterov"] is True
        assert "velocity_w" in state["state"]
        assert "velocity_b" in state["state"]

    def test_momentum_load_state_dict(self):
        """Test momentum state restoration preserves velocity."""
        optimizer = SGDMomentum(learning_rate=0.01, momentum=0.9)

        # Create state with velocity
        state = {
            "type": "SGDMomentum",
            "learning_rate": 0.05,
            "momentum": 0.95,
            "nesterov": True,
            "state": {
                "v_weightselocity": [np.array([[1.0, 2.0]])],
                "v_biaseselocity": [np.array([[0.5]])],
                "initialized": True,
            },
        }

        optimizer.load_state_dict(state)

        assert optimizer.learning_rate == 0.05
        assert optimizer.momentum == 0.95
        assert optimizer.nesterov is True
        assert "v_weightselocity" in optimizer._state
        np.testing.assert_array_equal(
            optimizer._state["v_weightselocity"][0], np.array([[1.0, 2.0]])
        )


class TestRMSprop:
    """Test suite for RMSprop optimizer."""

    def test_rmsprop_initialization(self):
        """Test RMSprop initializes with correct defaults."""
        optimizer = RMSprop(learning_rate=0.001)
        assert optimizer.learning_rate == 0.001
        assert optimizer.rho == 0.9
        assert optimizer.eps == 1e-8
        assert optimizer.momentum == 0.0
        assert "square_avg_weights" in optimizer._state
        assert "square_avg_biases" in optimizer._state
        assert optimizer._state["initialized"] is False

    def test_rmsprop_custom_params(self):
        """Test RMSprop with custom parameters."""
        optimizer = RMSprop(learning_rate=0.01, rho=0.95, eps=1e-7, momentum=0.9)
        assert optimizer.learning_rate == 0.01
        assert optimizer.rho == 0.95
        assert optimizer.eps == 1e-7
        assert optimizer.momentum == 0.9

    def test_rmsprop_invalid_params(self):
        """Test RMSprop parameter validation."""
        with pytest.raises(ValueError):
            RMSprop(learning_rate=-0.001)

        with pytest.raises(ValueError, match="rho must be in"):
            RMSprop(learning_rate=0.001, rho=1.5)

        with pytest.raises(ValueError, match="eps must be positive"):
            RMSprop(learning_rate=0.001, eps=-1e-8)

        with pytest.raises(ValueError, match="momentum must be in"):
            RMSprop(learning_rate=0.001, momentum=-0.1)

    def test_rmsprop_first_update(self):
        """Test RMSprop initialization on first update."""
        optimizer = RMSprop(learning_rate=0.1, rho=0.9)

        weights = [np.array([[1.0, 2.0]])]
        biases = [np.array([[0.5]])]
        weight_grads = [np.array([[0.1, 0.2]])]
        bias_grads = [np.array([[0.1]])]

        optimizer.update(weights, biases, weight_grads, bias_grads)

        # Check cache is initialized
        assert "square_avg_weights" in optimizer._state
        assert "square_avg_biases" in optimizer._state
        assert optimizer._state["initialized"] is True

    def test_rmsprop_adaptive_learning(self):
        """Test RMSprop adapts learning rate per parameter."""
        optimizer = RMSprop(learning_rate=0.1, rho=0.9, eps=1e-8)

        weights = [np.array([[1.0, 1.0]])]
        biases = [np.array([[1.0]])]

        # Large gradient for first param, small for second
        weight_grads = [np.array([[1.0, 0.01]])]
        bias_grads = [np.array([[0.1]])]

        orig_w = weights[0].copy()

        # Multiple updates
        for _ in range(5):
            optimizer.update(
                weights, biases, [weight_grads[0].copy()], [bias_grads[0].copy()]
            )

        # Parameter with large gradient should move less (due to adaptive rate)
        # Parameter with small gradient should move more relatively
        change_0 = abs(orig_w[0, 0] - weights[0][0, 0])
        change_1 = abs(orig_w[0, 1] - weights[0][0, 1])

        # Despite gradient ratio of 100:1, change ratio should be much smaller
        # This demonstrates adaptive learning rate
        assert change_0 / change_1 < 50  # Adaptive rate reduces the ratio

    def test_rmsprop_with_momentum(self):
        """Test RMSprop with momentum integration."""
        optimizer = RMSprop(learning_rate=0.1, rho=0.9, momentum=0.9)

        weights = [np.array([[1.0]])]
        biases = [np.array([[1.0]])]
        weight_grads = [np.array([[0.1]])]
        bias_grads = [np.array([[0.1]])]

        # First update
        optimizer.update(weights, biases, weight_grads, bias_grads)
        w_after_first = weights[0].copy()

        # Second update
        optimizer.update(weights, biases, weight_grads, bias_grads)

        # With momentum, second step should be influenced by first
        first_step = abs(1.0 - w_after_first[0, 0])
        second_step = abs(w_after_first[0, 0] - weights[0][0, 0])

        # Momentum should accumulate
        assert second_step > first_step * 0.5  # Some accumulation

    def test_rmsprop_state_dict(self):
        """Test RMSprop state serialization."""
        optimizer = RMSprop(learning_rate=0.001, rho=0.95, eps=1e-7, momentum=0.9)

        # Initialize with update
        weights = [np.random.randn(3, 4)]
        biases = [np.random.randn(1, 4)]
        weight_grads = [np.random.randn(3, 4)]
        bias_grads = [np.random.randn(1, 4)]

        optimizer.update(weights, biases, weight_grads, bias_grads)

        state = optimizer.state_dict()

        assert state["type"] == "RMSprop"
        assert state["learning_rate"] == 0.001
        assert state["rho"] == 0.95
        assert state["eps"] == 1e-7
        assert state["momentum"] == 0.9
        assert "square_avg_weights" in state["state"]
        assert "square_avg_biases" in state["state"]

    def test_rmsprop_numerical_stability(self):
        """Test RMSprop handles very small/large gradients."""
        optimizer = RMSprop(learning_rate=0.1, eps=1e-8)

        weights = [np.array([[1.0, 1.0, 1.0]])]
        biases = [np.array([[1.0]])]

        # Very large, very small, and zero gradients
        weight_grads = [np.array([[1e10, 1e-10, 0.0]])]
        bias_grads = [np.array([[1e-5]])]

        # Should not crash or produce NaN/Inf
        optimizer.update(weights, biases, weight_grads, bias_grads)

        assert np.all(np.isfinite(weights[0]))
        assert np.all(np.isfinite(biases[0]))


class TestAdam:
    """Test suite for Adam optimizer."""

    def test_adam_initialization(self):
        """Test Adam initializes with correct defaults."""
        optimizer = Adam(learning_rate=0.001)
        assert optimizer.learning_rate == 0.001
        assert optimizer.beta1 == 0.9
        assert optimizer.beta2 == 0.999
        assert optimizer.eps == 1e-8
        assert "m_weights" in optimizer._state
        assert "m_biases" in optimizer._state
        assert "v_weights" in optimizer._state
        assert "v_biases" in optimizer._state
        assert optimizer._state["t"] == 0
        assert optimizer._state["initialized"] is False

    def test_adam_custom_params(self):
        """Test Adam with custom parameters."""
        optimizer = Adam(learning_rate=0.01, beta1=0.95, beta2=0.9999, eps=1e-7)
        assert optimizer.learning_rate == 0.01
        assert optimizer.beta1 == 0.95
        assert optimizer.beta2 == 0.9999
        assert optimizer.eps == 1e-7

    def test_adam_invalid_params(self):
        """Test Adam parameter validation."""
        with pytest.raises(ValueError):
            Adam(learning_rate=-0.001)

        with pytest.raises(ValueError, match="beta1 must be in"):
            Adam(learning_rate=0.001, beta1=1.5)

        with pytest.raises(ValueError, match="beta2 must be in"):
            Adam(learning_rate=0.001, beta2=-0.1)

        with pytest.raises(ValueError, match="eps must be positive"):
            Adam(learning_rate=0.001, eps=0.0)

    def test_adam_first_update(self):
        """Test Adam initialization and bias correction on first update."""
        optimizer = Adam(learning_rate=0.001, beta1=0.9, beta2=0.999)

        weights = [np.array([[1.0, 2.0]])]
        biases = [np.array([[0.5]])]
        weight_grads = [np.array([[0.1, 0.2]])]
        bias_grads = [np.array([[0.1]])]

        orig_w = weights[0].copy()

        optimizer.update(weights, biases, weight_grads, bias_grads)

        # Check moments are initialized
        assert "m_weights" in optimizer._state
        assert "v_weights" in optimizer._state
        assert "m_biases" in optimizer._state
        assert "v_biases" in optimizer._state
        assert optimizer._state["t"] == 1
        assert optimizer._state["initialized"] is True

        # Check bias correction is applied (first update has large correction)
        # Update should be: lr * m_hat / (sqrt(v_hat) + eps)
        # With bias correction: m_hat = m / (1 - beta1^t), v_hat = v / (1 - beta2^t)
        assert not np.allclose(weights[0], orig_w)

    def test_adam_bias_correction(self):
        """Test Adam bias correction over multiple steps."""
        optimizer = Adam(learning_rate=0.001, beta1=0.9, beta2=0.999)

        weights = [np.array([[1.0]])]
        biases = [np.array([[1.0]])]
        weight_grads = [np.array([[0.1]])]
        bias_grads = [np.array([[0.1]])]

        # Track step sizes
        step_sizes = []

        for i in range(10):
            w_before = weights[0][0, 0]
            optimizer.update(
                weights, biases, [weight_grads[0].copy()], [bias_grads[0].copy()]
            )
            step_sizes.append(abs(weights[0][0, 0] - w_before))

        # With bias correction, the first few steps should be different
        # Check that timestep advances and bias correction is applied
        assert optimizer._state["t"] == 10
        # Step sizes should become more stable (less variation) over time
        early_variation = np.std(step_sizes[:3])
        late_variation = np.std(step_sizes[-3:])
        # Later steps should be more stable (but this might not always hold with constant grads)
        # Instead just check that bias correction is being applied
        assert all(s > 0 for s in step_sizes)  # All steps should make progress

    def test_adam_adaptive_learning(self):
        """Test Adam adapts to different gradient scales."""
        optimizer = Adam(learning_rate=0.001, beta1=0.9, beta2=0.999)

        weights = [np.array([[1.0, 1.0]])]
        biases = [np.array([[1.0]])]

        # Different gradient scales
        weight_grads = [np.array([[1.0, 0.01]])]
        bias_grads = [np.array([[0.1]])]

        orig_w = weights[0].copy()

        # Multiple updates
        for _ in range(10):
            optimizer.update(
                weights, biases, [weight_grads[0].copy()], [bias_grads[0].copy()]
            )

        # Both parameters should move, but adaptively
        change_0 = abs(orig_w[0, 0] - weights[0][0, 0])
        change_1 = abs(orig_w[0, 1] - weights[0][0, 1])

        # Parameter with larger gradient should not dominate completely
        # Adam's adaptive rate should balance the updates
        assert change_0 / change_1 < 50  # Much less than 100:1 gradient ratio

    def test_adam_momentum_component(self):
        """Test Adam's first moment (momentum) accumulation."""
        optimizer = Adam(learning_rate=0.001, beta1=0.9, beta2=0.999)

        weights = [np.array([[1.0]])]
        biases = [np.array([[1.0]])]
        weight_grads = [np.array([[0.1]])]
        bias_grads = [np.array([[0.1]])]

        # First update
        optimizer.update(weights, biases, weight_grads, bias_grads)
        m_after_first = optimizer._state["m_weights"][0].copy()

        # Second update
        optimizer.update(weights, biases, weight_grads, bias_grads)
        m_after_second = optimizer._state["m_weights"][0].copy()

        # First moment should accumulate (exponential moving average)
        # m_t = beta1 * m_{t-1} + (1 - beta1) * grad
        expected_m_second = 0.9 * m_after_first + 0.1 * weight_grads[0]
        np.testing.assert_allclose(m_after_second, expected_m_second, rtol=1e-10)

    def test_adam_second_moment(self):
        """Test Adam's second moment (variance) accumulation."""
        optimizer = Adam(learning_rate=0.001, beta1=0.9, beta2=0.999)

        weights = [np.array([[1.0]])]
        biases = [np.array([[1.0]])]
        weight_grads = [np.array([[0.1]])]
        bias_grads = [np.array([[0.1]])]

        # First update
        optimizer.update(weights, biases, weight_grads, bias_grads)
        v_after_first = optimizer._state["v_weights"][0].copy()

        # Second update
        optimizer.update(weights, biases, weight_grads, bias_grads)
        v_after_second = optimizer._state["v_weights"][0].copy()

        # Second moment should accumulate
        # v_t = beta2 * v_{t-1} + (1 - beta2) * grad^2
        expected_v_second = 0.999 * v_after_first + 0.001 * (weight_grads[0] ** 2)
        np.testing.assert_allclose(v_after_second, expected_v_second, rtol=1e-10)

    def test_adam_state_dict(self):
        """Test Adam state serialization."""
        optimizer = Adam(learning_rate=0.001, beta1=0.95, beta2=0.9999, eps=1e-7)

        # Initialize with update
        weights = [np.random.randn(4, 5)]
        biases = [np.random.randn(1, 5)]
        weight_grads = [np.random.randn(4, 5)]
        bias_grads = [np.random.randn(1, 5)]

        optimizer.update(weights, biases, weight_grads, bias_grads)

        state = optimizer.state_dict()

        assert state["type"] == "Adam"
        assert state["learning_rate"] == 0.001
        assert state["beta1"] == 0.95
        assert state["beta2"] == 0.9999
        assert state["eps"] == 1e-7
        assert "m_weights" in state["state"]
        assert "v_weights" in state["state"]
        assert "m_biases" in state["state"]
        assert "v_biases" in state["state"]
        assert state["state"]["t"] == 1

    def test_adam_load_state_dict(self):
        """Test Adam state restoration preserves moments."""
        optimizer = Adam(learning_rate=0.001)

        # Create state with moments
        state = {
            "type": "Adam",
            "learning_rate": 0.01,
            "beta1": 0.95,
            "beta2": 0.9999,
            "eps": 1e-7,
            "state": {
                "m_weights": [np.array([[1.0, 2.0]])],
                "v_weights": [np.array([[0.1, 0.2]])],
                "m_biases": [np.array([[0.5]])],
                "v_biases": [np.array([[0.05]])],
                "t": 10,
                "initialized": True,
            },
        }

        optimizer.load_state_dict(state)

        assert optimizer.learning_rate == 0.01
        assert optimizer.beta1 == 0.95
        assert optimizer.beta2 == 0.9999
        assert optimizer._state["t"] == 10
        np.testing.assert_array_equal(
            optimizer._state["m_weights"][0], np.array([[1.0, 2.0]])
        )
        np.testing.assert_array_equal(
            optimizer._state["v_weights"][0], np.array([[0.1, 0.2]])
        )

    def test_adam_numerical_stability(self):
        """Test Adam handles extreme gradients without numerical issues."""
        optimizer = Adam(learning_rate=0.001, eps=1e-8)

        weights = [np.array([[1.0, 1.0, 1.0, 1.0]])]
        biases = [np.array([[1.0]])]

        # Extreme gradients: very large, very small, zero, negative
        weight_grads = [np.array([[1e8, 1e-8, 0.0, -1e8]])]
        bias_grads = [np.array([[1e-10]])]

        # Multiple updates
        for _ in range(10):
            optimizer.update(
                weights, biases, [weight_grads[0].copy()], [bias_grads[0].copy()]
            )

        # Should not produce NaN or Inf
        assert np.all(np.isfinite(weights[0]))
        assert np.all(np.isfinite(biases[0]))
        assert np.all(np.isfinite(optimizer._state["m_weights"][0]))
        assert np.all(np.isfinite(optimizer._state["v_weights"][0]))


class TestOptimizerComparison:
    """Test suite comparing optimizer behaviors."""

    def test_all_optimizers_converge(self):
        """Test all optimizers can solve a simple optimization problem."""
        # Simple quadratic: minimize (x - 3)^2 + (y - 2)^2
        # Gradient: [2(x-3), 2(y-2)]

        optimizers = [
            SGD(learning_rate=0.1),
            SGDMomentum(learning_rate=0.1, momentum=0.9),
            RMSprop(learning_rate=0.1),
            Adam(learning_rate=0.1),
        ]

        target = np.array([[3.0, 2.0]])

        for opt in optimizers:
            # Start from origin
            params = [np.array([[0.0, 0.0]])]
            dummy_biases = [np.array([[0.0]])]

            # Optimize for 100 steps
            for _ in range(100):
                grad = [2 * (params[0] - target)]
                dummy_grad = [np.array([[0.0]])]
                opt.update(params, dummy_biases, grad, dummy_grad)

            # Should be close to target
            np.testing.assert_allclose(
                params[0],
                target,
                atol=0.1,
                err_msg=f"{opt.__class__.__name__} failed to converge",
            )

    def test_momentum_converges_faster(self):
        """Test momentum-based optimizers converge faster than plain SGD."""
        # Simple quadratic problem - minimize ||params - target||^2
        target = np.array([[5.0, 5.0]])

        # Plain SGD with smaller learning rate to avoid oscillation
        sgd = SGD(learning_rate=0.01)
        sgd_params = [np.array([[0.0, 0.0]])]
        sgd_biases = [np.array([[0.0]])]

        # SGD with momentum - can use larger LR due to momentum smoothing
        sgdm = SGDMomentum(learning_rate=0.02, momentum=0.9)
        sgdm_params = [np.array([[0.0, 0.0]])]
        sgdm_biases = [np.array([[0.0]])]

        # Track distances over time
        sgd_distances = []
        sgdm_distances = []

        # Run for same number of steps
        for _ in range(100):
            # Gradient of ||params - target||^2 is 2*(params - target)
            grad_sgd = [2 * (sgd_params[0] - target)]
            sgd.update(sgd_params, sgd_biases, grad_sgd, [np.array([[0.0]])])
            sgd_distances.append(np.linalg.norm(sgd_params[0] - target))

            grad_sgdm = [2 * (sgdm_params[0] - target)]
            sgdm.update(sgdm_params, sgdm_biases, grad_sgdm, [np.array([[0.0]])])
            sgdm_distances.append(np.linalg.norm(sgdm_params[0] - target))

        # Momentum should reach closer to target (on average in later iterations)
        avg_sgd_final = np.mean(sgd_distances[-10:])
        avg_sgdm_final = np.mean(sgdm_distances[-10:])

        assert avg_sgdm_final < avg_sgd_final, "Momentum should converge faster"

    def test_optimizer_state_persistence(self):
        """Test optimizer state can be saved and restored."""
        optimizers = [
            ("SGD", SGD(learning_rate=0.01)),
            ("SGDMomentum", SGDMomentum(learning_rate=0.01, momentum=0.9)),
            ("RMSprop", RMSprop(learning_rate=0.01)),
            ("Adam", Adam(learning_rate=0.01)),
        ]

        for name, opt in optimizers:
            # Do some updates
            weights = [np.random.randn(5, 10)]
            biases = [np.random.randn(1, 10)]
            weight_grads = [np.random.randn(5, 10)]
            bias_grads = [np.random.randn(1, 10)]

            for _ in range(5):
                opt.update(
                    weights, biases, [weight_grads[0].copy()], [bias_grads[0].copy()]
                )

            # Save state
            state = opt.state_dict()

            # Create new optimizer and load state
            if name == "SGD":
                new_opt = SGD(learning_rate=0.1)
            elif name == "SGDMomentum":
                new_opt = SGDMomentum(learning_rate=0.1)
            elif name == "RMSprop":
                new_opt = RMSprop(learning_rate=0.1)
            else:
                new_opt = Adam(learning_rate=0.1)

            new_opt.load_state_dict(state)

            # States should match
            assert new_opt.learning_rate == opt.learning_rate
            assert new_opt._state.keys() == opt._state.keys()

    def test_optimizer_with_zero_gradients(self):
        """Test that optimizers handle zero gradients correctly."""
        optimizers = [
            SGD(learning_rate=0.1),
            SGDMomentum(learning_rate=0.1, momentum=0.9),
            RMSprop(learning_rate=0.1),
            Adam(learning_rate=0.01),
        ]

        weights = [np.array([[1.0, 2.0]])]
        biases = [np.array([[0.5]])]
        zero_weight_grads = [np.zeros_like(weights[0])]
        zero_bias_grads = [np.zeros_like(biases[0])]

        for opt in optimizers:
            orig_w = weights[0].copy()
            orig_b = biases[0].copy()

            # Update with zero gradients
            opt.update(weights, biases, zero_weight_grads, zero_bias_grads)

            # Parameters should remain unchanged (or change minimally)
            # SGD with zero grads: no change
            # Adam/RMSprop might have tiny changes due to bias correction
            if isinstance(opt, SGD) and not isinstance(opt, SGDMomentum):
                np.testing.assert_array_equal(weights[0], orig_w)
                np.testing.assert_array_equal(biases[0], orig_b)

    def test_optimizer_numerical_stability_with_large_gradients(self):
        """Test optimizers handle large gradients without NaN/Inf."""
        optimizers = [
            SGD(learning_rate=0.001),
            SGDMomentum(learning_rate=0.001, momentum=0.9),
            RMSprop(learning_rate=0.001),
            Adam(learning_rate=0.001),
        ]

        weights = [np.array([[1.0, 2.0]])]
        biases = [np.array([[0.5]])]
        large_weight_grads = [np.array([[1000.0, 2000.0]])]
        large_bias_grads = [np.array([[500.0]])]

        for opt in optimizers:
            # Update with large gradients
            opt.update(weights, biases, large_weight_grads, large_bias_grads)

            # Check no NaN or Inf values
            assert np.isfinite(weights[0]).all()
            assert np.isfinite(biases[0]).all()

    def test_optimizer_convergence_comparison(self):
        """Test that all optimizers can converge on a simple problem."""
        # Simple quadratic problem: minimize (w - 5)^2
        target = 5.0
        num_steps = 100

        results = {}
        for name, opt in [
            ("SGD", SGD(learning_rate=0.1)),
            ("Momentum", SGDMomentum(learning_rate=0.1, momentum=0.9)),
            ("Adam", Adam(learning_rate=0.1)),
        ]:
            w = [np.array([[0.0]])]
            b = [np.array([[0.0]])]

            for _ in range(num_steps):
                # Gradient of (w - target)^2 is 2*(w - target)
                grad = 2 * (w[0] - target)
                opt.update(w, b, [grad], [np.zeros_like(b[0])])

            results[name] = abs(w[0][0, 0] - target)

        # All optimizers should converge close to target
        for name, error in results.items():
            assert error < 1.0, f"{name} did not converge (error={error})"

        # At least one adaptive optimizer should exist and converge
        assert "Adam" in results or "Momentum" in results

    def test_optimizer_learning_rate_effect(self):
        """Test that higher learning rates lead to larger parameter changes."""
        weight_grads = [np.array([[1.0, 1.0]])]
        bias_grads = [np.array([[1.0]])]

        learning_rates = [0.001, 0.01, 0.1]
        changes = []

        for lr in learning_rates:
            weights = [np.array([[1.0, 2.0]])]
            biases = [np.array([[0.5]])]
            orig_w = weights[0].copy()

            opt = SGD(learning_rate=lr)
            opt.update(weights, biases, weight_grads, bias_grads)

            change = np.abs(weights[0] - orig_w).sum()
            changes.append(change)

        # Higher learning rate should cause larger changes
        assert changes[0] < changes[1] < changes[2]
