"""
Extended tests for Visualization functionality to increase coverage.
"""

import matplotlib

matplotlib.use("Agg")  # Use non-interactive backend for testing

import matplotlib.pyplot as plt
import numpy as np
import pytest

from neuroscope.viz.plots import Visualizer


class TestVisualizationExtended:
    """Extended tests for Visualization functionality."""

    @pytest.fixture
    def comprehensive_history(self):
        """Create comprehensive training history with all data types."""
        return {
            "method": "fit",
            "history": {
                "train_loss": [1.0, 0.8, 0.6, 0.4, 0.3],
                "train_acc": [0.5, 0.6, 0.7, 0.8, 0.85],
                "val_loss": [1.1, 0.9, 0.7, 0.5, 0.4],
                "val_acc": [0.45, 0.55, 0.65, 0.75, 0.8],
                "epochs": [1, 2, 3, 4, 5],
            },
            "weights": [np.random.randn(5, 8), np.random.randn(8, 1)],
            "biases": [np.random.randn(8), np.random.randn(1)],
            "activations": {
                "layer_0": np.random.randn(100, 5),
                "layer_1": np.random.randn(100, 8),
            },
            "gradients": {
                "layer_0": np.random.randn(5, 8),
                "layer_1": np.random.randn(8, 1),
            },
            "weight_stats_over_epochs": {
                "layer_0": {
                    "mean": [0.1, 0.12, 0.11, 0.13, 0.14],
                    "std": [0.2, 0.21, 0.19, 0.22, 0.23],
                },
                "layer_1": {
                    "mean": [0.05, 0.06, 0.07, 0.08, 0.09],
                    "std": [0.15, 0.16, 0.14, 0.17, 0.18],
                },
            },
            "activation_stats_over_epochs": {
                "layer_0": {
                    "mean": [1.1, 1.2, 1.0, 1.3, 1.4],
                    "std": [0.5, 0.6, 0.4, 0.7, 0.8],
                },
                "layer_1": {
                    "mean": [0.8, 0.9, 0.7, 1.0, 1.1],
                    "std": [0.3, 0.4, 0.2, 0.5, 0.6],
                },
            },
            "gradient_stats_over_epochs": {
                "layer_0": {
                    "mean": [1e-3, 1.2e-3, 0.9e-3, 1.1e-3, 1.3e-3],
                    "std": [2e-4, 2.1e-4, 1.9e-4, 2.2e-4, 2.3e-4],
                },
                "layer_1": {
                    "mean": [5e-4, 6e-4, 4e-4, 7e-4, 8e-4],
                    "std": [1e-4, 1.1e-4, 0.9e-4, 1.2e-4, 1.3e-4],
                },
            },
            "epoch_distributions": {
                "weights": {
                    "layer_0": [
                        np.random.randn(100),
                        np.random.randn(100),
                        np.random.randn(100),
                    ],
                    "layer_1": [
                        np.random.randn(50),
                        np.random.randn(50),
                        np.random.randn(50),
                    ],
                },
                "activations": {
                    "layer_0": [
                        np.random.randn(200),
                        np.random.randn(200),
                        np.random.randn(200),
                    ],
                    "layer_1": [
                        np.random.randn(150),
                        np.random.randn(150),
                        np.random.randn(150),
                    ],
                },
                "gradients": {
                    "layer_0": [
                        np.random.randn(80),
                        np.random.randn(80),
                        np.random.randn(80),
                    ],
                    "layer_1": [
                        np.random.randn(60),
                        np.random.randn(60),
                        np.random.randn(60),
                    ],
                },
            },
            "gradient_norms_over_epochs": {
                "layer_0": [1e-2, 1.1e-2, 0.9e-2, 1.2e-2, 1.3e-2],
                "layer_1": [5e-3, 6e-3, 4e-3, 7e-3, 8e-3],
            },
            "weight_update_ratios_over_epochs": {
                "layer_0": [1e-4, 1.1e-4, 0.9e-4, 1.2e-4, 1.3e-4],
                "layer_1": [5e-5, 6e-5, 4e-5, 7e-5, 8e-5],
            },
            "metric": "accuracy",
            "metric_display_name": "Accuracy",
        }

    @pytest.fixture
    def fit_fast_history(self):
        """Create fit_fast training history."""
        return {
            "method": "fit_fast",
            "history": {
                "train_loss": [1.0, 0.8, 0.6, 0.4, 0.3],
                "train_acc": [0.5, 0.6, 0.7, 0.8, 0.85],
                "val_loss": [1.1, 0.9, 0.7, 0.5, 0.4],
                "val_acc": [0.45, 0.55, 0.65, 0.75, 0.8],
            },
            "metric": "accuracy",
        }

    def test_visualizer_initialization_fit_fast(self, fit_fast_history):
        """Test Visualizer initialization with fit_fast history."""
        viz = Visualizer(fit_fast_history)
        assert viz.hist == fit_fast_history
        assert viz.history == fit_fast_history["history"]
        assert viz.metric == "accuracy"

    def test_visualizer_initialization_comprehensive(self, comprehensive_history):
        """Test Visualizer initialization with comprehensive history."""
        viz = Visualizer(comprehensive_history)
        assert viz.hist == comprehensive_history
        assert viz.history == comprehensive_history["history"]
        assert hasattr(viz, "activations")
        assert hasattr(viz, "gradients")
        assert hasattr(viz, "weight_stats_over_epochs")

    def test_plot_learning_curves_with_confidence_intervals(
        self, comprehensive_history
    ):
        """Test learning curves with confidence intervals."""
        viz = Visualizer(comprehensive_history)

        result = viz.plot_learning_curves(ci=True)
        assert result is None
        plt.close("all")

    def test_plot_learning_curves_without_markers(self, comprehensive_history):
        """Test learning curves without markers."""
        viz = Visualizer(comprehensive_history)

        result = viz.plot_learning_curves(markers=False)
        assert result is None
        plt.close("all")

    def test_plot_learning_curves_custom_figsize(self, comprehensive_history):
        """Test learning curves with custom figure size."""
        viz = Visualizer(comprehensive_history)

        result = viz.plot_learning_curves(figsize=(12, 6))
        assert result is None
        plt.close("all")

    def test_plot_learning_curves_custom_metric(self, comprehensive_history):
        """Test learning curves with custom metric name."""
        viz = Visualizer(comprehensive_history)

        result = viz.plot_learning_curves(metric="f1_score")
        assert result is None
        plt.close("all")

    def test_plot_activation_hist_with_kde(self, comprehensive_history):
        """Test activation histogram with KDE smoothing."""
        viz = Visualizer(comprehensive_history)

        result = viz.plot_activation_hist(kde=True)
        assert result is None
        plt.close("all")

    def test_plot_activation_hist_specific_epoch(self, comprehensive_history):
        """Test activation histogram for specific epoch."""
        viz = Visualizer(comprehensive_history)

        result = viz.plot_activation_hist(epoch=1)
        assert result is None
        plt.close("all")

    def test_plot_activation_hist_with_last_layer(self, comprehensive_history):
        """Test activation histogram including last layer."""
        viz = Visualizer(comprehensive_history)

        result = viz.plot_activation_hist(last_layer=True)
        assert result is None
        plt.close("all")

    def test_plot_gradient_hist_with_kde(self, comprehensive_history):
        """Test gradient histogram with KDE smoothing."""
        viz = Visualizer(comprehensive_history)

        result = viz.plot_gradient_hist(kde=True)
        assert result is None
        plt.close("all")

    def test_plot_weight_hist_with_kde(self, comprehensive_history):
        """Test weight histogram with KDE smoothing."""
        viz = Visualizer(comprehensive_history)

        result = viz.plot_weight_hist(kde=True)
        assert result is None
        plt.close("all")

    def test_plot_activation_stats_with_reference_lines(self, comprehensive_history):
        """Test activation stats with reference lines."""
        viz = Visualizer(comprehensive_history)

        result = viz.plot_activation_stats(reference_lines=True)
        assert result is None
        plt.close("all")

    def test_plot_gradient_stats_with_reference_lines(self, comprehensive_history):
        """Test gradient stats with reference lines."""
        viz = Visualizer(comprehensive_history)

        result = viz.plot_gradient_stats(reference_lines=True)
        assert result is None
        plt.close("all")

    def test_plot_weight_stats_with_reference_lines(self, comprehensive_history):
        """Test weight stats with reference lines."""
        viz = Visualizer(comprehensive_history)

        result = viz.plot_weight_stats(reference_lines=True)
        assert result is None
        plt.close("all")

    def test_plot_update_ratios(self, comprehensive_history):
        """Test weight update ratios plotting."""
        viz = Visualizer(comprehensive_history)

        result = viz.plot_update_ratios()
        assert result is None
        plt.close("all")

    def test_plot_update_ratios_with_reference_lines(self, comprehensive_history):
        """Test update ratios with reference lines."""
        viz = Visualizer(comprehensive_history)

        result = viz.plot_update_ratios(reference_lines=True)
        assert result is None
        plt.close("all")

    def test_plot_gradient_norms(self, comprehensive_history):
        """Test gradient norms plotting."""
        viz = Visualizer(comprehensive_history)

        result = viz.plot_gradient_norms()
        assert result is None
        plt.close("all")

    def test_empty_epoch_distributions(self):
        """Test plotting with empty epoch distributions."""
        history = {
            "method": "fit",
            "history": {"train_loss": [1.0, 0.8], "val_loss": [1.1, 0.9]},
            "epoch_distributions": {},
            "metric": "accuracy",
            "metric_display_name": "Accuracy",
        }

        viz = Visualizer(history)

        # Should handle empty distributions gracefully
        result = viz.plot_activation_hist()
        assert result is None
        plt.close("all")

    def test_missing_data_handling(self):
        """Test handling of missing data in various plots."""
        minimal_history = {
            "method": "fit",
            "history": {"train_loss": [1.0, 0.8], "val_loss": [1.1, 0.9]},
            "metric": "accuracy",
        }

        viz = Visualizer(minimal_history)

        # Should handle missing stats gracefully
        result = viz.plot_activation_stats()
        assert result is None
        plt.close("all")

        result = viz.plot_gradient_stats()
        assert result is None
        plt.close("all")

        result = viz.plot_weight_stats()
        assert result is None
        plt.close("all")

    def test_confidence_intervals_edge_cases(self, comprehensive_history):
        """Test confidence intervals with edge cases."""
        # Test with very short history
        short_history = comprehensive_history.copy()
        short_history["history"] = {
            "train_loss": [1.0, 0.8],
            "val_loss": [1.1, 0.9],
            "train_acc": [0.5, 0.6],
            "val_acc": [0.45, 0.55],
        }

        viz = Visualizer(short_history)

        # Should handle short sequences gracefully
        result = viz.plot_learning_curves(ci=True)
        assert result is None
        plt.close("all")

    def test_plot_save_functionality(self, comprehensive_history, tmp_path):
        """Test saving plots to file."""
        viz = Visualizer(comprehensive_history)

        save_path = tmp_path / "test_plot.png"
        result = viz.plot_learning_curves(save_path=str(save_path))

        assert result is None
        assert save_path.exists()
        plt.close("all")

    def test_different_history_formats(self):
        """Test visualization with different history formats."""
        # Test with simple format (no mean/std structure)
        simple_stats_history = {
            "method": "fit",
            "history": {"train_loss": [1.0, 0.8], "val_loss": [1.1, 0.9]},
            "weight_stats_over_epochs": {
                "layer_0": [0.1, 0.12],  # Simple list instead of dict
                "layer_1": [0.05, 0.06],
            },
            "metric": "accuracy",
        }

        viz = Visualizer(simple_stats_history)

        result = viz.plot_weight_stats()
        assert result is None
        plt.close("all")

    def test_large_epoch_distributions(self, comprehensive_history):
        """Test with large epoch distribution data."""
        # Create larger distributions
        large_distributions = {
            "weights": {
                "layer_0": [np.random.randn(1000) for _ in range(10)],
                "layer_1": [np.random.randn(500) for _ in range(10)],
            }
        }

        history = comprehensive_history.copy()
        history["epoch_distributions"] = large_distributions

        viz = Visualizer(history)

        result = viz.plot_weight_hist(kde=True)
        assert result is None
        plt.close("all")

    def test_invalid_epoch_selection(self, comprehensive_history):
        """Test with invalid epoch selection."""
        viz = Visualizer(comprehensive_history)

        # Test with epoch beyond available data
        result = viz.plot_activation_hist(epoch=100)
        assert result is None
        plt.close("all")

        # Test with negative epoch
        result = viz.plot_activation_hist(epoch=-1)
        assert result is None
        plt.close("all")

    def test_color_and_style_consistency(self, comprehensive_history):
        """Test color and style consistency across plots."""
        viz = Visualizer(comprehensive_history)

        # Test that colors are properly defined
        assert hasattr(viz, "colors")
        assert "train" in viz.colors
        assert "validation" in viz.colors
        assert "layers" in viz.colors

        # Test line style settings
        assert hasattr(viz, "line_style")
        assert "width" in viz.line_style
        assert "alpha" in viz.line_style

    def test_matplotlib_style_setup(self, comprehensive_history):
        """Test matplotlib style configuration."""
        viz = Visualizer(comprehensive_history)

        # Should have configured matplotlib settings
        import matplotlib.pyplot as plt

        # Test that style has been applied
        assert plt.rcParams["font.family"] == ["serif"]
        assert plt.rcParams["axes.spines.top"] is False
        assert plt.rcParams["axes.spines.right"] is False

    def test_plot_configuration_method(self, comprehensive_history):
        """Test internal plot configuration method."""
        viz = Visualizer(comprehensive_history)

        # Test _configure_plot method
        fig, ax = viz._configure_plot("Test Title", "X Label", "Y Label", (10, 6))

        assert fig is not None
        assert ax is not None
        assert ax.get_title() == "Test Title"
        assert ax.get_xlabel() == "X Label"
        assert ax.get_ylabel() == "Y Label"

        plt.close(fig)

    def test_epoch_distribution_plotting_edge_cases(self, comprehensive_history):
        """Test epoch distribution plotting with edge cases."""
        viz = Visualizer(comprehensive_history)

        # Test with empty epoch samples
        empty_dist_history = comprehensive_history.copy()
        empty_dist_history["epoch_distributions"] = {
            "weights": {
                "layer_0": [np.array([]), np.array([])],  # Empty arrays
            }
        }

        viz_empty = Visualizer(empty_dist_history)
        result = viz_empty.plot_weight_hist()
        assert result is None
        plt.close("all")

    def test_custom_activation_stats_data(self, comprehensive_history):
        """Test activation stats with custom data."""
        viz = Visualizer(comprehensive_history)

        custom_stats = {
            "layer_0": {"mean": [1.0, 1.1, 1.2], "std": [0.5, 0.6, 0.7]},
            "layer_1": {"mean": [0.8, 0.9, 1.0], "std": [0.3, 0.4, 0.5]},
        }

        result = viz.plot_activation_stats(activation_stats=custom_stats)
        assert result is None
        plt.close("all")

    def test_custom_update_ratios_data(self, comprehensive_history):
        """Test update ratios with custom data."""
        viz = Visualizer(comprehensive_history)

        custom_ratios = {
            "layer_0": [1e-4, 1.1e-4, 1.2e-4],
            "layer_1": [5e-5, 6e-5, 7e-5],
        }

        result = viz.plot_update_ratios(update_ratios=custom_ratios)
        assert result is None
        plt.close("all")

    def test_custom_gradient_norms_data(self, comprehensive_history):
        """Test gradient norms with custom data."""
        viz = Visualizer(comprehensive_history)

        custom_norms = {
            "layer_0": [1e-2, 1.1e-2, 1.2e-2],
            "layer_1": [5e-3, 6e-3, 7e-3],
        }

        result = viz.plot_gradient_norms(gradient_norms=custom_norms)
        assert result is None
        plt.close("all")
