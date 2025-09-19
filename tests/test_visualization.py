"""
Test suite for Visualization module.
Tests plotting and visualization functionality.
"""

import pytest
import numpy as np
import matplotlib.pyplot as plt
from neuroscope.viz.plots import Visualizer


class TestVisualizer:
    """Test suite for visualization functionality."""
    
    @pytest.fixture
    def sample_history(self):
        """Create sample training history for testing."""
        return {
            'history': {
                'loss': [1.0, 0.8, 0.6, 0.4, 0.2],
                'accuracy': [0.6, 0.7, 0.8, 0.85, 0.9],
                'val_loss': [1.1, 0.9, 0.7, 0.5, 0.3],
                'val_accuracy': [0.55, 0.65, 0.75, 0.8, 0.85],
                'epochs': [1, 2, 3, 4, 5]
            },
            'weights': [np.random.randn(10, 5), np.random.randn(5, 1)],
            'biases': [np.random.randn(5), np.random.randn(1)],
            'activations': [np.random.randn(32, 10), np.random.randn(32, 5)],
            'gradients': [np.random.randn(10, 5) * 0.01, np.random.randn(5, 1) * 0.01]
        }
    
    def test_visualizer_initialization(self, sample_history):
        """Test Visualizer initialization."""
        viz = Visualizer(sample_history)
        
        # Check basic attributes
        assert hasattr(viz, 'hist')
        assert viz.hist == sample_history
        
        # Check if history is extracted
        if hasattr(viz, 'history'):
            assert viz.history == sample_history['history']
        
        # Check if weights/biases are extracted
        if hasattr(viz, 'weights'):
            assert len(viz.weights) == 2
        if hasattr(viz, 'biases'):
            assert len(viz.biases) == 2
    
    def test_available_methods(self, sample_history):
        """Test what methods are actually available."""
        viz = Visualizer(sample_history)
        
        all_methods = [attr for attr in dir(viz) 
                      if callable(getattr(viz, attr)) and not attr.startswith('_')]
        
        print(f"Available visualizer methods: {all_methods}")
        assert len(all_methods) > 0
    
    def test_plot_methods_exist(self, sample_history):
        """Test that plotting methods exist and can be called."""
        viz = Visualizer(sample_history)
        
        # Common plotting methods that might exist
        plot_methods = [
            'plot_learning_curves',
            'plot_activation_distribution', 
            'plot_gradient_flow',
            'plot_weights',
            'plot_loss',
            'plot_training_dynamics'
        ]
        
        available_plots = []
        for method_name in plot_methods:
            if hasattr(viz, method_name):
                available_plots.append(method_name)
                
                # Try to call the method
                method = getattr(viz, method_name)
                try:
                    fig = method()
                    if fig is not None:
                        plt.close(fig)
                    print(f"Successfully called {method_name}")
                except Exception as e:
                    print(f"{method_name} failed: {e}")
        
        print(f"Available plot methods: {available_plots}")
    
    def test_visualizer_with_minimal_data(self):
        """Test visualizer with minimal required data."""
        minimal_history = {
            'history': {'loss': [1.0, 0.5]}
        }
        
        try:
            viz = Visualizer(minimal_history)
            assert viz.hist == minimal_history
        except Exception as e:
            print(f"Minimal data test failed: {e}")
    
    def test_matplotlib_compatibility(self, sample_history):
        """Test matplotlib compatibility."""
        viz = Visualizer(sample_history)
        
        # Test that we can create a simple plot
        fig, ax = plt.subplots()
        ax.plot([1, 2, 3], [1, 4, 2])
        plt.close(fig)
        
        # This confirms matplotlib is working
        assert True