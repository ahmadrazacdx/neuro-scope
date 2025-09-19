"""
Test the new convenient import structure for NeuroScope.

This demonstrates the improved import experience that allows direct function access.
"""

import sys
import os

# Add the src directory to Python path so we can import neuroscope
src_path = os.path.join(os.path.dirname(__file__), 'src')
if src_path not in sys.path:
    sys.path.insert(0, src_path)

# Test the new convenient import structure
print("Testing NeuroScope convenient imports...")
print(f"Added path: {src_path}")

try:
    # Test main package imports - direct function access
    from neuroscope import MLP, mse, bce, accuracy_binary, relu, he_init
    print("✅ Main package direct imports work!")
    print(f"   - mse function: {mse}")
    print(f"   - accuracy_binary function: {accuracy_binary}")
    print(f"   - relu function: {relu}")
    
    # Test MLP module imports
    from neuroscope.mlp import (
        MLP, mse, bce, cce, accuracy_binary, accuracy_multiclass, 
        relu, leaky_relu, sigmoid, he_init, xavier_init
    )
    print("✅ MLP module direct imports work!")
    
    # Test diagnostics imports
    from neuroscope.diagnostics import (
        PreTrainingAnalyzer, TrainingMonitor, 
        monitor_dead_neurons, monitor_vanishing_gradients,
        analyze_initial_loss, analyze_weight_init
    )
    print("✅ Diagnostics module direct imports work!")
    
    # Test viz imports
    from neuroscope.viz import (
        Visualizer, plot_learning_curves, plot_activation_hist,
        plot_gradient_hist, plot_weight_stats
    )
    print("✅ Visualization module direct imports work!")
    
    # Test aliases
    from neuroscope import PTA, TM, PTE, VIZ
    print("✅ Convenient aliases work!")
    print(f"   - PTA (PreTrainingAnalyzer): {PTA}")
    print(f"   - TM (TrainingMonitor): {TM}")
    print(f"   - PTE (PostTrainingEvaluator): {PTE}")
    print(f"   - VIZ (Visualizer): {VIZ}")
    
    print("\n🎉 All convenient imports are working perfectly!")
    print("\nNow you can use:")
    print("   loss = mse(y_true, y_pred)  # Instead of LossFunctions.mse()")
    print("   acc = accuracy_binary(y_true, y_pred)  # Instead of Metrics.accuracy_binary()")
    print("   activated = relu(z)  # Instead of ActivationFunctions.relu()")
    print("   plot_learning_curves(history)  # Direct plotting function")
    print("   analyzer = PTA(model)  # Instead of PreTrainingAnalyzer(model)")
    
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("\n💡 Solutions:")
    print("1. Install in development mode: pip install -e .")
    print("2. Or run from project root with: python -m pytest test_imports.py")
    print("3. Or use PYTHONPATH: set PYTHONPATH=src && python test_imports.py")
except Exception as e:
    print(f"❌ Error: {e}")