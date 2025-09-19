"""
Simple test of NeuroScope imports without installation.
Run this from the NeuroScope project root directory.
"""

import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

# Test basic functionality
try:
    from neuroscope import MLP, PTA, TM, PTE, VIZ, mse, accuracy_binary, relu
    
    print("🎉 SUCCESS! All imports working!")
    print("\nAvailable convenient imports:")
    print("✅ MLP - Multi-Layer Perceptron")
    print("✅ PTA - PreTrainingAnalyzer") 
    print("✅ TM  - TrainingMonitor")
    print("✅ PTE - PostTrainingEvaluator")
    print("✅ VIZ - Visualizer")
    print("✅ mse - Mean Squared Error function")
    print("✅ accuracy_binary - Binary accuracy function")
    print("✅ relu - ReLU activation function")
    
    print("\nExample usage:")
    print(">>> model = MLP([2, 4, 1])")
    print(">>> analyzer = PTA(model)")
    print(">>> monitor = TM()")
    print(">>> evaluator = PTE(model)")
    print(">>> viz = VIZ(history)")
    
    # Test creating an MLP instance
    model = MLP([2, 4, 1])
    print(f"\n✅ Successfully created MLP model: {model}")
    print(f"   Layer dimensions: {model.layer_dims}")
    print(f"   Hidden activation: {model.hidden_activation}")
    
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()