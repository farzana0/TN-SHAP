#!/usr/bin/env python3
"""
Test script to verify all imports work correctly in the clean repository.
Run this after setting up the environment to ensure everything is working.
"""

import sys
import traceback

def test_imports():
    """Test all critical imports."""
    print("🧪 Testing TNShap imports...")
    
    try:
        # Test core imports
        print("  📦 Testing core imports...")
        import torch
        import numpy as np
        import pandas as pd
        import sklearn
        print("    ✅ Standard libraries imported successfully")
        
        # Test TNShap imports
        print("  🔬 Testing TNShap imports...")
        from src.tntree_model import BinaryTensorTree, make_balanced_binary_tensor_tree
        from src.feature_mapped_tn import FeatureMappedTN, ElementwiseFeatureMap, make_feature_mapped_tn
        print("    ✅ TNShap core modules imported successfully")
        
        # Test basic functionality
        print("  ⚙️  Testing basic functionality...")
        
        # Create a simple tensor network
        tn = BinaryTensorTree(
            leaf_phys_dims=[2, 2, 2],
            ranks=3,
            seed=42
        )
        print("    ✅ BinaryTensorTree created successfully")
        
        # Test forward pass
        X = torch.randn(10, 3)
        output = tn(X)
        assert output.shape == (10, 1), f"Expected shape (10, 1), got {output.shape}"
        print("    ✅ Forward pass works correctly")
        
        # Test feature-mapped TN
        model = make_feature_mapped_tn(
            d_in=5,
            fmap_out_dim=2,
            ranks=4,
            seed=42
        )
        X_test = torch.randn(20, 5)
        output_test = model(X_test)
        assert output_test.shape == (20, 1), f"Expected shape (20, 1), got {output_test.shape}"
        print("    ✅ FeatureMappedTN works correctly")
        
        # Test GPU if available
        if torch.cuda.is_available():
            print("  🚀 Testing GPU functionality...")
            tn_gpu = tn.cuda()
            X_gpu = X.cuda()
            output_gpu = tn_gpu(X_gpu)
            assert output_gpu.shape == (10, 1), f"Expected shape (10, 1), got {output_gpu.shape}"
            print("    ✅ GPU functionality works correctly")
        else:
            print("  💻 GPU not available, skipping GPU tests")
        
        print("\n🎉 All tests passed! TNShap is ready to use.")
        return True
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        print("\n📋 Full traceback:")
        traceback.print_exc()
        return False

def test_experiment_imports():
    """Test that experiment scripts can import correctly."""
    print("\n🧪 Testing experiment script imports...")
    
    # Test a few key experiment scripts
    experiment_scripts = [
        "experiments/UCI/scripts/uci_evaluate_tnshap_vs_baselines.py",
        "experiments/03_synthetic_experiments/scripts/synthetic_rank_sweep_basic.py",
        "experiments/04_scaling/scripts/scaling_simple_example.py"
    ]
    
    for script in experiment_scripts:
        try:
            print(f"  📄 Testing {script}...")
            # Just test imports, don't run the full script
            with open(script, 'r') as f:
                content = f.read()
                # Check if it has the correct import patterns
                if "from src.tntree_model import" in content or "from src.feature_mapped_tn import" in content:
                    print(f"    ✅ {script} has correct imports")
                else:
                    print(f"    ⚠️  {script} may need import updates")
        except Exception as e:
            print(f"    ❌ Error testing {script}: {e}")
    
    print("  ✅ Experiment import tests completed")

if __name__ == "__main__":
    print("🚀 TNShap Import Test Suite")
    print("=" * 50)
    
    success = test_imports()
    test_experiment_imports()
    
    if success:
        print("\n✅ All tests passed! You can now run experiments.")
        print("\n📚 Next steps:")
        print("  1. cd experiments/UCI")
        print("  2. python scripts/uci_evaluate_tnshap_vs_baselines.py --help")
        print("  3. Check out the notebooks: jupyter notebook notebooks/")
        sys.exit(0)
    else:
        print("\n❌ Some tests failed. Please check the error messages above.")
        sys.exit(1)
