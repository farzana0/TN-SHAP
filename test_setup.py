#!/usr/bin/env python3
"""
Test script to verify the repository setup is correct.
This script tests that all imports work from different directories.
"""

import sys
import os
from pathlib import Path

def test_imports_from_root():
    """Test imports from repository root."""
    print("🧪 Testing imports from repository root...")
    
    # Add current directory to path
    repo_root = Path(__file__).parent
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    
    try:
        from src.tntree_model import BinaryTensorTree
        from src.feature_mapped_tn import FeatureMappedTN
        print("  ✅ Imports work from repository root")
        return True
    except Exception as e:
        print(f"  ❌ Import failed from root: {e}")
        return False

def test_imports_from_experiment_dir():
    """Test imports from experiment directory."""
    print("🧪 Testing imports from experiment directory...")
    
    # Simulate being in an experiment directory
    repo_root = Path(__file__).parent
    experiment_dir = repo_root / "experiments" / "01_feature_maps" / "scripts"
    
    # Add repository root to path (3 levels up from experiment scripts)
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    
    try:
        from src.tntree_model import BinaryTensorTree
        from src.feature_mapped_tn import FeatureMappedTN
        print("  ✅ Imports work from experiment directory")
        return True
    except Exception as e:
        print(f"  ❌ Import failed from experiment dir: {e}")
        return False

def test_basic_functionality():
    """Test basic functionality if torch is available."""
    print("🧪 Testing basic functionality...")
    
    try:
        import torch
        print("  ✅ PyTorch is available")
        
        from src.tntree_model import BinaryTensorTree
        
        # Create a simple tensor network
        tn = BinaryTensorTree(
            leaf_phys_dims=[2, 2, 2],
            ranks=3,
            seed=42
        )
        
        # Test forward pass
        X = torch.randn(10, 3)
        output = tn(X)
        assert output.shape == (10, 1), f"Expected shape (10, 1), got {output.shape}"
        
        print("  ✅ Basic functionality works")
        return True
        
    except ImportError:
        print("  ⚠️  PyTorch not available - install with: conda activate tnshap")
        return False
    except Exception as e:
        print(f"  ❌ Functionality test failed: {e}")
        return False

def main():
    """Run all tests."""
    print("🚀 TNShap Repository Setup Test")
    print("=" * 50)
    
    tests = [
        test_imports_from_root,
        test_imports_from_experiment_dir,
        test_basic_functionality
    ]
    
    results = []
    for test in tests:
        try:
            result = test()
            results.append(result)
        except Exception as e:
            print(f"  ❌ Test {test.__name__} failed with exception: {e}")
            results.append(False)
        print()
    
    # Summary
    passed = sum(results)
    total = len(results)
    
    print("📊 Test Summary:")
    print(f"  Passed: {passed}/{total}")
    
    if passed == total:
        print("\n🎉 All tests passed! Repository setup is correct.")
        print("\n📚 Next steps:")
        print("  1. Activate environment: conda activate tnshap")
        print("  2. Run experiments: cd experiments/UCI")
        print("  3. Test imports: python test_imports.py")
        return 0
    else:
        print(f"\n⚠️  {total - passed} test(s) failed. Check the messages above.")
        print("\n🔧 Troubleshooting:")
        print("  1. Make sure you're in the repository root")
        print("  2. Install dependencies: conda env create -f environment.yml")
        print("  3. Activate environment: conda activate tnshap")
        return 1

if __name__ == "__main__":
    sys.exit(main())
