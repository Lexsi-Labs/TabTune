#!/usr/bin/env python3

import sys
import os
import traceback

def test_package_structure():
    """Test the package structure and imports"""
    print("=== Testing Package Structure ===")
    
    # Test 1: Basic module import
    try:
        import tabtune
        print("✅ tabtune module imported successfully")
        print(f"   Version: {tabtune.__version__}")
        print(f"   Author: {tabtune.__author__}")
    except Exception as e:
        print(f"❌ Failed to import tabtune: {e}")
        return False
    
    # Test 2: Lazy import functionality
    try:
        pipeline_class = tabtune.TabularPipeline
        print("✅ TabularPipeline lazy import successful")
        print(f"   Class: {pipeline_class}")
    except Exception as e:
        print(f"❌ Failed lazy import of TabularPipeline: {e}")
        return False
    
    # Test 3: Direct import
    try:
        from tabtune.TabularPipeline.pipeline import TabularPipeline
        print("✅ TabularPipeline direct import successful")
    except Exception as e:
        print(f"❌ Failed direct import of TabularPipeline: {e}")
        return False
    
    # Test 4: Other components
    try:
        leaderboard_class = tabtune.TabularLeaderboard
        print("✅ TabularLeaderboard lazy import successful")
    except Exception as e:
        print(f"❌ Failed lazy import of TabularLeaderboard: {e}")
        return False
    
    return True

def test_pipeline_instantiation():
    """Test if TabularPipeline can be instantiated"""
    print("\n=== Testing Pipeline Instantiation ===")
    
    try:
        from tabtune import TabularPipeline
        
        # Test basic instantiation
        pipeline = TabularPipeline(
            model_name='TabPFN',  # Use a simpler model first
            tuning_strategy='inference'
        )
        print("✅ TabularPipeline instantiation successful")
        print(f"   Model: {pipeline.model_name}")
        print(f"   Strategy: {pipeline.tuning_strategy}")
        return True
        
    except Exception as e:
        print(f"❌ Failed to instantiate TabularPipeline: {e}")
        traceback.print_exc()
        return False

def test_data_loading():
    """Test data loading functionality"""
    print("\n=== Testing Data Loading ===")
    
    try:
        import pandas as pd
        import numpy as np
        from sklearn.model_selection import train_test_split
        
        # Create simple test data
        np.random.seed(42)
        X = pd.DataFrame({
            'feature1': np.random.randn(100),
            'feature2': np.random.randn(100),
            'feature3': np.random.randint(0, 3, 100)
        })
        y = np.random.randint(0, 2, 100)
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
        
        print("✅ Test data created successfully")
        print(f"   Training set: {X_train.shape}")
        print(f"   Test set: {X_test.shape}")
        return True, X_train, X_test, y_train, y_test
        
    except Exception as e:
        print(f"❌ Failed to create test data: {e}")
        return False, None, None, None, None

def test_pipeline_fit_predict():
    """Test pipeline fit and predict functionality"""
    print("\n=== Testing Pipeline Fit/Predict ===")
    
    try:
        from tabtune import TabularPipeline
        
        # Create test data
        success, X_train, X_test, y_train, y_test = test_data_loading()
        if not success:
            return False
        
        # Use a smaller subset for faster testing
        X_train_small = X_train[:50]
        y_train_small = y_train[:50]
        X_test_small = X_test[:20]
        y_test_small = y_test[:20]
        
        # Test with TabPFN (simpler model)
        pipeline = TabularPipeline(
            model_name='TabPFN',
            tuning_strategy='inference'
        )
        
        print("   Fitting pipeline...")
        pipeline.fit(X_train_small, y_train_small)
        print("✅ Pipeline fit successful")
        
        print("   Making predictions...")
        predictions = pipeline.predict(X_test_small)
        print("✅ Pipeline predict successful")
        print(f"   Predictions shape: {predictions.shape}")
        
        return True
        
    except Exception as e:
        print(f"❌ Failed pipeline fit/predict: {e}")
        traceback.print_exc()
        return False

def test_leaderboard():
    """Test TabularLeaderboard functionality"""
    print("\n=== Testing TabularLeaderboard ===")
    
    try:
        from tabtune import TabularLeaderboard
        
        # Create test data
        success, X_train, X_test, y_train, y_test = test_data_loading()
        if not success:
            return False
        
        # Use smaller subsets
        X_train_small = X_train[:50]
        y_train_small = y_train[:50]
        X_test_small = X_test[:20]
        y_test_small = y_test[:20]
        
        # Test leaderboard creation
        leaderboard = TabularLeaderboard(X_train_small, X_test_small, y_train_small, y_test_small)
        print("✅ TabularLeaderboard instantiation successful")
        
        # Test adding models
        leaderboard.add_model(
            model_name='TabPFN',
            tuning_strategy='inference'
        )
        print("✅ Model added to leaderboard successfully")
        
        return True
        
    except Exception as e:
        print(f"❌ Failed TabularLeaderboard test: {e}")
        traceback.print_exc()
        return False

def main():
    """Run all tests"""
    print("TabTune Comprehensive Test Suite")
    print("=" * 50)
    
    tests = [
        test_package_structure,
        test_pipeline_instantiation,
        test_pipeline_fit_predict,
        test_leaderboard
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        try:
            if test():
                passed += 1
        except Exception as e:
            print(f"❌ Test {test.__name__} failed with exception: {e}")
            traceback.print_exc()
    
    print(f"\n=== Test Results ===")
    print(f"Passed: {passed}/{total}")
    print(f"Success Rate: {passed/total*100:.1f}%")
    
    if passed == total:
        print("🎉 All tests passed!")
    else:
        print("⚠️  Some tests failed. Check the output above for details.")

if __name__ == "__main__":
    main()
