# test_no_nan.py
# Basic tests for the Kalopathor Forecast Engine
# Ensures no NaN values in critical data paths

import pytest
import pandas as pd
import numpy as np
import sys
import os

# Import the engine from the same directory
from kalopathor_engine_v11_fixed import KalopathorEngine

class TestKalopathorEngine:
    
    def test_data_loading_no_nan(self):
        """Test that loaded data has no NaN values in critical columns."""
        engine = KalopathorEngine(quick_mode=True)
        
        # This will fail if data files don't exist, which is expected in CI
        try:
            df = engine.load_data()
            
            # Check that critical columns exist and have no NaN values
            critical_cols = ['feuw_price', 'uwfe_price']
            # Following exact instruction: assert all(df[cols].notna().all() for cols in feature_list)
            assert all(df[col].notna().all() for col in critical_cols), "Critical columns contain NaN values"
                
        except FileNotFoundError:
            pytest.skip("Data files not available for testing")
    
    def test_feature_creation_no_nan(self):
        """Test that feature creation doesn't introduce NaN values inappropriately."""
        engine = KalopathorEngine(quick_mode=True)
        
        try:
            df = engine.load_data()
            
            # Create a small subset for testing
            test_df = df.head(100).copy()
            test_df['target'] = test_df['feuw_price'].shift(-7)
            test_df = test_df.dropna(subset=['target'])
            
            # Create features
            features_df = engine.create_features(test_df)
            
            # Check that we have features
            feature_cols = [col for col in features_df.columns 
                          if col not in ['feuw_price', 'uwfe_price', 'target']]
            assert len(feature_cols) > 0, "No features created"
            
            # Check that features don't have excessive NaN values
            # (Some NaN is expected due to lagging, but not all)
            for col in feature_cols:
                nan_ratio = features_df[col].isna().sum() / len(features_df)
                assert nan_ratio < 0.5, f"Feature {col} has too many NaN values: {nan_ratio:.2%}"
                
        except FileNotFoundError:
            pytest.skip("Data files not available for testing")
    
    def test_no_target_leakage(self):
        """Test that features don't contain target variable leakage."""
        engine = KalopathorEngine(quick_mode=True)
        
        try:
            df = engine.load_data()
            test_df = df.head(100).copy()
            test_df['target'] = test_df['feuw_price'].shift(-7)
            test_df = test_df.dropna(subset=['target'])
            
            features_df = engine.create_features(test_df)
            
            # Check that no features contain the target column name
            feature_cols = [col for col in features_df.columns 
                          if col not in ['feuw_price', 'uwfe_price', 'target']]
            
            for col in feature_cols:
                assert 'feuw_price' not in col, f"Feature {col} contains target variable name"
                
        except FileNotFoundError:
            pytest.skip("Data files not available for testing")
    
    def test_quick_mode_functionality(self):
        """Test that quick mode works without errors."""
        engine = KalopathorEngine(quick_mode=True)
        
        try:
            df = engine.load_data()
            
            # Test that quick mode only processes 7-day horizon
            engine.run_forecasting_foundry(df, forecast_horizon=7)
            
            # Check that results contain expected structure
            assert 'forecasting' in engine.results
            assert '7_day' in engine.results['forecasting']
            assert 'champion' in engine.results['forecasting']['7_day']
            
        except FileNotFoundError:
            pytest.skip("Data files not available for testing")

if __name__ == "__main__":
    pytest.main([__file__])
