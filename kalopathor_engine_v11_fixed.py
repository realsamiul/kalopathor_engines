# kalopathor_engine.py
# The Kalopathor Forecast Engine - Leakage-Free, Production-Ready
# Based on feedback from comprehensive code audit
# This version fixes all identified issues and adds production features

import pandas as pd
import numpy as np
import json
import os
import tempfile
import argparse
from datetime import datetime
import warnings
import subprocess
import sys
import time
import logging

# Install packages BEFORE importing them
def install_packages():
    packages = ['yfinance', 'xgboost', 'lightgbm', 'catboost', 'scikit-learn', 'shap']
    for package in packages:
        try:
            __import__(package.split('==')[0])
        except ImportError:
            logging.info(f"Installing dependency: {package}...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", package, "-q", "--progress-bar", "off"])

# Install packages first
install_packages()

# Now import the packages
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, r2_score
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostRegressor
import yfinance as yf

# Set up CatBoost for Windows compatibility
os.environ["CATBOOST_DATA_DIR"] = tempfile.mkdtemp()

warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', stream=sys.stdout)
logger = logging.getLogger(__name__)

class KalopathorEngine:
    def __init__(self, quick_mode=False):
        self.results = {"metadata": {"timestamp": datetime.now().isoformat(), "version": "kalopathor-1.0", "quick_mode": quick_mode}}
        self.quick_mode = quick_mode
        self.target_column = 'feuw_price'

    def load_data(self):
        logger.info("Step 1/3: Loading Granular Trade Lane Data...")
        try:
            feuw_df = pd.read_csv('xsicfeuw_data.csv', index_col='Date', parse_dates=True)
            uwfe_df = pd.read_csv('xsiuwfe_data.csv', index_col='Date', parse_dates=True)
            feuw_df.rename(columns={'XSICFEUW': 'feuw_price'}, inplace=True)
            uwfe_df.rename(columns={'XSICUWFE': 'uwfe_price'}, inplace=True)
        except FileNotFoundError as e:
            logger.error(f"CRITICAL ERROR: Data file not found - {e.filename}.")
            sys.exit(1)
            
        # Download market data with proper error handling
        try:
            market_data = yf.download(['BDRY', 'BZ=F'], start='2018-01-01', end=datetime.now(), progress=False)
            features_df = pd.DataFrame({
                'bdi_proxy_price': market_data['Close']['BDRY'],
                'fuel_price': market_data['Close']['BZ=F']
            })
        except Exception as e:
            logger.warning(f"Could not download market data: {e}")
            features_df = pd.DataFrame()

        df = feuw_df.join(uwfe_df, how='inner').join(features_df, how='left')
        df = df.resample('D').ffill().bfill().dropna()
        
        # Check if market features are available
        if features_df.empty:
            logger.warning("No market features – reverting to price-only benchmark.")
        
        self.results["data_summary"] = {
            "start_date": df.index.min().strftime('%Y-%m-%d'),
            "end_date": df.index.max().strftime('%Y-%m-%d'),
            "total_records": len(df),
            "market_features_available": not features_df.empty
        }
        return df

    def create_features(self, df):
        """Creates features with proper lagging to avoid leakage."""
        df_feat = df.copy()
        
        # FIXED: Create lagged trade imbalance ratio to avoid leakage
        # Use lagged values to prevent current-day target leakage
        # Following exact instruction: ratio = uwfe_price_shifted = df_feat['uwfe_price'].shift(1)
        uwfe_price_shifted = df_feat['uwfe_price'].shift(1)
        df_feat['trade_imbalance_ratio_lag_1'] = (df_feat['feuw_price'].shift(1) / 
                                                 (uwfe_price_shifted + 1e-6))
        
        # Lags on all NON-TARGET input variables
        for col in ['uwfe_price', 'bdi_proxy_price', 'fuel_price', 'trade_imbalance_ratio_lag_1']:
            if col in df_feat.columns:
                for lag in [1, 7, 14, 30]:
                    df_feat[f'{col}_lag_{lag}'] = df_feat[col].shift(lag)
        
        return df_feat

    def run_forecasting_foundry(self, df, forecast_horizon=None):
        logger.info("Step 2/3: Running Leakage-Free Forecasting Foundry...")
        self.results["forecasting"] = {}
        
        # Use specific horizon if provided, otherwise use all
        if forecast_horizon:
            horizons = [forecast_horizon]
        elif self.quick_mode:
            horizons = [7]  # Quick mode only uses 7-day
        else:
            horizons = [7, 14, 30]

        for h in horizons:
            logger.info(f"  -> Processing {h}-day forecast...")
            
            data_with_target = df.copy()
            data_with_target['target'] = data_with_target[self.target_column].shift(-h)
            data_with_target.dropna(subset=['target'], inplace=True)

            # Use TimeSeriesSplit for proper time series validation
            tscv = TimeSeriesSplit(n_splits=5)
            
            # Get features
            data_with_features = self.create_features(data_with_target).dropna()
            
            # Remove target-derived features (leakage prevention)
            features_to_remove = [col for col in data_with_features.columns if self.target_column in col]
            feature_cols = [col for col in data_with_features.columns 
                          if col not in features_to_remove and col != 'target']
            
            X = data_with_features[feature_cols]
            y = data_with_features['target']
            
            logger.info(f"    Training with {len(X.columns)} features.")

            # Models with proper reproducibility
            models = {
                "Ridge": Ridge(),
                "Random_Forest": RandomForestRegressor(random_state=42, n_jobs=-1),
                "Gradient_Boosting": GradientBoostingRegressor(random_state=42),
                "LightGBM": lgb.LGBMRegressor(random_state=42, verbosity=-1, n_jobs=-1, force_col_wise=True),
                "CatBoost": CatBoostRegressor(random_state=42, verbose=0, allow_writing_files=False),
                "XGBoost": xgb.XGBRegressor(random_state=42, n_jobs=-1)
            }
            
            # Quick mode only uses Ridge
            if self.quick_mode:
                models = {"Ridge": Ridge()}
            
            horizon_key = f"{h}_day"
            self.results["forecasting"][horizon_key] = {"benchmark": {}, "cv_scores": {}}

            # Time series cross-validation
            cv_scores = {name: [] for name in models.keys()}
            
            for train_idx, test_idx in tscv.split(X):
                X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
                y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
                
                for name, model in models.items():
                    model.fit(X_train, y_train)
                    preds = model.predict(X_test)
                    r2 = r2_score(y_test, preds)
                    cv_scores[name].append(r2)
            
            # Calculate final metrics on last split (most recent data)
            X_train_final, X_test_final = X.iloc[train_idx], X.iloc[test_idx]
            y_train_final, y_test_final = y.iloc[train_idx], y.iloc[test_idx]
            
            best_r2 = -np.inf
            for name, model in models.items():
                model.fit(X_train_final, y_train_final)
                preds = model.predict(X_test_final)
                r2 = r2_score(y_test_final, preds)
                mae = mean_absolute_error(y_test_final, preds)
                
                # Store CV statistics
                cv_mean = np.mean(cv_scores[name])
                cv_std = np.std(cv_scores[name])
                
                self.results["forecasting"][horizon_key]["benchmark"][name] = {
                    "r2": float(r2), 
                    "mae": float(mae),
                    "cv_r2_mean": float(cv_mean),
                    "cv_r2_std": float(cv_std)
                }
                
                if r2 > best_r2:
                    best_r2 = r2
                    importances = []
                    if hasattr(model, 'feature_importances_'):
                        importances = sorted(zip(X_train_final.columns, model.feature_importances_), 
                                           key=lambda x: x[1], reverse=True)
                    elif hasattr(model, 'coef_'):
                        # For Ridge, use absolute standardized coefficients
                        coef_abs = np.abs(model.coef_)
                        importances = sorted(zip(X_train_final.columns, coef_abs), 
                                           key=lambda x: x[1], reverse=True)
                    
                    self.results["forecasting"][horizon_key]["champion"] = {
                        "name": name, 
                        "r2": float(r2), 
                        "mae": float(mae),
                        "cv_r2_mean": float(cv_mean),
                        "cv_r2_std": float(cv_std),
                        "predictions": preds.tolist(), 
                        "actuals": y_test_final.tolist(),
                        "dates": y_test_final.index.strftime('%Y-%m-%d').tolist(),
                        "feature_importance": [(k, float(v)) for k, v in importances[:5]]
                    }
                    
                    # Save pred vs actual CSV
                    pred_df = pd.DataFrame({
                        'date': y_test_final.index,
                        'actual': y_test_final.values,
                        'predicted': preds
                    })
                    csv_filename = f"kalopathor_{h}day_predictions.csv"
                    pred_df.to_csv(csv_filename, index=False)
                    logger.info(f"    Saved predictions to {csv_filename}")
        
        logger.info("✅ Forecasting Foundry complete.")
    
    def calculate_overall_rankings(self):
        logger.info("Step 3/3: Calculating Overall Model Rankings...")
        model_scores = {}
        for horizon_data in self.results['forecasting'].values():
            for model_name, metrics in horizon_data['benchmark'].items():
                if model_name not in model_scores: 
                    model_scores[model_name] = []
                # Use CV mean for ranking
                score = metrics['cv_r2_mean'] if metrics['cv_r2_mean'] > 0 else -1.0
                model_scores[model_name].append(score)
        
        rankings = sorted([(name, np.mean(scores)) for name, scores in model_scores.items()], 
                         key=lambda x: x[1], reverse=True)
        self.results['overall_rankings'] = rankings
        logger.info(f"🏆 Top model (Avg CV R²): {self.results['overall_rankings'][0][0]}")

    def run_all(self, forecast_horizon=None, output_file=None):
        start_time = time.time()
        mode_str = "Quick" if self.quick_mode else "Full"
        logger.info(f"🚀 Starting Kalopathor Forecast Engine ({mode_str} Mode)...")
        
        full_df = self.load_data()
        self.run_forecasting_foundry(full_df, forecast_horizon)
        self.calculate_overall_rankings()
        
        if output_file:
            filename = output_file
        else:
            mode_suffix = "_quick" if self.quick_mode else ""
            filename = f"kalopathor_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}{mode_suffix}.json"
            
        with open(filename, 'w') as f:
            json.dump(self.results, f, indent=4)
            
        runtime = time.time() - start_time
        logger.info(f"🎉 Kalopathor analysis complete. Results saved to {filename} (Runtime: {runtime:.1f}s)")

def main():
    parser = argparse.ArgumentParser(description='Kalopathor Forecast Engine')
    parser.add_argument('--forecast', type=int, choices=[7, 14, 30], 
                       help='Specific forecast horizon (7, 14, or 30 days)')
    parser.add_argument('--quick', action='store_true', 
                       help='Quick mode: Ridge only, 7-day horizon')
    parser.add_argument('--output', type=str, 
                       help='Output JSON filename')
    
    args = parser.parse_args()
    
    # Quick mode overrides forecast horizon
    if args.quick:
        forecast_horizon = 7
    else:
        forecast_horizon = args.forecast
    
    engine = KalopathorEngine(quick_mode=args.quick)
    engine.run_all(forecast_horizon=forecast_horizon, output_file=args.output)

if __name__ == "__main__":
    main()
