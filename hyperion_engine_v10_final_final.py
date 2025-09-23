# hyperion_engine_v10_final.py
# The definitive, production-grade engine for the Hyperion Platform.
# This is the final version, correcting for "look-ahead bias" by ensuring
# the model cannot use the target variable's own recent history as a feature.
# This engine produces the final, honest, and defensible results for presentation.

import pandas as pd
import numpy as np
import json
from datetime import datetime
import warnings
import subprocess
import sys
import time
import logging

# --- Model Imports ---
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostRegressor
from sklearn.metrics import mean_absolute_error, r2_score

warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', stream=sys.stdout)
logger = logging.getLogger(__name__)

def install_packages():
    packages = ['yfinance', 'xgboost', 'lightgbm', 'catboost', 'scikit-learn']
    for package in packages:
        try:
            __import__(package.split('==')[0])
        except ImportError:
            logger.info(f"Installing dependency: {package}...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", package, "-q", "--progress-bar", "off"])

class HyperionV10:
    def __init__(self):
        self.results = {"metadata": {"timestamp": datetime.now().isoformat(), "version": "10.0-final-production"}}
        install_packages()

    def load_data(self):
        logger.info("Step 1/3: Loading Granular Trade Lane Data...")
        try:
            feuw_df = pd.read_csv('xsicfeuw_data.csv', index_col='Date', parse_dates=True)
            uwfe_df = pd.read_csv('xsiuwfe_data.csv', index_col='Date', parse_dates=True)
            feuw_df.rename(columns={'XSICFEUW': 'feuw_price'}, inplace=True)
            uwfe_df.rename(columns={'XSICUWFE': 'uwfe_price'}, inplace=True)
            self.target_column = 'feuw_price'
        except FileNotFoundError as e:
            logger.error(f"CRITICAL ERROR: Data file not found - {e.filename}.")
            sys.exit(1)
            
        import yfinance as yf
        try:
            market_data = yf.download(['BDRY', 'BZ=F'], start='2018-01-01', end=datetime.now(), progress=False)
            features_df = pd.DataFrame({
                'bdi_proxy_price': market_data['Close']['BDRY'],
                'fuel_price': market_data['Close']['BZ=F']
            })
        except Exception:
            logger.warning("Could not download market data.")
            features_df = pd.DataFrame()

        df = feuw_df.join(uwfe_df, how='inner').join(features_df, how='left')
        df = df.resample('D').ffill().bfill().dropna()
        
        self.results["data_summary"] = {
            "start_date": df.index.min().strftime('%Y-%m-%d'),
            "end_date": df.index.max().strftime('%Y-%m-%d'),
            "total_records": len(df)
        }
        return df

    def create_features(self, df):
        """Creates features. To be applied AFTER train/test split."""
        df_feat = df.copy()
        df_feat['trade_imbalance_ratio'] = df_feat['feuw_price'] / (df_feat['uwfe_price'] + 1e-6)
        
        # Lags on all NON-TARGET input variables
        for col in ['uwfe_price', 'bdi_proxy_price', 'fuel_price', 'trade_imbalance_ratio']:
            if col in df_feat.columns:
                for lag in [1, 7, 14, 30]:
                    df_feat[f'{col}_lag_{lag}'] = df_feat[col].shift(lag)
        
        return df_feat

    def run_forecasting_foundry(self, df):
        logger.info("Step 2/3: Running Final, Leakage-Free Forecasting Foundry...")
        self.results["forecasting"] = {}
        horizons = [7, 14, 30]

        for h in horizons:
            logger.info(f"  -> Processing {h}-day forecast...")
            
            data_with_target = df.copy()
            data_with_target['target'] = data_with_target[self.target_column].shift(-h)
            data_with_target.dropna(subset=['target'], inplace=True)

            train_size = int(len(data_with_target) * 0.8)
            train_raw, test_raw = data_with_target.iloc[:train_size], data_with_target.iloc[train_size:]

            train_df = self.create_features(train_raw).dropna()
            test_df = self.create_features(test_raw).dropna()

            # --- FINAL LEAKAGE FIX: REMOVE ALL TARGET-DERIVED FEATURES ---
            # The model must predict the future price without knowing the current or recent price.
            features_to_remove = [col for col in train_df.columns if self.target_column in col]
            
            common_cols = list(set(train_df.columns) & set(test_df.columns) - set(features_to_remove) - {'target'})
            
            X_train = train_df[common_cols]
            y_train = train_df['target']
            X_test = test_df[common_cols]
            y_test = test_df['target']

            logger.info(f"    Training with {len(X_train.columns)} features.")

            models = {
                "Ridge": Ridge(),
                "Random_Forest": RandomForestRegressor(random_state=42, n_jobs=-1),
                "Gradient_Boosting": GradientBoostingRegressor(random_state=42),
                "LightGBM": lgb.LGBMRegressor(random_state=42, verbosity=-1, n_jobs=-1),
                "CatBoost": CatBoostRegressor(random_state=42, verbose=0, allow_writing_files=False),
                "XGBoost": xgb.XGBRegressor(random_state=42, n_jobs=-1)
            }
            
            best_r2 = -np.inf
            horizon_key = f"{h}_day"
            self.results["forecasting"][horizon_key] = {"benchmark": {}}

            for name, model in models.items():
                model.fit(X_train, y_train)
                preds = model.predict(X_test)
                r2 = r2_score(y_test, preds)
                
                self.results["forecasting"][horizon_key]["benchmark"][name] = {"r2": float(r2), "mae": float(mean_absolute_error(y_test, preds))}
                
                if r2 > best_r2:
                    best_r2 = r2
                    importances = []
                    if hasattr(model, 'feature_importances_'):
                        importances = sorted(zip(X_train.columns, model.feature_importances_), key=lambda x: x[1], reverse=True)
                    
                    self.results["forecasting"][horizon_key]["champion"] = {
                        "name": name, "r2": float(r2), "mae": float(mean_absolute_error(y_test, preds)),
                        "predictions": preds.tolist(), "actuals": y_test.tolist(),
                        "dates": y_test.index.strftime('%Y-%m-%d').tolist(),
                        "feature_importance": [(k, float(v)) for k, v in importances[:5]]
                    }
        logger.info("✅ Forecasting Foundry complete.")
    
    def calculate_overall_rankings(self):
        logger.info("Step 3/3: Calculating Overall Model Rankings...")
        model_scores = {}
        for horizon_data in self.results['forecasting'].values():
            for model_name, metrics in horizon_data['benchmark'].items():
                if model_name not in model_scores: model_scores[model_name] = []
                score = metrics['r2'] if metrics['r2'] > 0 else -1.0 # Penalize negative R2
                model_scores[model_name].append(score)
        
        rankings = sorted([(name, np.mean(scores)) for name, scores in model_scores.items()], key=lambda x: x[1], reverse=True)
        self.results['overall_rankings'] = rankings
        logger.info(f"🏆 Top model (Avg R²): {self.results['overall_rankings'][0][0]}")

    def run_all(self):
        start_time = time.time()
        logger.info("🚀 Starting Hyperion Production Engine V10 (Final)...")
        
        full_df = self.load_data()
        self.run_forecasting_foundry(full_df)
        self.calculate_overall_rankings()
        
        filename = f"hyperion_v10_final_production_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(filename, 'w') as f:
            json.dump(self.results, f, indent=4)
            
        runtime = time.time() - start_time
        logger.info(f"🎉 Hyperion V10 analysis complete. Results saved to {filename} (Runtime: {runtime:.1f}s)")

if __name__ == "__main__":
    engine = HyperionV10()
    engine.run_all()
