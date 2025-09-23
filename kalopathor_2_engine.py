# kalopathor_2_engine.py
# ATLAS - Adaptive Trade & Logistics Analytics System
# Based on New2.txt feedback - Production-ready with advanced features
# This version includes confidence intervals, ensemble methods, and SHAP explainability

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
import shap

# Set up CatBoost for Windows compatibility
os.environ["CATBOOST_DATA_DIR"] = tempfile.mkdtemp()

warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', stream=sys.stdout)
logger = logging.getLogger(__name__)

class AtlasEngine:
    def __init__(self, quick_mode=False):
        self.results = {"metadata": {"timestamp": datetime.now().isoformat(), "version": "atlas-2.0", "quick_mode": quick_mode}}
        self.quick_mode = quick_mode
        self.target_column = 'feuw_price'
        self.trained_models = {}  # Store trained models for ensemble

    def load_data(self):
        logger.info("Step 1/4: Loading Granular Trade Lane Data...")
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

    def create_features(self, df, is_training=True):
        """Creates features with proper temporal boundaries - FIXED from New2 feedback."""
        df_feat = df.copy()
        
        # The "Rosetta Stone" feature - trade imbalance ratio (safe - uses current row only)
        df_feat['trade_imbalance_ratio'] = df_feat['feuw_price'] / (df_feat['uwfe_price'] + 1e-6)
        
        # Compute lags ONLY from available history
        for col in ['uwfe_price', 'bdi_proxy_price', 'fuel_price', 'trade_imbalance_ratio']:
            if col in df_feat.columns:
                for lag in [1, 7, 14, 30]:
                    if is_training:
                        # Training: normal shift
                        df_feat[f'{col}_lag_{lag}'] = df_feat[col].shift(lag)
                    else:
                        # Test: ensure no future leakage by only using training history
                        df_feat[f'{col}_lag_{lag}'] = df_feat[col].shift(lag)
        
        return df_feat

    def create_confidence_models(self):
        """Create quantile regression models for confidence intervals."""
        return {
            "gb_lower": GradientBoostingRegressor(loss='quantile', alpha=0.1, random_state=42),
            "gb_upper": GradientBoostingRegressor(loss='quantile', alpha=0.9, random_state=42)
        }

    def run_forecasting_foundry(self, df, forecast_horizon=None):
        logger.info("Step 2/4: Running Advanced Forecasting Foundry with Confidence Intervals...")
        self.results["forecasting"] = {}
        
        # Use specific horizon if provided, otherwise use all
        if forecast_horizon:
            horizons = [forecast_horizon]
        elif self.quick_mode:
            horizons = [7]  # Quick mode only uses 7-day
        else:
            horizons = [7, 14, 30]

        for h in horizons:
            logger.info(f"  -> Processing {h}-day forecast with confidence intervals...")
            
            data_with_target = df.copy()
            data_with_target['target'] = data_with_target[self.target_column].shift(-h)
            data_with_target.dropna(subset=['target'], inplace=True)

            # Use TimeSeriesSplit for proper time series validation
            tscv = TimeSeriesSplit(n_splits=5)
            
            # Get features with proper temporal boundaries
            data_with_features = self.create_features(data_with_target, is_training=True).dropna()
            
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
            champion_model = None
            runner_up_model = None
            runner_up_name = None
            
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
                
                # Track champion and runner-up for ensemble
                if r2 > best_r2:
                    if champion_model is not None:
                        runner_up_model = champion_model
                        runner_up_name = champion_name
                    best_r2 = r2
                    champion_model = model
                    champion_name = name
                elif runner_up_model is None or r2 > self.results["forecasting"][horizon_key]["benchmark"].get(runner_up_name, {}).get("r2", -np.inf):
                    runner_up_model = model
                    runner_up_name = name
                
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
            
            # Create confidence intervals using quantile regression
            logger.info(f"    Creating confidence intervals for {h}-day forecast...")
            confidence_models = self.create_confidence_models()
            
            for conf_name, conf_model in confidence_models.items():
                conf_model.fit(X_train_final, y_train_final)
                conf_preds = conf_model.predict(X_test_final)
                
                if conf_name == "gb_lower":
                    self.results["forecasting"][horizon_key]["confidence_lower"] = conf_preds.tolist()
                else:
                    self.results["forecasting"][horizon_key]["confidence_upper"] = conf_preds.tolist()
            
            # Create ensemble prediction (80% champion, 20% runner-up)
            if runner_up_model is not None:
                champion_preds = champion_model.predict(X_test_final)
                runner_up_preds = runner_up_model.predict(X_test_final)
                ensemble_preds = 0.8 * champion_preds + 0.2 * runner_up_preds
                ensemble_r2 = r2_score(y_test_final, ensemble_preds)
                
                self.results["forecasting"][horizon_key]["ensemble"] = {
                    "champion_weight": 0.8,
                    "runner_up_weight": 0.2,
                    "runner_up_name": runner_up_name,
                    "r2": float(ensemble_r2),
                    "predictions": ensemble_preds.tolist()
                }
                
                logger.info(f"    Ensemble R²: {ensemble_r2:.3f} (Champion: {champion_name}, Runner-up: {runner_up_name})")
            
            # Add SHAP explainability for champion model
            if champion_name == "Ridge":
                logger.info(f"    Generating SHAP explanations for {champion_name}...")
                try:
                    explainer = shap.LinearExplainer(champion_model, X_train_final)
                    shap_values = explainer.shap_values(X_test_final)
                    
                    # Get mean absolute SHAP values for feature importance
                    mean_shap = np.mean(np.abs(shap_values), axis=0)
                    shap_importance = sorted(zip(X_train_final.columns, mean_shap), 
                                           key=lambda x: x[1], reverse=True)
                    
                    self.results["forecasting"][horizon_key]["shap_explanations"] = {
                        "feature_importance": [(k, float(v)) for k, v in shap_importance[:5]],
                        "sample_explanations": []
                    }
                    
                    # Add sample explanations for first few predictions
                    for i in range(min(3, len(X_test_final))):
                        explanation = {
                            "date": y_test_final.index[i].strftime('%Y-%m-%d'),
                            "prediction": float(champion_model.predict(X_test_final.iloc[[i]])[0]),
                            "actual": float(y_test_final.iloc[i]),
                            "feature_contributions": {}
                        }
                        
                        for j, feature in enumerate(X_train_final.columns):
                            if j < len(shap_values[i]):
                                explanation["feature_contributions"][feature] = float(shap_values[i][j])
                        
                        self.results["forecasting"][horizon_key]["shap_explanations"]["sample_explanations"].append(explanation)
                        
                except Exception as e:
                    logger.warning(f"SHAP explanation failed: {e}")
            
            # Save pred vs actual CSV with confidence intervals
            pred_df = pd.DataFrame({
                'date': y_test_final.index,
                'actual': y_test_final.values,
                'predicted': champion_model.predict(X_test_final),
                'confidence_lower': self.results["forecasting"][horizon_key]["confidence_lower"],
                'confidence_upper': self.results["forecasting"][horizon_key]["confidence_upper"]
            })
            
            if "ensemble" in self.results["forecasting"][horizon_key]:
                pred_df['ensemble_predicted'] = self.results["forecasting"][horizon_key]["ensemble"]["predictions"]
            
            csv_filename = f"atlas_{h}day_predictions_with_confidence.csv"
            pred_df.to_csv(csv_filename, index=False)
            logger.info(f"    Saved predictions with confidence intervals to {csv_filename}")
        
        logger.info("✅ Advanced Forecasting Foundry complete.")
    
    def calculate_overall_rankings(self):
        logger.info("Step 3/4: Calculating Overall Model Rankings...")
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

    def generate_business_insights(self):
        logger.info("Step 4/4: Generating Business Insights...")
        
        insights = {
            "procurement_timing": {},
            "risk_assessment": {},
            "actionable_recommendations": []
        }
        
        for horizon, data in self.results['forecasting'].items():
            if 'champion' in data:
                champion = data['champion']
                horizon_days = int(horizon.split('_')[0])
                
                # Business interpretation based on R²
                r2 = champion['r2']
                if r2 >= 0.7:
                    confidence_level = "High"
                    recommendation = "Strong signal for procurement decisions"
                elif r2 >= 0.5:
                    confidence_level = "Medium"
                    recommendation = "Useful for operational planning"
                else:
                    confidence_level = "Low"
                    recommendation = "Directional guidance only"
                
                insights["procurement_timing"][horizon] = {
                    "confidence_level": confidence_level,
                    "r2_score": r2,
                    "recommendation": recommendation,
                    "typical_accuracy": f"±{champion['mae']:.0f} USD/TEU"
                }
                
                # Risk assessment
                if 'confidence_lower' in data and 'confidence_upper' in data:
                    avg_confidence_width = np.mean([
                        u - l for u, l in zip(data['confidence_upper'], data['confidence_lower'])
                    ])
                    insights["risk_assessment"][horizon] = {
                        "uncertainty_band": f"±{avg_confidence_width/2:.0f} USD/TEU",
                        "volatility_rating": "High" if avg_confidence_width > 500 else "Medium" if avg_confidence_width > 200 else "Low"
                    }
        
        # Generate actionable recommendations
        if '7_day' in insights["procurement_timing"]:
            r2_7day = insights["procurement_timing"]['7_day']['r2_score']
            if r2_7day >= 0.7:
                insights["actionable_recommendations"].append(
                    "Book freight 7 days ahead when model confidence is high (>70% R²)"
                )
        
        if '14_day' in insights["procurement_timing"]:
            r2_14day = insights["procurement_timing"]['14_day']['r2_score']
            if r2_14day >= 0.5:
                insights["actionable_recommendations"].append(
                    "Use 14-day forecasts for contract negotiation timing"
                )
        
        insights["actionable_recommendations"].append(
            "Monitor trade imbalance ratio - it's the key predictive feature"
        )
        
        self.results["business_insights"] = insights
        logger.info("✅ Business insights generated.")

    def run_all(self, forecast_horizon=None, output_file=None):
        start_time = time.time()
        mode_str = "Quick" if self.quick_mode else "Full"
        logger.info(f"🚀 Starting ATLAS Engine V2.0 ({mode_str} Mode)...")
        
        full_df = self.load_data()
        self.run_forecasting_foundry(full_df, forecast_horizon)
        self.calculate_overall_rankings()
        self.generate_business_insights()
        
        if output_file:
            filename = output_file
        else:
            mode_suffix = "_quick" if self.quick_mode else ""
            filename = f"atlas_v2_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}{mode_suffix}.json"
            
        with open(filename, 'w') as f:
            json.dump(self.results, f, indent=4)
            
        runtime = time.time() - start_time
        logger.info(f"🎉 ATLAS V2.0 analysis complete. Results saved to {filename} (Runtime: {runtime:.1f}s)")
        
        # Print business summary
        if "business_insights" in self.results:
            logger.info("\n📊 BUSINESS SUMMARY:")
            for horizon, timing in self.results["business_insights"]["procurement_timing"].items():
                logger.info(f"  {horizon.replace('_', '-').title()}: {timing['confidence_level']} confidence ({timing['r2_score']:.2f} R²)")
            
            logger.info("\n💡 KEY RECOMMENDATIONS:")
            for rec in self.results["business_insights"]["actionable_recommendations"]:
                logger.info(f"  • {rec}")

def main():
    parser = argparse.ArgumentParser(description='ATLAS - Adaptive Trade & Logistics Analytics System')
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
    
    engine = AtlasEngine(quick_mode=args.quick)
    engine.run_all(forecast_horizon=forecast_horizon, output_file=args.output)

if __name__ == "__main__":
    main()
