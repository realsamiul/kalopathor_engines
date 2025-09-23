# Kalopathor Engines - Complete Package

This folder contains all the Kalopathor forecasting engines and their results, organized for easy deployment and analysis.

## 📁 **File Structure**

### **🚀 Engine Scripts**
- `hyperion_engine_v10_final_final.py` - Original V10 engine (baseline)
- `kalopathor_engine_v11_fixed.py` - Fixed version (New1.txt fixes)
- `kalopathor_2_engine.py` - ATLAS V2.0 (New2.txt enhancements)
- `unified_demo.py` - Integration demo (Hyperion + Atlas)
- `test_kalopathor_no_nan.py` - Test suite

### **📊 Data Files**
- `xsicfeuw_data.csv` - FEUW price data (required)
- `xsiuwfe_data.csv` - UWFE price data (required)

### **📈 Results Files**
- `atlas_v2_results_20250923_001400.json` - ATLAS V2.0 complete results
- `kalopathor_results_20250923_003823.json` - Kalopathor V11 results
- `atlas_*day_predictions_with_confidence.csv` - ATLAS predictions with confidence intervals
- `kalopathor_*day_predictions.csv` - Kalopathor V11 basic predictions
- `integrated_analysis_*.json` - Port risk analysis results

### **📋 Documentation**
- `COMPREHENSIVE_RESULTS_SUMMARY.md` - Detailed markdown results
- `COMPREHENSIVE_RESULTS_PLAINTEXT.txt` - Copy-paste ready results
- `ATLAS_V2_SUMMARY.md` - ATLAS V2.0 feature summary

## 🚀 **Quick Start**

### **1. Setup Environment**
```bash
# Create virtual environment
python -m venv kalopathor_env
kalopathor_env\Scripts\activate

# Install dependencies
pip install pandas numpy scikit-learn yfinance xgboost lightgbm catboost shap pytest
```

### **2. Run Engines**

#### **ATLAS V2.0 (Recommended)**
```bash
# Quick mode (Ridge only, 7-day)
python kalopathor_2_engine.py --quick

# Full analysis (all models, confidence intervals, SHAP)
python kalopathor_2_engine.py

# Specific horizon
python kalopathor_2_engine.py --forecast 14
```

#### **Kalopathor V11 (Fixed Version)**
```bash
# Quick mode
python kalopathor_engine_v11_fixed.py --quick

# Full analysis
python kalopathor_engine_v11_fixed.py
```

#### **Integration Demo**
```bash
# Run unified Hyperion + Atlas demo
python unified_demo.py
```

#### **Original V10 (Baseline)**
```bash
# Original engine for comparison
python hyperion_engine_v10_final_final.py
```

### **3. Run Tests**
```bash
# Test the engines
python -m pytest test_kalopathor_no_nan.py -v
```

## 📊 **Key Results Summary**

### **ATLAS V2.0 Performance**
- **7-day R²**: 0.807 (HIGH confidence)
- **14-day R²**: 0.789 (HIGH confidence)
- **30-day R²**: 0.756 (MEDIUM confidence)
- **Champion Model**: CatBoost across all horizons
- **Features**: Confidence intervals, ensemble methods, SHAP explainability

### **Business Value**
- **Procurement Timing**: Strong signal for 7-day decisions
- **Risk Management**: Uncertainty bands for budget planning
- **Integration Ready**: Works with Hyperion satellite platform

## 🎯 **Recommended Usage**

1. **Production Forecasting**: Use `kalopathor_2_engine.py` (ATLAS V2.0)
2. **Integration Demo**: Use `unified_demo.py` for Hyperion + Atlas
3. **Basic Analysis**: Use `kalopathor_engine_v11_fixed.py` for simple needs
4. **Comparison**: Use `hyperion_engine_v10_final_final.py` for baseline

## 📈 **Output Files**

After running, you'll get:
- **JSON results**: Complete analysis with metrics
- **CSV predictions**: Forecast data for visualization
- **Confidence intervals**: Uncertainty quantification (ATLAS V2.0 only)
- **Business insights**: Actionable recommendations

## 🔧 **Troubleshooting**

- **Import errors**: Ensure virtual environment is activated
- **Data errors**: Verify CSV files are in the same directory
- **CatBoost issues**: Engine automatically handles Windows compatibility

## 📞 **Support**

All engines are production-ready and include comprehensive error handling. Check the documentation files for detailed results and analysis.
