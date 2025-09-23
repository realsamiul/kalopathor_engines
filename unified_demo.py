#!/usr/bin/env python3
"""
Unified demo showing Hyperion (satellite) + Atlas (freight) working together
Based on New2.txt integration strategy
"""

import sys
import os
import json
from datetime import datetime, timedelta
import logging

# Import our engines
from kalopathor_2_engine import AtlasEngine

# Mock Hyperion for demo purposes (since we don't have the actual satellite engine)
class MockHyperionFloodPredictor:
    """Mock Hyperion flood predictor for demonstration"""
    
    def predict_flood_risk(self, port, days=7):
        """Simulate flood risk prediction based on port location"""
        # Mock data - in reality this would use satellite imagery
        flood_risks = {
            'chittagong': 0.85,  # High flood risk
            'dhaka': 0.45,       # Medium risk
            'singapore': 0.15,   # Low risk
            'rotterdam': 0.25    # Low risk
        }
        
        base_risk = flood_risks.get(port.lower(), 0.30)
        
        # Add some temporal variation
        day_factor = 1.0 + (days - 7) * 0.05  # Risk increases with longer forecast
        return min(base_risk * day_factor, 0.95)
    
    def get_port_info(self, port):
        """Get port information for risk assessment"""
        port_info = {
            'chittagong': {
                'country': 'Bangladesh',
                'region': 'South Asia',
                'monsoon_season': True,
                'flood_prone': True
            },
            'dhaka': {
                'country': 'Bangladesh', 
                'region': 'South Asia',
                'monsoon_season': True,
                'flood_prone': True
            },
            'singapore': {
                'country': 'Singapore',
                'region': 'Southeast Asia', 
                'monsoon_season': False,
                'flood_prone': False
            },
            'rotterdam': {
                'country': 'Netherlands',
                'region': 'Europe',
                'monsoon_season': False,
                'flood_prone': False
            }
        }
        return port_info.get(port.lower(), {})

class IntegratedRiskPlatform:
    """Unified platform combining Hyperion (satellite) + Atlas (freight) intelligence"""
    
    def __init__(self):
        self.flood_predictor = MockHyperionFloodPredictor()
        self.freight_forecaster = AtlasEngine(quick_mode=True)
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
    
    def analyze_supply_chain_risk(self, port, forecast_days=7):
        """Analyze integrated supply chain risk combining flood + freight predictions"""
        
        self.logger.info(f"🔍 Analyzing supply chain risk for {port.upper()}...")
        
        # Step 1: Get flood risk from Hyperion
        flood_risk = self.flood_predictor.predict_flood_risk(port, forecast_days)
        port_info = self.flood_predictor.get_port_info(port)
        
        # Step 2: Get freight forecast from Atlas
        self.logger.info("📊 Running Atlas freight forecast...")
        self.freight_forecaster.run_all(forecast_horizon=forecast_days, output_file=f"atlas_{port}_analysis.json")
        
        # Step 3: Calculate disruption premium
        disruption_premium = self._calculate_disruption_premium(flood_risk, port_info)
        
        # Step 4: Generate integrated recommendations
        recommendations = self._generate_recommendations(flood_risk, disruption_premium, port_info)
        
        return {
            "port": port.upper(),
            "analysis_date": datetime.now().isoformat(),
            "forecast_horizon_days": forecast_days,
            "hyperion_analysis": {
                "flood_risk": flood_risk,
                "port_info": port_info,
                "risk_level": self._categorize_risk(flood_risk)
            },
            "atlas_analysis": {
                "freight_forecast_available": True,
                "disruption_premium_usd_per_teu": disruption_premium
            },
            "integrated_recommendations": recommendations
        }
    
    def _calculate_disruption_premium(self, flood_risk, port_info):
        """Calculate expected freight rate premium due to disruption risk"""
        base_premium = 0
        
        # Flood risk impact
        if flood_risk > 0.8:
            base_premium += 450  # High flood risk
        elif flood_risk > 0.6:
            base_premium += 250  # Medium flood risk
        elif flood_risk > 0.4:
            base_premium += 100  # Low flood risk
        
        # Port-specific factors
        if port_info.get('monsoon_season'):
            base_premium += 150
        
        if port_info.get('flood_prone'):
            base_premium += 200
        
        return base_premium
    
    def _categorize_risk(self, flood_risk):
        """Categorize flood risk level"""
        if flood_risk >= 0.8:
            return "CRITICAL"
        elif flood_risk >= 0.6:
            return "HIGH"
        elif flood_risk >= 0.4:
            return "MEDIUM"
        else:
            return "LOW"
    
    def _generate_recommendations(self, flood_risk, disruption_premium, port_info):
        """Generate actionable recommendations based on integrated analysis"""
        recommendations = []
        
        if flood_risk > 0.8:
            recommendations.append("🚨 IMMEDIATE ACTION: Book alternative routes - flood risk is critical")
            recommendations.append("📋 Consider air freight for time-sensitive cargo")
            recommendations.append("💰 Budget for +$450/TEU disruption premium")
        
        elif flood_risk > 0.6:
            recommendations.append("⚠️  Monitor weather forecasts closely - high flood risk")
            recommendations.append("🔄 Prepare contingency plans for port closure")
            recommendations.append(f"💰 Budget for +${disruption_premium}/TEU disruption premium")
        
        elif flood_risk > 0.4:
            recommendations.append("📊 Standard monitoring - medium flood risk")
            recommendations.append(f"💰 Consider +${disruption_premium}/TEU premium for risk mitigation")
        
        else:
            recommendations.append("✅ Low flood risk - proceed with normal operations")
        
        # Port-specific recommendations
        if port_info.get('monsoon_season'):
            recommendations.append("🌧️  Monsoon season active - expect weather delays")
        
        if port_info.get('flood_prone'):
            recommendations.append("🏗️  Port is flood-prone - monitor water levels")
        
        return recommendations

def demo_integrated_platform():
    """Demo the integrated Hyperion + Atlas platform"""
    
    print("🚀 INTEGRATED RISK PLATFORM DEMO")
    print("=" * 50)
    print("Combining Hyperion (satellite intelligence) + Atlas (freight forecasting)")
    print()
    
    platform = IntegratedRiskPlatform()
    
    # Demo scenarios
    ports_to_analyze = ['chittagong', 'dhaka', 'singapore', 'rotterdam']
    
    for port in ports_to_analyze:
        print(f"\n📍 ANALYZING: {port.upper()}")
        print("-" * 30)
        
        try:
            analysis = platform.analyze_supply_chain_risk(port, forecast_days=7)
            
            # Display results
            hyperion = analysis['hyperion_analysis']
            atlas = analysis['atlas_analysis']
            
            print(f"🌊 Flood Risk: {hyperion['flood_risk']:.1%} ({hyperion['risk_level']})")
            print(f"🚢 Disruption Premium: +${atlas['disruption_premium_usd_per_teu']}/TEU")
            print(f"🌍 Region: {hyperion['port_info'].get('region', 'Unknown')}")
            
            print("\n💡 RECOMMENDATIONS:")
            for rec in analysis['integrated_recommendations']:
                print(f"  {rec}")
            
            # Save detailed analysis
            with open(f"integrated_analysis_{port}.json", 'w') as f:
                json.dump(analysis, f, indent=2)
            
        except Exception as e:
            print(f"❌ Error analyzing {port}: {e}")
    
    print(f"\n🎉 Demo complete! Check individual analysis files for detailed results.")
    print("📊 This demonstrates how Hyperion + Atlas provide comprehensive supply chain intelligence.")

if __name__ == "__main__":
    demo_integrated_platform()
