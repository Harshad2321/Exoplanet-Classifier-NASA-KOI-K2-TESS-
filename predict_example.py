"""
NASA Space Apps Challenge 2025 - Exoplanet Classifier
Standardized Prediction System Examples and Testing

This script demonstrates the usage of the standardized prediction system
for both single predictions and batch processing.
"""

import sys
import pandas as pd
import numpy as np
from pathlib import Path

# Add src directory to path for imports
sys.path.append(str(Path(__file__).parent / "src"))

from src.standardized_predict import StandardizedExoplanetPredictor, predict_single, predict_csv

def main():
    print("🌌 NASA Space Apps Challenge 2025")
    print("🚀 Exoplanet Classifier - Standardized Prediction Examples")
    print("=" * 70)
    
    # Example 1: Single Prediction - Valid Input
    print("\n📍 Example 1: Single Prediction - Valid Input")
    print("-" * 50)
    
    valid_input = {
        "orbital_period": 10.5,
        "transit_duration": 2.0,
        "planet_radius": 2.1,
        "stellar_radius": 1.0,
        "stellar_temp": 5800,
        "mission": "tess"
    }
    
    result = predict_single(valid_input)
    print("Input:")
    for key, value in valid_input.items():
        print(f"  {key}: {value}")
    
    print("\nOutput:")
    if "error" not in result:
        print(f"  predicted_label: {result['predicted_label']}")
        print("  probabilities:")
        for class_name, prob in result['probabilities'].items():
            print(f"    {class_name}: {prob}")
    else:
        print(f"  error: {result['error']}")
    
    # Example 2: Single Prediction - Missing Field Error
    print("\n📍 Example 2: Single Prediction - Missing Field Error")
    print("-" * 50)
    
    invalid_input = {
        "orbital_period": 10.5,
        # Missing transit_duration
        "planet_radius": 2.1,
        "stellar_radius": 1.0,
        "stellar_temp": 5800,
        "mission": "tess"
    }
    
    result = predict_single(invalid_input)
    print("Input (missing transit_duration):")
    for key, value in invalid_input.items():
        print(f"  {key}: {value}")
    
    print("\nOutput:")
    print(f"  error: {result.get('error', 'No error')}")
    
    # Example 3: Single Prediction - Invalid Value Error
    print("\n📍 Example 3: Single Prediction - Invalid Value Error")
    print("-" * 50)
    
    invalid_value_input = {
        "orbital_period": -5.0,  # Negative period (invalid)
        "transit_duration": 2.0,
        "planet_radius": 2.1,
        "stellar_radius": 1.0,
        "stellar_temp": 5800,
        "mission": "tess"
    }
    
    result = predict_single(invalid_value_input)
    print("Input (negative orbital_period):")
    for key, value in invalid_value_input.items():
        print(f"  {key}: {value}")
    
    print("\nOutput:")
    print(f"  error: {result.get('error', 'No error')}")
    
    # Example 4: Batch Prediction - Create Sample CSV
    print("\n📍 Example 4: Batch Prediction - Sample CSV")
    print("-" * 50)
    
    # Create sample data for batch prediction
    sample_data = pd.DataFrame([
        {
            "orbital_period": 10.5,
            "transit_duration": 2.0,
            "planet_radius": 2.1,
            "stellar_radius": 1.0,
            "stellar_temp": 5800,
            "mission": "tess"
        },
        {
            "orbital_period": 365.25,
            "transit_duration": 6.0,
            "planet_radius": 1.0,
            "stellar_radius": 1.0,
            "stellar_temp": 5778,
            "mission": "kepler"
        },
        {
            "orbital_period": 1.5,
            "transit_duration": 1.2,
            "planet_radius": 0.8,
            "stellar_radius": 0.9,
            "stellar_temp": 4500,
            "mission": "k2"
        },
        {
            "orbital_period": 88.0,
            "transit_duration": 3.5,
            "planet_radius": 1.8,
            "stellar_radius": 1.2,
            "stellar_temp": 6200,
            "mission": "tess"
        }
    ])
    
    print("Sample input data:")
    print(sample_data.to_string(index=False))
    
    # Process batch prediction
    results_df = predict_csv(sample_data)
    
    print("\nBatch prediction results:")
    if "error" not in results_df.columns:
        # Display results in a clean format
        display_cols = [
            "orbital_period", "transit_duration", "planet_radius", 
            "predicted_label", "prob_CONFIRMED", "prob_CANDIDATE", "prob_FALSE_POSITIVE"
        ]
        print(results_df[display_cols].to_string(index=False))
    else:
        print(f"Error in batch processing: {results_df}")
    
    # Example 5: Advanced Usage - Custom Predictor Instance
    print("\n📍 Example 5: Advanced Usage - Custom Predictor Instance")
    print("-" * 50)
    
    try:
        # Create predictor instance for multiple predictions
        predictor = StandardizedExoplanetPredictor()
        
        # Test multiple predictions with the same instance
        test_cases = [
            {"name": "Hot Jupiter", "orbital_period": 3.2, "transit_duration": 2.5, 
             "planet_radius": 1.2, "stellar_radius": 1.1, "stellar_temp": 6000, "mission": "tess"},
            {"name": "Earth-like", "orbital_period": 365.0, "transit_duration": 12.0, 
             "planet_radius": 1.0, "stellar_radius": 1.0, "stellar_temp": 5778, "mission": "kepler"},
            {"name": "Super-Earth", "orbital_period": 20.0, "transit_duration": 4.0, 
             "planet_radius": 1.5, "stellar_radius": 0.8, "stellar_temp": 4200, "mission": "k2"}
        ]
        
        for i, test_case in enumerate(test_cases, 1):
            case_name = test_case.pop("name")
            result = predictor.predict(test_case)
            
            print(f"\nTest Case {i}: {case_name}")
            if "error" not in result:
                print(f"  Prediction: {result['predicted_label']}")
                print(f"  Confidence: {max(result['probabilities'].values()):.1%}")
            else:
                print(f"  Error: {result['error']}")
    
    except Exception as e:
        print(f"❌ Advanced example failed: {e}")
    
    print("\n" + "=" * 70)
    print("🎯 Examples completed! Your standardized prediction system is ready.")
    print("🏆 Ready for NASA Space Apps Challenge 2025!")

if __name__ == "__main__":
    main()