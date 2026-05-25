"""
Test script to verify saved models can be loaded.
Run after training to ensure models are saved correctly.
"""

import joblib
import json
from pathlib import Path
import sys

# Add parent directory to path to import from ml/
sys.path.append(str(Path(__file__).parent.parent))

def test_load_models():
    """Test loading all saved model files."""
    
    print("=" * 60)
    print("TESTING MODEL LOADING")
    print("=" * 60)
    
    # Models directory is one level up
    models_dir = Path(__file__).parent.parent / "models"
    
    # Check if models directory exists
    if not models_dir.exists():
        print(" ERROR: models/ directory not found!")
        print("   Run train.py first to train and save models.")
        return
    
    print("\n1. Testing Model Loading...")
    try:
        model = joblib.load(models_dir / "random_forest_best.pkl")
        print(f"  Model loaded: {type(model).__name__}")
    except Exception as e:
        print(f"  Failed to load model: {e}")
        return
    
    print("\n2. Testing Scaler Loading...")
    try:
        scaler = joblib.load(models_dir / "scaler.pkl")
        print(f" Scaler loaded: {type(scaler).__name__}")
    except Exception as e:
        print(f"  Failed to load scaler: {e}")
        return
    
    print("\n3. Testing Label Encoder Loading...")
    try:
        le = joblib.load(models_dir / "label_encoder.pkl")
        print(f"   Label Encoder loaded: {type(le).__name__}")
        print(f"   Number of classes: {len(le.classes_)}")
        print(f"   Classes: {list(le.classes_[:5])}..." if len(le.classes_) > 5 else f"   ✅ Classes: {list(le.classes_)}")
    except Exception as e:
        print(f"   Failed to load label encoder: {e}")
        return
    
    print("\n4. Testing Model Info...")
    info_path = models_dir / "model_info.json"
    if info_path.exists():
        with open(info_path) as f:
            info = json.load(f)
        print(f"   Model Info loaded:")
        print(f"      - Model: {info.get('model_name')}")
        print(f"      - Version: {info.get('version')}")
        print(f"      - Accuracy: {info.get('accuracy')}%")
        print(f"      - Trained: {info.get('trained_date')}")
    else:
        print("    model_info.json not found")
    
    print("\n5. Testing Model Prediction (Dummy Input)...")
    try:
        import numpy as np
        # Create dummy input (79 features)
        dummy_input = np.random.rand(1, 79)
        dummy_scaled = scaler.transform(dummy_input)
        prediction = model.predict(dummy_scaled)
        disease = le.inverse_transform(prediction)
        
        print(f"  Prediction successful!")
        print(f"      - Predicted disease: {disease[0]}")
    except Exception as e:
        print(f"  Prediction failed: {e}")
        return
    
    print("\n" + "=" * 60)
    print(" ALL TESTS PASSED!")
    print("=" * 60)
    print("\nModels are ready for deployment!")

if __name__ == "__main__":
    test_load_models()