"""
core/model_loader.py

Loads the trained Random Forest model and preprocessing objects from disk.
"""

import os
import json
import joblib
import streamlit as st

# Path to models directory
# os.path.dirname(__file__) gives us the core/ folder path
# We go one level up (..) to reach the project root, then into models/
# This way the path works on any machine regardless of where the project is saved
MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models")
@st.cache_resource
def load_models():
    """
    Load model, scaler, label encoder, and metadata from the models/ directory.
    
    @st.cache_resource means Streamlit runs this function only once per session.
    Without it, the model reloads on every button click — very slow.
    """
    result = {
        "loaded": False,
        "model": None,
        "scaler": None,
        "label_encoder": None,
        "model_info": {},
        "error": None
    }

    try:
        # Load the Random Forest model
        # random_forest_best.pkl is the deployment pointer file
        # If you train a better model later, just replace this file — app code stays the same
        model_path = os.path.join(MODELS_DIR, "random_forest_best.pkl")
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found at: {model_path}")
        result["model"] = joblib.load(model_path)

        # Load the StandardScaler
        # CRITICAL: must be the same scaler fitted on training data
        # A different scaler would silently produce wrong predictions
        scaler_path = os.path.join(MODELS_DIR, "scaler.pkl")
        if not os.path.exists(scaler_path):
            raise FileNotFoundError(f"Scaler not found at: {scaler_path}")
        result["scaler"] = joblib.load(scaler_path)

        # Load the LabelEncoder
        # Converts numeric predictions (0,1,2...) back to disease names
        le_path = os.path.join(MODELS_DIR, "label_encoder.pkl")
        if not os.path.exists(le_path):
            raise FileNotFoundError(f"Label encoder not found at: {le_path}")
        result["label_encoder"] = joblib.load(le_path)
        # Load model metadata (optional but useful for sidebar display)
        info_path = os.path.join(MODELS_DIR, "model_info.json")
        if os.path.exists(info_path):
            with open(info_path, "r") as f:
                result["model_info"] = json.load(f)

        result["loaded"] = True

    except FileNotFoundError as e:
        result["error"] = str(e)
    except Exception as e:
        result["error"] = f"Unexpected error loading models: {str(e)}"

    return result
