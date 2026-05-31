"""
core/prediction_agent.py

Two-layer prediction system.

Layer 1: Random Forest ML model (always runs first)
Layer 2: LLM reasoning agent (activates when ML confidence < 50%
         or predicted category doesn't match symptoms)

WHY TWO LAYERS:
ML is fast and consistent for clear cases.
LLM handles edge cases where ML training data
doesn't cover the disease (e.g. Gastroenteritis).
Together they cover more ground than either alone.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from core.predictor import predict_disease
from llm.openai_client import call_llm
import json

# -----------------------------------------------------------------------
# Symptom to category mapping
# Used to detect when ML prediction category doesn't match symptoms
# Example: if patient has GI symptoms but ML predicts Respiratory
# that mismatch triggers Layer 2
# -----------------------------------------------------------------------
SYMPTOM_CATEGORIES = {
    "respiratory": [
        "sym_cough", "sym_productive_cough", "sym_shortness_of_breath",
        "sym_wheezing", "sym_chest_tightness", "sym_nasal_congestion",
        "sym_sore_throat"
    ],
    "gastrointestinal": [
        "sym_nausea", "sym_vomiting", "sym_abdominal_pain",
        "sym_diarrhoea", "sym_constipation", "sym_loss_of_appetite",
        "sym_jaundice"
    ],
    "cardiac": [
        "sym_chest_pain", "sym_palpitations", "sym_swollen_ankles",
        "sym_cold_extremities", "sym_fainting"
    ],
    "neurological": [
        "sym_headache", "sym_dizziness", "sym_confusion",
        "sym_blurred_vision", "sym_numbness_tingling",
        "sym_light_sensitivity"
    ],
    "metabolic": [
        "sym_excessive_thirst", "sym_frequent_urination",
        "sym_unexplained_weight_loss", "sym_excessive_sweating",
        "sym_fatigue", "sym_weakness"
    ],
    "systemic": [
        "sym_fever", "sym_chills", "sym_night_sweats",
        "sym_joint_pain", "sym_muscle_aches", "sym_back_pain"
    ]
}

DISEASE_CATEGORY_MAP = {
    "Bronchitis": "respiratory", "Pneumonia": "respiratory",
    "Asthma": "respiratory", "COPD": "respiratory",
    "COVID-19": "respiratory", "Tuberculosis": "respiratory",
    "Lung Cancer": "respiratory",
    "Dengue Fever": "systemic", "Typhoid": "systemic",
    "Malaria": "systemic",
    "Type 2 Diabetes": "metabolic", "Hypothyroidism": "metabolic",
    "Anaemia": "metabolic", "Hypertensive Crisis": "cardiac",
    "Arrhythmia": "cardiac", "Heart Failure": "cardiac",
    "Migraine": "neurological", "Anxiety Attack": "neurological"
}

def get_dominant_symptom_category(patient_data: dict) -> str:
    """
    Finds which symptom category dominates the patient's profile.
    Used to detect mismatch with ML prediction.
    
    Example: patient has 4 GI symptoms and 1 respiratory
    -> dominant category is gastrointestinal
    """
    category_scores = {}
    
    for category, symptoms in SYMPTOM_CATEGORIES.items():
        score = sum(patient_data.get(sym, 0) for sym in symptoms)
        category_scores[category] = score
    
    # Return category with highest score
    dominant = max(category_scores, key=category_scores.get)
    dominant_score = category_scores[dominant]
    
    # If no symptoms present return unknown
    if dominant_score == 0:
        return "unknown"
    
    return dominant


def is_category_mismatch(predicted_disease: str, dominant_category: str) -> bool:
    """
    Checks if ML prediction category matches dominant symptom category.
    
    Example: ML predicts Bronchitis (respiratory) but patient
    has mostly GI symptoms -> mismatch -> trigger Layer 2
    """
    predicted_category = DISEASE_CATEGORY_MAP.get(predicted_disease, "unknown")
    
    # These categories should match
    # systemic diseases (fever, chills) can match with any category
    if dominant_category == "systemic":
        return False
    
    if predicted_category == "unknown" or dominant_category == "unknown":
        return False
    
    return predicted_category != dominant_category

def layer2_llm_reasoning(patient_data: dict, ml_results: dict) -> dict:
    """
    Layer 2: LLM clinical reasoning agent.
    Activates when ML confidence is low or category mismatch detected.
    
    Receives ML results as context so LLM knows what ML already tried.
    Returns refined prediction with clinical reasoning.
    """
    # Build symptom summary
    present_symptoms = [
        key.replace("sym_", "").replace("_", " ")
        for key, val in patient_data.items()
        if key.startswith("sym_") and val == 1
    ]
    symptoms_str = ", ".join(present_symptoms) if present_symptoms else "none reported"

    # Build ML context
    ml_top5_str = "\n".join([
        f"  - {disease}: {round(prob*100, 1)}%"
        for disease, prob in ml_results["top_5"]
    ])

    prompt = f"""You are a clinical reasoning assistant helping with disease prediction.

PATIENT PROFILE:
- Age: {patient_data.get('age', 'unknown')}
- Gender: {patient_data.get('gender', 'unknown')}
- Temperature: {patient_data.get('temperature_c', 37)}°C
- Heart Rate: {patient_data.get('heart_rate_bpm', 75)} bpm
- SpO2: {patient_data.get('spo2_percent', 98)}%
- Blood Pressure: {patient_data.get('systolic_bp', 120)}/{patient_data.get('diastolic_bp', 80)} mmHg
- Symptom Duration: {patient_data.get('symptom_duration_days', 1)} days
- Symptoms present: {symptoms_str}

ML MODEL PREDICTIONS (for context):
{ml_top5_str}
Note: ML model confidence is low or category mismatch detected.

YOUR TASK:
Based on the symptom profile, provide your clinical reasoning.
Consider diseases the ML model may have missed.

Return ONLY a JSON object with exactly these fields:
{{
    "predicted_disease": "most likely disease name",
    "confidence_pct": 65,
    "reasoning": "2-3 sentence clinical explanation",
    "top_3": [
        {{"disease": "Disease 1", "probability": 0.65}},
        {{"disease": "Disease 2", "probability": 0.20}},
        {{"disease": "Disease 3", "probability": 0.15}}
    ]
}}

Return only the JSON object, no other text."""

    try:
        response = call_llm(prompt)
        # Clean response
        cleaned = response.strip()
        if cleaned.startswith("```"):
            cleaned = cleaned.split("```")[1]
            if cleaned.startswith("json"):
                cleaned = cleaned[4:]
        cleaned = cleaned.strip()
        
        result = json.loads(cleaned)
        result["source"] = "llm_reasoning"
        return result

    except Exception as e:
        print(f"Layer 2 LLM reasoning failed: {str(e)}")
        return None
    

def predict_with_agent(patient_data: dict, model, scaler, label_encoder) -> dict:
    """
    Main entry point for two-layer prediction.
    
    Always runs Layer 1 (ML) first.
    Activates Layer 2 (LLM) only when needed.
    
    Returns unified result dict regardless of which layer was used.
    """
    # -----------------------------------------------------------------------
    # LAYER 1: ML Model
    # -----------------------------------------------------------------------
    ml_result = predict_disease(patient_data, model, scaler, label_encoder)
    
    top_disease = ml_result["top_disease"]
    top_confidence = ml_result["top_confidence"]
    
    # Check if Layer 2 is needed
    dominant_category = get_dominant_symptom_category(patient_data)
    category_mismatch = is_category_mismatch(top_disease, dominant_category)
    low_confidence = top_confidence < 0.50
    
    trigger_layer2 = low_confidence or category_mismatch
    
    print(f"Layer 1: {top_disease} ({round(top_confidence*100, 1)}%)")
    print(f"Dominant symptom category: {dominant_category}")
    print(f"Category mismatch: {category_mismatch}")
    print(f"Low confidence: {low_confidence}")
    print(f"Layer 2 triggered: {trigger_layer2}")
    
    # -----------------------------------------------------------------------
    # LAYER 2: LLM Reasoning (only if needed)
    # -----------------------------------------------------------------------
    if trigger_layer2:
        print("Activating Layer 2 LLM reasoning...")
        llm_result = layer2_llm_reasoning(patient_data, ml_result)
        
        if llm_result:
            # Build combined result
            return {
                "top_disease": llm_result["predicted_disease"],
                "top_confidence": llm_result["confidence_pct"] / 100,
                "top_5": [
                    (item["disease"], item["probability"])
                    for item in llm_result["top_3"]
                ],
                "all_probabilities": ml_result["all_probabilities"],
                "reasoning": llm_result.get("reasoning", ""),
                "prediction_source": "two_layer",
                "ml_top_disease": top_disease,
                "ml_confidence": top_confidence,
                "layer2_triggered": True
            }
    
    # Layer 1 result was confident and correct category
    return {
        "top_disease": top_disease,
        "top_confidence": top_confidence,
        "top_5": ml_result["top_5"],
        "all_probabilities": ml_result["all_probabilities"],
        "reasoning": "",
        "prediction_source": "ml_model",
        "layer2_triggered": False
    }

# -----------------------------------------------------------------------
# Quick test
# -----------------------------------------------------------------------
if __name__ == "__main__":
    from core.model_loader import load_models

    models = load_models()

    # Test with GI symptoms - should trigger Layer 2
    patient_data = {
        'age': 26, 'gender': 'Male',
        'height_cm': 175, 'weight_kg': 72, 'bmi': 23.5,
        'temperature_c': 38.0, 'heart_rate_bpm': 77,
        'systolic_bp': 120, 'diastolic_bp': 80,
        'spo2_percent': 98, 'respiratory_rate': 16,
        'blood_glucose_mgdl': 90, 'symptom_duration_days': 3,
        'sym_fever': 1, 'sym_nausea': 1, 'sym_vomiting': 1,
        'sym_abdominal_pain': 1, 'sym_diarrhoea': 1,
        'sym_cough': 0, 'sym_productive_cough': 0,
        'sym_shortness_of_breath': 0, 'sym_wheezing': 0,
        'sym_chest_tightness': 0, 'sym_chest_pain': 0,
        'sym_nasal_congestion': 0, 'sym_sore_throat': 0,
        'sym_chills': 0, 'sym_fatigue': 0, 'sym_weakness': 0,
        'sym_night_sweats': 0, 'sym_loss_of_appetite': 0,
        'sym_palpitations': 0, 'sym_dizziness': 0,
        'sym_fainting': 0, 'sym_swollen_ankles': 0,
        'sym_cold_extremities': 0, 'sym_constipation': 0,
        'sym_jaundice': 0, 'sym_headache': 0,
        'sym_blurred_vision': 0, 'sym_numbness_tingling': 0,
        'sym_confusion': 0, 'sym_light_sensitivity': 0,
        'sym_excessive_thirst': 0, 'sym_frequent_urination': 0,
        'sym_unexplained_weight_loss': 0, 'sym_excessive_sweating': 0,
        'sym_joint_pain': 0, 'sym_muscle_aches': 0, 'sym_back_pain': 0
    }

    result = predict_with_agent(
        patient_data, models["model"],
        models["scaler"], models["label_encoder"]
    )

    print("\nFINAL RESULT:")
    print(f"Disease: {result['top_disease']}")
    print(f"Confidence: {round(result['top_confidence']*100, 1)}%")
    print(f"Source: {result['prediction_source']}")
    print(f"Layer 2 triggered: {result['layer2_triggered']}")
    if result['reasoning']:
        print(f"Reasoning: {result['reasoning']}")