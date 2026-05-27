"""
core/predictor.py

Converts raw patient data (vitals + symptoms) into a prediction.

CRITICAL RULE: every transformation here must exactly match ml/preprocess.py.
A mismatch = garbage predictions even with a perfect model.
"""

import numpy as np
import pandas as pd
# Symptom columns in the exact order used during training
# This must match the column order in dr_friend_dataset.csv
# Exact feature names from model.feature_names_in_
SYMPTOM_COLUMNS = [
    "sym_cough", "sym_productive_cough", "sym_shortness_of_breath", "sym_wheezing",
    "sym_chest_tightness", "sym_chest_pain", "sym_nasal_congestion", "sym_sore_throat",
    "sym_fever", "sym_chills", "sym_fatigue", "sym_weakness", "sym_night_sweats",
    "sym_loss_of_appetite", "sym_nausea", "sym_vomiting", "sym_palpitations",
    "sym_dizziness", "sym_fainting", "sym_swollen_ankles", "sym_cold_extremities",
    "sym_abdominal_pain", "sym_diarrhoea", "sym_constipation", "sym_jaundice",
    "sym_headache", "sym_blurred_vision", "sym_numbness_tingling", "sym_confusion",
    "sym_light_sensitivity", "sym_excessive_thirst", "sym_frequent_urination",
    "sym_unexplained_weight_loss", "sym_excessive_sweating", "sym_joint_pain",
    "sym_muscle_aches", "sym_back_pain"
]

VITAL_COLUMNS = [
    "age", "height_cm", "weight_kg", "bmi",
    "temperature_c", "heart_rate_bpm", "systolic_bp", "diastolic_bp",
    "pulse_pressure", "spo2_percent", "respiratory_rate",
    "blood_glucose_mgdl", "symptom_duration_days"
]


def engineer_features(patient_data: dict) -> dict:
    """
    Apply the same feature engineering from ml/preprocess.py.
    Must match training exactly — same bins, same thresholds, same names.
    """
    enriched = patient_data.copy()

    # -- age_group --
    # Bins: 0-18=child, 18-45=adult, 45-65=middle_age, 65+=elderly
    # child is the reference category (dropped during training) = all zeros
    age = patient_data.get("age", 30)
    if age <= 18:
        enriched["age_group"] = "child"      # all age_group columns = 0
    elif age <= 45:
        enriched["age_group"] = "adult"
    elif age <= 65:
        enriched["age_group"] = "middle_age"
    else:
        enriched["age_group"] = "elderly"

    # -- fever_category --
    # Model only has 2 fever flags: low_grade and high_grade
    # Normal temperature = both flags are 0
    temp = patient_data.get("temperature_c", 37.0)
    if temp >= 38.5:
        enriched["fever_category"] = "high_grade_fever"
    elif temp >= 37.5:
        enriched["fever_category"] = "low_grade_fever"
    else:
        enriched["fever_category"] = "normal"

    # -- symptom_count --
    enriched["symptom_count"] = sum(
        patient_data.get(sym, 0) for sym in SYMPTOM_COLUMNS
    )

    # -- vital_risk_score --
    risk = 0
    if patient_data.get("temperature_c", 37.0) > 38.5:
        risk += 1
    if patient_data.get("heart_rate_bpm", 75) > 100:
        risk += 1
    if patient_data.get("systolic_bp", 120) > 140:
        risk += 1
    if patient_data.get("spo2_percent", 98) < 95:
        risk += 1
    if patient_data.get("respiratory_rate", 16) > 20:
        risk += 1
    enriched["vital_risk_score"] = risk

    # -- pulse_pressure --
    # Derived from systolic - diastolic if not provided
    if "pulse_pressure" not in patient_data:
        enriched["pulse_pressure"] = (
            patient_data.get("systolic_bp", 120) - 
            patient_data.get("diastolic_bp", 80)
        )

    return enriched

def build_feature_vector(patient_data: dict) -> pd.DataFrame:
    """
    Converts patient data into a single-row DataFrame matching
    the exact 79-column order the model was trained on.
    """
    enriched = engineer_features(patient_data)
    row = {}

    # Raw vitals and demographics
    for col in VITAL_COLUMNS:
        row[col] = enriched.get(col, 0)

    # row_id is meaningless for prediction, set to 0
    row["row_id"] = 0

    # Gender one-hot (1=Male, 0=Female)
    row["gender_Male"] = 1 if enriched.get("gender", "Male") == "Male" else 0

    # Symptom binary flags
    for sym in SYMPTOM_COLUMNS:
        row[sym] = enriched.get(sym, 0)

    # Engineered features
    row["symptom_count"] = enriched["symptom_count"]
    row["vital_risk_score"] = enriched["vital_risk_score"]
    row["pulse_pressure"] = enriched.get("pulse_pressure",
        enriched.get("systolic_bp", 120) - enriched.get("diastolic_bp", 80))

    # broad_category - unknown at inference time, set all to 0
    for cat in ["Infectious", "Metabolic", "Neurological", "Respiratory"]:
        row[f"broad_category_{cat}"] = 0

    # comorbidity - default to none unless user provides
    comorbidity_options = [
        "Anxiety Attack", "Arrhythmia", "Asthma", "Bronchitis", "COPD",
        "COVID-19", "Dengue Fever", "Heart Failure", "Hypertensive Crisis",
        "Hypothyroidism", "Malaria", "Migraine", "Pneumonia",
        "Type 2 Diabetes", "Typhoid", "none"
    ]
    user_comorbidity = enriched.get("comorbidity", "none")
    for option in comorbidity_options:
        row[f"comorbidity_{option}"] = 1 if user_comorbidity == option else 0

    # age_group one-hot (child is reference = all zeros)
    for group in ["adult", "middle_age", "elderly"]:
        row[f"age_group_{group}"] = 1 if enriched["age_group"] == group else 0

    # fever_category one-hot (normal is reference = all zeros)
    for cat in ["low_grade_fever", "high_grade_fever"]:
        row[f"fever_category_{cat}"] = 1 if enriched["fever_category"] == cat else 0

    return pd.DataFrame([row])

def predict_disease(patient_data: dict, model, scaler, label_encoder) -> dict:
    """
    End-to-end prediction pipeline:
        1. Build feature vector from patient data
        2. Align columns to match training order exactly
        3. Scale features using trained scaler
        4. Get probability distribution across all 15 diseases
        5. Return ranked predictions
    """
    # Step 1: build feature vector
    feature_df = build_feature_vector(patient_data)

    # Step 2: align columns to exact training order
    # This is critical — wrong column order = wrong predictions
    # model.feature_names_in_ is the ground truth column order
    if hasattr(model, "feature_names_in_"):
        for col in model.feature_names_in_:
            if col not in feature_df.columns:
                feature_df[col] = 0
        feature_df = feature_df[model.feature_names_in_]

    # Step 3: scale features
    feature_scaled = scaler.transform(feature_df)

    # Step 4: get probability distribution across all 15 diseases
    # predict_proba returns shape (1, 15)
    probabilities = model.predict_proba(feature_scaled)[0]

    # Step 5: decode class labels
    disease_names = label_encoder.classes_
    prob_dict = {
        disease: float(prob)
        for disease, prob in zip(disease_names, probabilities)
    }

    # Sort by probability descending
    sorted_predictions = sorted(
        prob_dict.items(), key=lambda x: x[1], reverse=True
    )

    return {
        "top_disease": sorted_predictions[0][0],
        "top_confidence": sorted_predictions[0][1],
        "top_5": sorted_predictions[:5],
        "all_probabilities": prob_dict
    }