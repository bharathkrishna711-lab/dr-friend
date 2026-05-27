# Urgency level labels - shown directly to user as actionable instructions
URGENCY_LEVELS = {
    0: "Self-Care at Home",
    1: "See a Doctor Soon",
    2: "See a Doctor Today",
    3: "Go to Emergency"
}

# Urgency level descriptions shown below the label
URGENCY_DESCRIPTIONS = {
    0: "Your vitals look stable. Rest, stay hydrated, and monitor your symptoms.",
    1: "Please see a doctor within the next 48 hours. Monitor symptoms closely.",
    2: "Please see a doctor today. Don't delay — your symptoms need attention.",
    3: "Go to emergency immediately or call an ambulance. Do not wait."
}

# -----------------------------------------------------------------------
# Vital sign thresholds
# Based on NEWS2 (NHS) and WHO/AHA clinical guidelines
# -----------------------------------------------------------------------

# SpO2 thresholds (WHO)
SPO2_EMERGENCY = 93.0     # Go to emergency immediately
SPO2_CONCERN = 95.0       # Needs attention today

# Temperature thresholds (WHO)
TEMP_FEVER = 38.5         # Fever
TEMP_HIGH_FEVER = 39.5    # High fever

# Heart rate thresholds
HR_HIGH = 100             # Tachycardia
HR_VERY_HIGH = 130        # Significantly elevated - emergency
HR_LOW = 60               # Bradycardia

# Blood pressure thresholds (AHA)
BP_SYS_HIGH = 140         # Hypertension
BP_SYS_EMERGENCY = 180    # Hypertensive crisis - emergency

# Respiratory rate threshold
RESP_RATE_HIGH = 20       # Tachypnoea

# -----------------------------------------------------------------------
# Disease-based urgency overrides
# -----------------------------------------------------------------------
EMERGENCY_DISEASES = [
    "Heart Failure", "Arrhythmia"
]
HIGH_RISK_DISEASES = [
    "Lung Cancer", "Tuberculosis", "Pneumonia"
]
MEDIUM_RISK_DISEASES = [
    "COPD", "COVID-19", "Dengue Fever",
    "Type 2 Diabetes", "Hypertensive Crisis", "Bronchitis"
]

def assess_urgency(vitals: dict, predicted_disease: str) -> dict:
    """
    Evaluate urgency based on vitals, symptoms and predicted disease.

    Args:
        vitals (dict)           : patient data dict (same format as predictor)
        predicted_disease (str) : top predicted disease from predictor.py

    Returns dict with:
        level (str)             : human readable action e.g. "See a Doctor Today"
        description (str)       : explanation text shown below label
        score (int)             : internal numeric score 0-3
        triggered_rules (list)  : flags that raised urgency, shown to user
    """
    triggered_rules = []
    score = 0

    # -----------------------------------------------------------------------
    # RULE 1: SpO2 (oxygen saturation)
    # Most critical vital for respiratory conditions
    # Source: WHO COVID-19 clinical management guidelines
    # -----------------------------------------------------------------------
    spo2 = vitals.get("spo2_percent", 99)
    if spo2 < SPO2_EMERGENCY:
        triggered_rules.append(
            f"SpO2 {spo2}% is critically low (below {SPO2_EMERGENCY}%). "
            "Your oxygen level needs emergency attention."
        )
        score += 3
    elif spo2 <= SPO2_CONCERN:
        triggered_rules.append(
            f"SpO2 {spo2}% is below the safe threshold of {SPO2_CONCERN}%. "
            "Your oxygen level needs monitoring."
        )
        score += 1
    # -----------------------------------------------------------------------
    # RULE 2: Temperature (fever)
    # Source: WHO fever thresholds
    # -----------------------------------------------------------------------
    temperature = vitals.get("temperature_c", 37.0)
    if temperature > TEMP_HIGH_FEVER:
        triggered_rules.append(
            f"Temperature {temperature}C - very high fever "
            f"(above {TEMP_HIGH_FEVER}C). Needs medical attention."
        )
        score += 2
    elif temperature > TEMP_FEVER:
        triggered_rules.append(
            f"Temperature {temperature}C - fever present "
            f"(above {TEMP_FEVER}C)."
        )
        score += 1
    # -----------------------------------------------------------------------
    # RULE 3: Heart rate
    # Source: NEWS2 scoring system (NHS)
    # -----------------------------------------------------------------------
    heart_rate = vitals.get("heart_rate_bpm", 75)
    if heart_rate > HR_VERY_HIGH:
        triggered_rules.append(
            f"Heart rate {heart_rate} bpm - significantly elevated "
            f"(above {HR_VERY_HIGH} bpm). This needs emergency evaluation."
        )
        score += 3
    elif heart_rate > HR_HIGH:
        triggered_rules.append(
            f"Heart rate {heart_rate} bpm - above normal range "
            f"(60-100 bpm)."
        )
        score += 1
    elif heart_rate < HR_LOW:
        triggered_rules.append(
            f"Heart rate {heart_rate} bpm - below normal range "
            f"(60-100 bpm)."
        )
        score += 1
    
    # -----------------------------------------------------------------------
    # RULE 4: Blood pressure
    # Source: AHA hypertension guidelines
    # -----------------------------------------------------------------------
    bp_sys = vitals.get("systolic_bp", 120)
    if bp_sys >= BP_SYS_EMERGENCY:
        triggered_rules.append(
            f"Systolic BP {bp_sys} mmHg - hypertensive crisis "
            f"(above {BP_SYS_EMERGENCY} mmHg). Go to emergency."
        )
        score += 3
    elif bp_sys >= BP_SYS_HIGH:
        triggered_rules.append(
            f"Systolic BP {bp_sys} mmHg - elevated "
            f"(above {BP_SYS_HIGH} mmHg AHA threshold)."
        )
        score += 1

    # -----------------------------------------------------------------------
    # RULE 5: Respiratory rate
    # Source: NEWS2 scoring system (NHS)
    # -----------------------------------------------------------------------
    resp_rate = vitals.get("respiratory_rate", 16)
    if resp_rate > RESP_RATE_HIGH:
        triggered_rules.append(
            f"Respiratory rate {resp_rate} breaths/min - "
            f"above normal (above {RESP_RATE_HIGH})."
        )
        score += 1
    
    # -----------------------------------------------------------------------
    # RULE 6: Dangerous symptom combinations
    # Source: NHS red flag symptoms guidelines
    # -----------------------------------------------------------------------
    has_chest_pain = vitals.get("sym_chest_pain", 0)
    has_sob = vitals.get("sym_shortness_of_breath", 0)
    has_chest_tightness = vitals.get("sym_chest_tightness", 0)
    has_confusion = vitals.get("sym_confusion", 0)
    has_fainting = vitals.get("sym_fainting", 0)
    has_cough = vitals.get("sym_cough", 0)
    has_phlegm = vitals.get("sym_productive_cough", 0)

    if has_chest_pain and has_sob:
        triggered_rules.append(
            "Chest pain combined with shortness of breath - "
            "requires ruling out cardiac or pulmonary causes."
        )
        score += 2

    if has_sob and has_chest_tightness:
        triggered_rules.append(
            "Shortness of breath with chest tightness - "
            "warrants evaluation for pneumonia or cardiac involvement."
        )
        score += 1

    if has_cough and has_phlegm and temperature > TEMP_FEVER:
        triggered_rules.append(
            "Productive cough with fever - "
            "indicates possible bacterial or viral respiratory infection."
        )
        score += 1

    if has_confusion or has_fainting:
        triggered_rules.append(
            "Confusion or fainting reported - "
            "serious signs requiring prompt evaluation."
        )
        score += 2

    # -----------------------------------------------------------------------
    # RULE 7: Disease-based urgency override
    # Even with normal vitals, some diseases need professional confirmation
    # -----------------------------------------------------------------------
    if predicted_disease in EMERGENCY_DISEASES:
        triggered_rules.append(
            f"Predicted condition ({predicted_disease}) requires "
            "immediate medical evaluation."
        )
        score = max(score, 3)

    elif predicted_disease in HIGH_RISK_DISEASES:
        triggered_rules.append(
            f"Predicted condition ({predicted_disease}) warrants "
            "same-day medical evaluation to confirm or rule out."
        )
        score = max(score, 2)

    elif predicted_disease in MEDIUM_RISK_DISEASES:
        triggered_rules.append(
            f"Predicted condition ({predicted_disease}) should "
            "be confirmed by a doctor soon."
        )
        score = max(score, 1)
    
    # -----------------------------------------------------------------------
    # Convert accumulated score to urgency level
    # Score 0   -> Self-Care at Home
    # Score 1   -> See a Doctor Soon
    # Score 2   -> See a Doctor Today
    # Score 3+  -> Go to Emergency
    # -----------------------------------------------------------------------
    score = min(score, 3)  # cap at 3
    level = URGENCY_LEVELS[score]
    description = URGENCY_DESCRIPTIONS[score]

    return {
        "level": level,
        "description": description,
        "score": score,
        "triggered_rules": triggered_rules
    }


# -----------------------------------------------------------------------
# Quick test - run directly to verify urgency engine works
# -----------------------------------------------------------------------
if __name__ == "__main__":
    # Rahul Sharma from sample PDF
    test_vitals = {
        "spo2_percent": 95.0,
        "temperature_c": 38.5,
        "heart_rate_bpm": 94,
        "systolic_bp": 124,
        "diastolic_bp": 82,
        "respiratory_rate": 20,
        "sym_cough": 1,
        "sym_productive_cough": 1,
        "sym_shortness_of_breath": 1,
        "sym_chest_tightness": 1,
        "sym_chest_pain": 0,
        "sym_confusion": 0,
        "sym_fainting": 0
    }

    result = assess_urgency(test_vitals, "Bronchitis")
    print("Urgency Level:", result["level"])
    print("Description:", result["description"])
    print("Score:", result["score"])
    print("\nTriggered Rules:")
    for rule in result["triggered_rules"]:
        print(f"  - {rule}")