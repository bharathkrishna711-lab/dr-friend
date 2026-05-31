"""
llm/entity_extractor.py

Extracts structured patient data from conversation text using LLM.

WHY THIS IS THE HARDEST PART OF UC1:
The LLM must return valid JSON with exact field names matching
our predictor.py expectations. If field names are wrong or JSON
is malformed, the prediction pipeline breaks.

Solution: give the LLM an explicit JSON schema in the prompt
and validate the output before passing to predictor.
"""

import json
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from llm.groq_client import call_llm

# -----------------------------------------------------------------------
# Expected output schema
# This is the exact structure predictor.py needs
# All 37 symptoms must be present as 0 or 1
# -----------------------------------------------------------------------
PATIENT_DATA_SCHEMA = {
    "age": "integer",
    "gender": "Male or Female",
    "height_cm": "float",
    "weight_kg": "float",
    "bmi": "float (calculate from height and weight if not given)",
    "temperature_c": "float (default 37.0 if not mentioned)",
    "heart_rate_bpm": "integer (default 75 if not mentioned)",
    "systolic_bp": "integer (default 120 if not mentioned)",
    "diastolic_bp": "integer (default 80 if not mentioned)",
    "spo2_percent": "float (default 98.0 if not mentioned)",
    "respiratory_rate": "integer (default 16 if not mentioned)",
    "blood_glucose_mgdl": "float (default 90 if not mentioned)",
    "symptom_duration_days": "integer (default 1 if not mentioned)",
    "sym_cough": "0 or 1",
    "sym_productive_cough": "0 or 1",
    "sym_shortness_of_breath": "0 or 1",
    "sym_wheezing": "0 or 1",
    "sym_chest_tightness": "0 or 1",
    "sym_chest_pain": "0 or 1",
    "sym_nasal_congestion": "0 or 1",
    "sym_sore_throat": "0 or 1",
    "sym_fever": "0 or 1",
    "sym_chills": "0 or 1",
    "sym_fatigue": "0 or 1",
    "sym_weakness": "0 or 1",
    "sym_night_sweats": "0 or 1",
    "sym_loss_of_appetite": "0 or 1",
    "sym_nausea": "0 or 1",
    "sym_vomiting": "0 or 1",
    "sym_palpitations": "0 or 1",
    "sym_dizziness": "0 or 1",
    "sym_fainting": "0 or 1",
    "sym_swollen_ankles": "0 or 1",
    "sym_cold_extremities": "0 or 1",
    "sym_abdominal_pain": "0 or 1",
    "sym_diarrhoea": "0 or 1",
    "sym_constipation": "0 or 1",
    "sym_jaundice": "0 or 1",
    "sym_headache": "0 or 1",
    "sym_blurred_vision": "0 or 1",
    "sym_numbness_tingling": "0 or 1",
    "sym_confusion": "0 or 1",
    "sym_light_sensitivity": "0 or 1",
    "sym_excessive_thirst": "0 or 1",
    "sym_frequent_urination": "0 or 1",
    "sym_unexplained_weight_loss": "0 or 1",
    "sym_excessive_sweating": "0 or 1",
    "sym_joint_pain": "0 or 1",
    "sym_muscle_aches": "0 or 1",
    "sym_back_pain": "0 or 1"
}

def build_extraction_prompt(conversation: str) -> str:
    """
    Builds the prompt that instructs LLM to extract patient data.
    
    WHY EXPLICIT SCHEMA IN PROMPT:
    LLMs are good at understanding text but without explicit
    instructions they return inconsistent field names.
    Giving the exact schema forces consistent output every time.
    """
    schema_str = json.dumps(PATIENT_DATA_SCHEMA, indent=2)
    
    prompt = f"""You are a medical data extraction assistant.
Extract patient information from the conversation below and return ONLY a JSON object.

STRICT RULES:
1. Return ONLY valid JSON. No explanation, no markdown, no backticks.
2. Use EXACTLY the field names shown in the schema below.
3. All symptom fields must be 0 (absent) or 1 (present).
4. If a vital sign is not mentioned, use the default value shown.
5. If a symptom is not mentioned, set it to 0.
6. Calculate BMI as weight_kg / (height_m * height_m) if not given.

SCHEMA TO FOLLOW:
{schema_str}

CONVERSATION:
{conversation}

Return only the JSON object:"""
    
    return prompt 


def extract_patient_data(conversation: str) -> dict:
    """
    Main function - takes conversation text, returns patient_data dict.
    
    Args:
        conversation (str): full conversation between user and Dr. Friend
        
    Returns:
        dict: patient_data ready to pass to predict_disease()
        
    WHY VALIDATE AFTER EXTRACTION:
    LLMs occasionally return malformed JSON or miss fields.
    Validation catches this before it breaks the prediction pipeline.
    """
    # Build prompt
    prompt = build_extraction_prompt(conversation)
    
    # Call LLM
    raw_response = call_llm(prompt)
    
    # Clean response - remove any markdown backticks if present
    # Some LLMs add ```json ... ``` even when told not to
    cleaned = raw_response.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.split("```")[1]
        if cleaned.startswith("json"):
            cleaned = cleaned[4:]
    cleaned = cleaned.strip()
    
    # Parse JSON
    try:
        patient_data = json.loads(cleaned)
    except json.JSONDecodeError as e:
        raise ValueError(f"LLM returned invalid JSON: {str(e)}\nRaw response: {raw_response}")
    
    # Fill missing symptom fields with 0
    # Safety net in case LLM missed any symptoms
    for field in PATIENT_DATA_SCHEMA.keys():
        if field.startswith("sym_") and field not in patient_data:
            patient_data[field] = 0
    
    return patient_data 


# -----------------------------------------------------------------------
# Quick test - run directly to verify extraction works
# -----------------------------------------------------------------------
if __name__ == "__main__":
    test_conversation = """
    User: I have had a bad cough for 5 days. My chest feels tight 
    and I have been having fever on and off.
    
    Dr. Friend: How old are you and what is your gender?
    
    User: I am 42 years old, male.
    
    Dr. Friend: Can you share your vitals?
    
    User: Temperature is 38.5, BP is 124/82, oxygen is 95, heart rate 94.
    My height is 175cm and weight is 78kg.
    
    Dr. Friend: Any other symptoms?
    
    User: Shortness of breath yes, fatigue yes, some phlegm too.
    """
    
    print("Extracting patient data from conversation...")
    patient_data = extract_patient_data(test_conversation)
    print("\nExtracted patient data:")
    print(json.dumps(patient_data, indent=2))