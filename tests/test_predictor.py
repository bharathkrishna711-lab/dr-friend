import sys
sys.path.insert(0, 'C:\\Projects\\dr-friend')
from core.model_loader import load_models
from core.predictor import predict_disease


models = load_models()

patient_data = {
    'age': 42,
    'gender': 'Male',
    'height_cm': 175,
    'weight_kg': 78,
    'bmi': 25.5,
    'temperature_c': 38.5,
    'heart_rate_bpm': 94,
    'systolic_bp': 124,
    'diastolic_bp': 82,
    'spo2_percent': 95.0,
    'respiratory_rate': 20,
    'blood_glucose_mgdl': 95,
    'symptom_duration_days': 5,
    'sym_cough': 1,
    'sym_fever': 1,
    'sym_shortness_of_breath': 1,
    'sym_fatigue': 1,
    'sym_chest_tightness': 1,
    'sym_productive_cough': 1,
    'sym_chills': 1,
    'sym_weakness': 1,
    'sym_night_sweats': 0,
    'sym_nasal_congestion': 0,
    'sym_sore_throat': 0,
    'sym_wheezing': 1,
}

result = predict_disease(patient_data, models['model'], models['scaler'], models['label_encoder'])
print('Top disease:', result['top_disease'])
print('Confidence:', round(result['top_confidence'] * 100, 1), '%')
print('Top 5:')
for disease, prob in result['top_5']:
    print(f'  {disease}: {round(prob*100, 1)}%')