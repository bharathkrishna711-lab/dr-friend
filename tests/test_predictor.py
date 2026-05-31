import sys
sys.path.insert(0, 'C:\\Projects\\dr-friend')
from core.model_loader import load_models
from core.predictor import predict_disease
from core.urgency_engine import assess_urgency

models = load_models()

patient_data = {
    'age': 55,
    'gender': 'Male',
    'height_cm': 170,
    'weight_kg': 85,
    'bmi': 29.4,
    'temperature_c': 37.0,
    'heart_rate_bpm': 80,
    'systolic_bp': 135,
    'diastolic_bp': 85,
    'spo2_percent': 98.0,
    'respiratory_rate': 16,
    'blood_glucose_mgdl': 220,
    'symptom_duration_days': 30,
    'sym_excessive_thirst': 1,
    'sym_frequent_urination': 1,
    'sym_fatigue': 1,
    'sym_blurred_vision': 1,
    'sym_unexplained_weight_loss': 1
}

result = predict_disease(patient_data, models['model'], models['scaler'], models['label_encoder'])
print('Top disease:', result['top_disease'])
print('Confidence:', round(result['top_confidence'] * 100, 1), '%')
print('Top 5:')
for disease, prob in result['top_5']:
    print(f'  {disease}: {round(prob*100, 1)}%')

urgency = assess_urgency(patient_data, result['top_disease'])
print("\nUrgency:", urgency['level'])
print("Score:", urgency['score'])
print("Rules:", urgency['triggered_rules'])