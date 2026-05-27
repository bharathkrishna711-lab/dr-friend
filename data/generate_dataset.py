"""
Dr. Friend — Synthetic Healthcare Dataset Generator
=====================================================
Generates a clinically-informed synthetic dataset with:
- 15 diseases across 5 broad categories
- 13 vitals (home-measurable)
- 37 binary symptom flags + symptom duration
- Real-world noise: missing values, measurement errors,
  label noise, comorbidities, age-vital correlations,
  class imbalance
"""

import numpy as np
import pandas as pd 
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings("ignore")

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

# ─────────────────────────────────────────────
# 1. CONSTANTS & TAXONOMY
# ─────────────────────────────────────────────

BROAD_CATEGORIES = {
    "Respiratory":      ["Bronchitis", "Pneumonia", "Asthma", "COPD", "COVID-19"],
    "Cardiac":          ["Hypertensive Crisis", "Arrhythmia", "Heart Failure"],
    "Metabolic":        ["Type 2 Diabetes", "Hypothyroidism", "Anaemia"],
    "Infectious":       ["Dengue Fever", "Typhoid", "Malaria"],
    "Neurological":     ["Migraine", "Anxiety Attack"],
}

DISEASE_TO_CATEGORY = {
    d: cat for cat, diseases in BROAD_CATEGORIES.items() for d in diseases
}

ALL_DISEASES = list(DISEASE_TO_CATEGORY.keys())  # 15 diseases

# Realistic class distribution — imbalanced like real population
# Common conditions more frequent than rare ones
DISEASE_WEIGHTS = {
    "Bronchitis":          0.10,
    "Asthma":              0.09,
    "Migraine":            0.09,
    "Anxiety Attack":      0.08,
    "Type 2 Diabetes":     0.08,
    "COVID-19":            0.08,
    "Pneumonia":           0.07,
    "Typhoid":             0.07,
    "Dengue Fever":        0.07,
    "Malaria":             0.06,
    "Anaemia":             0.06,
    "Hypothyroidism":      0.05,
    "Arrhythmia":          0.04,
    "Heart Failure":       0.03,
    "Hypertensive Crisis": 0.03,
}

SYMPTOMS = [
    # Respiratory (8)
    "cough", "productive_cough", "shortness_of_breath", "wheezing",
    "chest_tightness", "chest_pain", "nasal_congestion", "sore_throat",
    # Systemic/Infectious (8)
    "fever", "chills", "fatigue", "weakness",
    "night_sweats", "loss_of_appetite", "nausea", "vomiting",
    # Cardiac/Circulatory (5)
    "palpitations", "dizziness", "fainting", "swollen_ankles", "cold_extremities",
    # Gastrointestinal (4)
    "abdominal_pain", "diarrhoea", "constipation", "jaundice",
    # Neurological (5)
    "headache", "blurred_vision", "numbness_tingling", "confusion", "light_sensitivity",
    # Metabolic (4)
    "excessive_thirst", "frequent_urination", "unexplained_weight_loss", "excessive_sweating",
    # Musculoskeletal (3)
    "joint_pain", "muscle_aches", "back_pain",
]  # 37 symptoms total


# ─────────────────────────────────────────────
# 2. CLINICAL PROFILES PER DISEASE
# Each symptom listed as (symptom_name, probability)
# Primary symptoms: 0.60-0.90
# Secondary symptoms: 0.25-0.50
# Background noise applied globally on top
# ─────────────────────────────────────────────

@dataclass
class DiseaseProfile:
    name: str
    # Vital ranges: (mean, std, min_clip, max_clip)
    temp_range:       Tuple = (37.0, 0.3, 35.5, 38.0)
    hr_range:         Tuple = (75,   10,  50,   110)
    sbp_range:        Tuple = (120,  12,  90,   160)
    dbp_range:        Tuple = (80,   8,   60,   100)
    spo2_range:       Tuple = (98,   1.0, 94,   100)
    rr_range:         Tuple = (16,   2,   12,   22)
    glucose_range:    Tuple = (95,   12,  70,   130)
    # Age distribution: (mean, std, min, max)
    age_range:        Tuple = (35,   15,  18,   80)
    # Symptom probabilities dict
    symptom_probs:    Dict  = field(default_factory=dict)
    # Duration of symptoms in days: (mean, std, min, max)
    duration_range:   Tuple = (5,    3,   1,    30)
    # Gender bias: probability of being Male (0.5 = no bias)
    male_prob:        float = 0.5


DISEASE_PROFILES: Dict[str, DiseaseProfile] = {

    "Bronchitis": DiseaseProfile(
        name="Bronchitis",
        temp_range=(37.8, 0.5, 37.0, 39.5),
        hr_range=(85, 10, 65, 115),
        sbp_range=(118, 12, 95, 150),
        dbp_range=(76, 8, 58, 96),
        spo2_range=(96, 1.2, 92, 99),
        rr_range=(18, 2, 14, 25),
        glucose_range=(92, 10, 70, 120),
        age_range=(35, 15, 18, 70),
        duration_range=(7, 4, 2, 21),
        symptom_probs={
            "cough": 0.92, "productive_cough": 0.80, "chest_tightness": 0.65,
            "fatigue": 0.70, "fever": 0.55, "sore_throat": 0.50,
            "shortness_of_breath": 0.40, "nasal_congestion": 0.35,
            "wheezing": 0.30, "loss_of_appetite": 0.30, "muscle_aches": 0.25,
            "headache": 0.20, "chills": 0.20, "nausea": 0.15,
        }
    ),

    "Pneumonia": DiseaseProfile(
        name="Pneumonia",
        temp_range=(39.0, 0.7, 38.0, 41.0),
        hr_range=(100, 12, 75, 135),
        sbp_range=(112, 14, 88, 145),
        dbp_range=(72, 9, 55, 92),
        spo2_range=(93, 2.0, 85, 97),
        rr_range=(24, 4, 18, 35),
        glucose_range=(100, 15, 75, 140),
        age_range=(45, 20, 18, 85),
        duration_range=(6, 3, 2, 14),
        symptom_probs={
            "fever": 0.90, "cough": 0.88, "shortness_of_breath": 0.82,
            "productive_cough": 0.70, "chest_pain": 0.65, "fatigue": 0.80,
            "chills": 0.65, "loss_of_appetite": 0.60, "weakness": 0.65,
            "nausea": 0.35, "confusion": 0.25, "muscle_aches": 0.40,
            "night_sweats": 0.30, "vomiting": 0.20,
        }
    ),

    "Asthma": DiseaseProfile(
        name="Asthma",
        temp_range=(36.8, 0.3, 36.0, 37.8),
        hr_range=(90, 12, 68, 125),
        sbp_range=(118, 11, 95, 148),
        dbp_range=(76, 8, 58, 96),
        spo2_range=(94, 2.0, 88, 98),
        rr_range=(22, 4, 16, 32),
        glucose_range=(90, 10, 70, 115),
        age_range=(28, 14, 12, 65),
        duration_range=(3, 2, 1, 14),
        symptom_probs={
            "wheezing": 0.90, "shortness_of_breath": 0.88, "chest_tightness": 0.82,
            "cough": 0.78, "fatigue": 0.55, "palpitations": 0.35,
            "nasal_congestion": 0.40, "excessive_sweating": 0.30,
            "anxiety_like": 0.25, "nausea": 0.15,
        }
    ),

    "COPD": DiseaseProfile(
        name="COPD",
        temp_range=(37.0, 0.4, 36.2, 38.5),
        hr_range=(88, 10, 68, 115),
        sbp_range=(130, 14, 100, 165),
        dbp_range=(82, 9, 62, 102),
        spo2_range=(92, 2.5, 82, 96),
        rr_range=(22, 4, 16, 32),
        glucose_range=(95, 12, 72, 130),
        age_range=(62, 10, 45, 85),
        duration_range=(20, 8, 7, 60),
        male_prob=0.60,
        symptom_probs={
            "shortness_of_breath": 0.92, "cough": 0.88, "productive_cough": 0.75,
            "wheezing": 0.70, "fatigue": 0.82, "chest_tightness": 0.65,
            "weakness": 0.70, "loss_of_appetite": 0.50, "cold_extremities": 0.35,
            "swollen_ankles": 0.30, "confusion": 0.20, "night_sweats": 0.20,
        }
    ),

    "COVID-19": DiseaseProfile(
        name="COVID-19",
        temp_range=(38.5, 0.7, 37.5, 40.5),
        hr_range=(92, 12, 68, 125),
        sbp_range=(115, 13, 90, 148),
        dbp_range=(74, 9, 55, 94),
        spo2_range=(94, 2.5, 84, 99),
        rr_range=(20, 4, 14, 30),
        glucose_range=(100, 18, 72, 145),
        age_range=(40, 18, 18, 85),
        duration_range=(8, 4, 3, 21),
        symptom_probs={
            "fever": 0.88, "fatigue": 0.88, "cough": 0.78,
            "shortness_of_breath": 0.65, "loss_of_appetite": 0.65,
            "muscle_aches": 0.65, "headache": 0.60, "sore_throat": 0.50,
            "nasal_congestion": 0.45, "nausea": 0.35, "diarrhoea": 0.30,
            "chills": 0.40, "weakness": 0.70, "chest_tightness": 0.40,
            "confusion": 0.20, "vomiting": 0.20,
        }
    ),

    "Hypertensive Crisis": DiseaseProfile(
        name="Hypertensive Crisis",
        temp_range=(37.0, 0.3, 36.5, 37.8),
        hr_range=(95, 15, 68, 135),
        sbp_range=(185, 15, 160, 230),   # defining feature
        dbp_range=(118, 12, 100, 145),
        spo2_range=(97, 1.0, 93, 99),
        rr_range=(18, 3, 13, 26),
        glucose_range=(105, 20, 75, 160),
        age_range=(55, 12, 35, 80),
        duration_range=(1, 1, 1, 5),
        male_prob=0.55,
        symptom_probs={
            "headache": 0.90, "blurred_vision": 0.72, "dizziness": 0.70,
            "nausea": 0.55, "chest_pain": 0.55, "vomiting": 0.40,
            "confusion": 0.40, "shortness_of_breath": 0.45,
            "palpitations": 0.40, "fainting": 0.25, "numbness_tingling": 0.30,
        }
    ),

    "Arrhythmia": DiseaseProfile(
        name="Arrhythmia",
        temp_range=(36.8, 0.3, 36.2, 37.5),
        hr_range=(108, 25, 40, 180),     # wide range — both brady and tachy
        sbp_range=(118, 16, 88, 158),
        dbp_range=(76, 10, 55, 98),
        spo2_range=(96, 1.5, 90, 99),
        rr_range=(18, 3, 12, 26),
        glucose_range=(95, 12, 70, 130),
        age_range=(52, 16, 25, 82),
        duration_range=(2, 2, 1, 14),
        symptom_probs={
            "palpitations": 0.90, "dizziness": 0.72, "shortness_of_breath": 0.60,
            "chest_pain": 0.55, "fainting": 0.40, "fatigue": 0.65,
            "weakness": 0.50, "anxiety_like": 0.35, "chest_tightness": 0.40,
            "blurred_vision": 0.20, "cold_extremities": 0.30,
        }
    ),

    "Heart Failure": DiseaseProfile(
        name="Heart Failure",
        temp_range=(36.8, 0.4, 36.0, 37.8),
        hr_range=(98, 14, 68, 130),
        sbp_range=(128, 18, 88, 170),
        dbp_range=(82, 11, 58, 105),
        spo2_range=(92, 2.5, 82, 96),
        rr_range=(22, 4, 16, 32),
        glucose_range=(108, 20, 75, 160),
        age_range=(65, 12, 45, 88),
        duration_range=(14, 7, 5, 45),
        male_prob=0.55,
        symptom_probs={
            "shortness_of_breath": 0.92, "fatigue": 0.88, "swollen_ankles": 0.82,
            "weakness": 0.75, "loss_of_appetite": 0.65, "cough": 0.55,
            "dizziness": 0.50, "cold_extremities": 0.55, "chest_pain": 0.45,
            "nausea": 0.40, "confusion": 0.35, "back_pain": 0.30,
            "palpitations": 0.45, "night_sweats": 0.25,
        }
    ),

    "Type 2 Diabetes": DiseaseProfile(
        name="Type 2 Diabetes",
        temp_range=(36.9, 0.3, 36.2, 37.8),
        hr_range=(80, 10, 60, 105),
        sbp_range=(135, 15, 105, 175),
        dbp_range=(86, 10, 65, 108),
        spo2_range=(97, 1.0, 93, 99),
        rr_range=(16, 2, 12, 20),
        glucose_range=(195, 45, 130, 380),   # defining feature
        age_range=(50, 14, 30, 80),
        duration_range=(25, 10, 7, 60),
        symptom_probs={
            "excessive_thirst": 0.85, "frequent_urination": 0.82,
            "fatigue": 0.78, "blurred_vision": 0.55, "weakness": 0.65,
            "unexplained_weight_loss": 0.50, "excessive_sweating": 0.40,
            "numbness_tingling": 0.45, "loss_of_appetite": 0.30,
            "nausea": 0.25, "headache": 0.30, "dizziness": 0.30,
            "cold_extremities": 0.35, "back_pain": 0.25,
        }
    ),

    "Hypothyroidism": DiseaseProfile(
        name="Hypothyroidism",
        temp_range=(36.3, 0.3, 35.5, 37.0),   # low temp
        hr_range=(58, 8, 42, 72),               # bradycardia
        sbp_range=(118, 12, 92, 148),
        dbp_range=(78, 8, 58, 96),
        spo2_range=(97, 1.0, 94, 99),
        rr_range=(14, 2, 10, 18),
        glucose_range=(92, 12, 68, 120),
        age_range=(42, 14, 22, 75),
        duration_range=(30, 10, 10, 90),
        male_prob=0.25,                         # predominantly female
        symptom_probs={
            "fatigue": 0.88, "weakness": 0.80, "constipation": 0.68,
            "unexplained_weight_loss": 0.55, "cold_extremities": 0.65,
            "muscle_aches": 0.55, "headache": 0.45, "back_pain": 0.40,
            "loss_of_appetite": 0.40, "dizziness": 0.35,
            "swollen_ankles": 0.30, "excessive_sweating": 0.15,
            "confusion": 0.20, "joint_pain": 0.30,
        }
    ),

    "Anaemia": DiseaseProfile(
        name="Anaemia",
        temp_range=(36.8, 0.3, 36.0, 37.5),
        hr_range=(95, 12, 72, 125),             # compensatory tachycardia
        sbp_range=(108, 12, 85, 135),
        dbp_range=(70, 8, 52, 88),
        spo2_range=(96, 1.5, 91, 99),
        rr_range=(18, 3, 13, 24),
        glucose_range=(88, 10, 65, 110),
        age_range=(32, 16, 15, 72),
        duration_range=(20, 10, 7, 60),
        male_prob=0.35,                         # more common in women
        symptom_probs={
            "fatigue": 0.90, "weakness": 0.85, "dizziness": 0.72,
            "shortness_of_breath": 0.60, "cold_extremities": 0.65,
            "headache": 0.55, "palpitations": 0.50, "loss_of_appetite": 0.45,
            "fainting": 0.30, "back_pain": 0.25, "numbness_tingling": 0.30,
            "chest_pain": 0.20, "nausea": 0.25,
        }
    ),

    "Dengue Fever": DiseaseProfile(
        name="Dengue Fever",
        temp_range=(39.5, 0.6, 38.5, 41.0),
        hr_range=(88, 12, 65, 115),
        sbp_range=(108, 14, 82, 138),
        dbp_range=(68, 9, 50, 88),
        spo2_range=(97, 1.0, 93, 99),
        rr_range=(18, 3, 13, 24),
        glucose_range=(88, 12, 65, 115),
        age_range=(28, 14, 5, 65),
        duration_range=(5, 2, 3, 10),
        symptom_probs={
            "fever": 0.95, "joint_pain": 0.88, "muscle_aches": 0.85,
            "headache": 0.82, "fatigue": 0.80, "nausea": 0.65,
            "vomiting": 0.50, "loss_of_appetite": 0.65, "rash": 0.55,
            "chills": 0.60, "abdominal_pain": 0.45, "weakness": 0.70,
            "dizziness": 0.40, "back_pain": 0.50,
        }
    ),

    "Typhoid": DiseaseProfile(
        name="Typhoid",
        temp_range=(39.2, 0.6, 38.0, 40.5),
        hr_range=(78, 10, 58, 100),             # relative bradycardia
        sbp_range=(110, 12, 85, 138),
        dbp_range=(70, 8, 52, 88),
        spo2_range=(97, 1.0, 93, 99),
        rr_range=(18, 2, 14, 23),
        glucose_range=(88, 10, 65, 112),
        age_range=(25, 12, 5, 55),
        duration_range=(10, 4, 5, 21),
        symptom_probs={
            "fever": 0.95, "loss_of_appetite": 0.82, "weakness": 0.80,
            "abdominal_pain": 0.72, "headache": 0.70, "fatigue": 0.78,
            "constipation": 0.55, "nausea": 0.55, "diarrhoea": 0.45,
            "chills": 0.50, "vomiting": 0.40, "muscle_aches": 0.40,
            "joint_pain": 0.35, "dizziness": 0.30,
        }
    ),

    "Malaria": DiseaseProfile(
        name="Malaria",
        temp_range=(39.8, 0.8, 38.0, 41.5),    # high spiking fever
        hr_range=(95, 14, 68, 128),
        sbp_range=(105, 14, 80, 135),
        dbp_range=(66, 9, 48, 86),
        spo2_range=(96, 1.5, 90, 99),
        rr_range=(20, 3, 14, 28),
        glucose_range=(82, 14, 55, 115),        # hypoglycaemia risk
        age_range=(28, 16, 5, 65),
        duration_range=(4, 2, 2, 10),
        symptom_probs={
            "fever": 0.95, "chills": 0.90, "excessive_sweating": 0.82,
            "muscle_aches": 0.78, "headache": 0.80, "fatigue": 0.82,
            "nausea": 0.65, "vomiting": 0.55, "joint_pain": 0.55,
            "loss_of_appetite": 0.60, "weakness": 0.72, "dizziness": 0.45,
            "abdominal_pain": 0.40, "confusion": 0.25, "jaundice": 0.20,
        }
    ),

    "Migraine": DiseaseProfile(
        name="Migraine",
        temp_range=(36.8, 0.3, 36.2, 37.5),
        hr_range=(72, 10, 52, 98),
        sbp_range=(122, 14, 95, 155),
        dbp_range=(78, 9, 58, 98),
        spo2_range=(98, 0.8, 95, 100),
        rr_range=(15, 2, 11, 20),
        glucose_range=(88, 10, 68, 112),
        age_range=(32, 12, 15, 60),
        duration_range=(2, 1, 1, 5),
        male_prob=0.35,                         # more common in women
        symptom_probs={
            "headache": 0.98, "nausea": 0.72, "light_sensitivity": 0.78,
            "blurred_vision": 0.55, "vomiting": 0.45, "dizziness": 0.50,
            "fatigue": 0.60, "weakness": 0.40, "loss_of_appetite": 0.45,
            "numbness_tingling": 0.30, "confusion": 0.20,
        }
    ),

    "Anxiety Attack": DiseaseProfile(
        name="Anxiety Attack",
        temp_range=(37.0, 0.3, 36.4, 37.8),
        hr_range=(105, 18, 78, 148),            # tachycardia
        sbp_range=(132, 16, 105, 168),
        dbp_range=(85, 10, 65, 108),
        spo2_range=(98, 1.0, 95, 100),          # normal SpO2 — key differentiator from asthma
        rr_range=(22, 4, 16, 32),
        glucose_range=(92, 12, 70, 120),
        age_range=(30, 12, 15, 60),
        duration_range=(1, 1, 1, 3),
        symptom_probs={
            "palpitations": 0.88, "shortness_of_breath": 0.82,
            "chest_tightness": 0.78, "dizziness": 0.72, "excessive_sweating": 0.68,
            "numbness_tingling": 0.55, "headache": 0.50, "nausea": 0.45,
            "fainting": 0.25, "confusion": 0.30, "muscle_aches": 0.25,
            "weakness": 0.40, "fatigue": 0.55,
        }
    ),
}


# ─────────────────────────────────────────────
# 3. HELPER FUNCTIONS
# ─────────────────────────────────────────────

def sample_vital(mean, std, vmin, vmax, age=None, age_effect=None):
    """Sample a vital with Gaussian noise, clipped to physiological range."""
    val = np.random.normal(mean, std)
    if age is not None and age_effect is not None:
        val += age_effect * (age - 40) / 40   # age scaling
    return float(np.clip(val, vmin, vmax))


def add_measurement_error(val, prob=0.05, error_scale=0.15):
    """Simulate device measurement errors with given probability."""
    if np.random.random() < prob:
        # Either systematic offset or gross error
        if np.random.random() < 0.5:
            val *= (1 + np.random.uniform(-error_scale, error_scale))
        else:
            val *= np.random.choice([0.85, 1.20])  # gross over/under read
    return val


def maybe_missing(val, missing_prob):
    """Return NaN with given probability (device not available)."""
    return np.nan if np.random.random() < missing_prob else val


def generate_symptoms(profile: DiseaseProfile, background_noise=0.08):
    """
    Generate binary symptom flags.
    Primary/secondary symptoms from disease profile.
    Background noise simulates unrelated symptoms.
    """
    symptoms_dict = {}
    for sym in SYMPTOMS:
        base_prob = profile.symptom_probs.get(sym, background_noise)
        # Add per-row random jitter ±0.05
        jittered_prob = np.clip(base_prob + np.random.uniform(-0.05, 0.05), 0, 1)
        symptoms_dict[sym] = int(np.random.random() < jittered_prob)
    return symptoms_dict


def apply_comorbidity(row_dict, secondary_disease, blend=0.4):
    """
    Blend symptom profile of a secondary condition into a row.
    Simulates patients with multiple conditions (10% of dataset).
    """
    sec_profile = DISEASE_PROFILES[secondary_disease]
    for sym, prob in sec_profile.symptom_probs.items():
        if sym in row_dict:
            # Blend: take max of current value and secondary probability
            blended_prob = max(row_dict.get(sym, 0), prob * blend)
            row_dict[sym] = int(np.random.random() < blended_prob)
    return row_dict


def apply_label_noise(label, all_diseases, noise_prob=0.05):
    """
    Flip label to a clinically similar disease with given probability.
    Mimics real-world diagnostic ambiguity.
    """
    if np.random.random() < noise_prob:
        category = DISEASE_TO_CATEGORY[label]
        same_cat = [d for d in BROAD_CATEGORIES[category] if d != label]
        if same_cat:
            return np.random.choice(same_cat)
    return label


# ─────────────────────────────────────────────
# 4. ROW GENERATOR
# ─────────────────────────────────────────────

def generate_row(disease_name: str, row_id: int) -> dict:
    profile = DISEASE_PROFILES[disease_name]

    # --- Demographics ---
    age = int(np.clip(np.random.normal(profile.age_range[0], profile.age_range[1]), profile.age_range[2], profile.age_range[3]))
    gender = "Male" if np.random.random() < profile.male_prob else "Female"

    # Height/Weight with gender and age adjustments
    if gender == "Male":
        height = float(np.clip(np.random.normal(170, 7), 150, 195))
        weight = float(np.clip(np.random.normal(72, 13), 45, 130))
    else:
        height = float(np.clip(np.random.normal(158, 6), 140, 185))
        weight = float(np.clip(np.random.normal(62, 12), 38, 110))

    bmi = round(weight / ((height / 100) ** 2), 1)

    # --- Vitals with age correlation ---
    # Older patients: higher BP, lower SpO2, higher RR
    age_bp_effect   = 0.5    # +0.5 mmHg per year above 40 for BP
    age_spo2_effect = -0.03  # -0.03% per year above 40

    temperature = round(sample_vital(*profile.temp_range), 1)
    heart_rate  = int(sample_vital(*profile.hr_range))
    sbp         = int(sample_vital(*profile.sbp_range, age=age, age_effect=age_bp_effect))
    dbp         = int(sample_vital(*profile.dbp_range, age=age, age_effect=age_bp_effect * 0.5))
    spo2        = round(sample_vital(*profile.spo2_range, age=age, age_effect=age_spo2_effect), 1)
    rr          = int(sample_vital(*profile.rr_range))
    glucose     = int(sample_vital(*profile.glucose_range))
    pulse_pressure = sbp - dbp   # derived feature

    # --- Measurement errors (5% of vitals) ---
    sbp   = int(add_measurement_error(sbp))
    dbp   = int(add_measurement_error(dbp))
    hr    = int(add_measurement_error(heart_rate))
    spo2  = round(add_measurement_error(spo2, prob=0.03), 1)
    spo2  = float(np.clip(spo2, 70, 100))

    # --- Missing values (device not available at home) ---
    # RR is harder to measure — 20% missing
    # Glucose — not everyone has glucometer — 15% missing
    # BP — 8% missing
    rr_obs      = maybe_missing(rr, missing_prob=0.20)
    glucose_obs = maybe_missing(glucose, missing_prob=0.15)
    sbp_obs     = maybe_missing(sbp, missing_prob=0.08)
    dbp_obs     = maybe_missing(dbp, missing_prob=0.08)
    # If BP missing, pulse pressure is also missing
    pp_obs      = np.nan if np.isnan(sbp_obs) else pulse_pressure

    # --- Symptom duration ---
    duration = int(np.clip(np.random.normal(profile.duration_range[0], profile.duration_range[1]),
                           profile.duration_range[2], profile.duration_range[3]))

    # --- Symptoms ---
    symptoms = generate_symptoms(profile)

    # --- Comorbidity (10% of rows) ---
    comorbidity_label = None
    if np.random.random() < 0.10:
        # Pick a disease from a DIFFERENT category as secondary
        other_diseases = [d for d in ALL_DISEASES 
                         if DISEASE_TO_CATEGORY[d] != DISEASE_TO_CATEGORY[disease_name]]
        secondary = np.random.choice(other_diseases)
        symptoms = apply_comorbidity(symptoms, secondary)
        comorbidity_label = secondary

    # --- Label noise (5% of rows) ---
    final_label = apply_label_noise(disease_name, ALL_DISEASES, noise_prob=0.05)

    # --- Assemble row ---
    row = {
        "row_id": row_id,
        "age": age,
        "gender": gender,
        "height_cm": round(height, 1),
        "weight_kg": round(weight, 1),
        "bmi": bmi,
        "temperature_c": temperature,
        "heart_rate_bpm": hr,
        "systolic_bp": sbp_obs,
        "diastolic_bp": dbp_obs,
        "pulse_pressure": pp_obs,
        "spo2_percent": spo2,
        "respiratory_rate": rr_obs,
        "blood_glucose_mgdl": glucose_obs,
        "symptom_duration_days": duration,
    }

    # Add symptom columns
    row.update({f"sym_{k}": v for k, v in symptoms.items()})

    # Labels
    row["disease"] = final_label
    row["broad_category"] = DISEASE_TO_CATEGORY[final_label]
    row["comorbidity"] = comorbidity_label if comorbidity_label else ""

    return row


# ─────────────────────────────────────────────
# 5. MAIN GENERATION LOOP
# ─────────────────────────────────────────────

def generate_dataset(n_total: int = 3000, output_path: str = "dr_friend_dataset.csv") -> pd.DataFrame:
    print(f"Generating {n_total} rows across {len(ALL_DISEASES)} diseases...")
    print(f"Class distribution (imbalanced by design):\n")

    rows = []
    row_id = 0

    for disease, weight in DISEASE_WEIGHTS.items():
        n_rows = int(n_total * weight)
        print(f"  {disease:<25} → {n_rows:>4} rows  ({weight*100:.0f}%)")
        for _ in range(n_rows):
            rows.append(generate_row(disease, row_id))
            row_id += 1

    # Shuffle rows
    np.random.shuffle(rows)
    df = pd.DataFrame(rows).reset_index(drop=True)
    df["row_id"] = df.index

    # Save
    df.to_csv(output_path, index=False)
    print(f"\n✅ Dataset saved: {output_path}")
    print(f"   Shape: {df.shape}")
    print(f"   Missing values:\n{df.isnull().sum()[df.isnull().sum() > 0]}")
    print(f"\n   Disease distribution:\n{df['disease'].value_counts()}")
    print(f"\n   Broad category distribution:\n{df['broad_category'].value_counts()}")
    return df


# ─────────────────────────────────────────────
# 6. BASIC VALIDATION
# ─────────────────────────────────────────────

def validate_dataset(df: pd.DataFrame):
    print("\n" + "="*50)
    print("DATASET VALIDATION")
    print("="*50)

    issues = []

    # Check physiological plausibility
    invalid_spo2 = df[(df["spo2_percent"] < 70) | (df["spo2_percent"] > 100)]
    if len(invalid_spo2):
        issues.append(f"Invalid SpO2 values: {len(invalid_spo2)} rows")

    invalid_hr = df[(df["heart_rate_bpm"] < 20) | (df["heart_rate_bpm"] > 250)]
    if len(invalid_hr):
        issues.append(f"Implausible HR values: {len(invalid_hr)} rows")

    invalid_temp = df[(df["temperature_c"] < 34) | (df["temperature_c"] > 43)]
    if len(invalid_temp):
        issues.append(f"Implausible temperature values: {len(invalid_temp)} rows")

    # Check symptom columns are binary
    sym_cols = [c for c in df.columns if c.startswith("sym_")]
    non_binary = [(c, df[c].unique()) for c in sym_cols if not set(df[c].dropna().unique()).issubset({0, 1})]
    if non_binary:
        issues.append(f"Non-binary symptom columns: {[c for c, _ in non_binary]}")

    # Check label noise was applied (not all rows match original disease)
    label_noise_rows = df[df["comorbidity"] != ""]
    print(f"Comorbidity rows: {len(label_noise_rows)} ({len(label_noise_rows)/len(df)*100:.1f}%)")

    if issues:
        print("\n⚠️  Issues found:")
        for i in issues:
            print(f"   - {i}")
    else:
        print("\n✅ All validation checks passed.")

    # Quick baseline: what accuracy would a dummy classifier get?
    majority_class_acc = df["disease"].value_counts().iloc[0] / len(df)
    print(f"\nDummy classifier baseline accuracy: {majority_class_acc:.3f}")
    print("(Your trained model should beat this — but not by too much for a realistic dataset)")

    # Symptom count stats
    df["symptom_count"] = df[sym_cols].sum(axis=1)
    print(f"\nSymptoms per patient: mean={df['symptom_count'].mean():.1f}, "
          f"std={df['symptom_count'].std():.1f}, "
          f"min={df['symptom_count'].min()}, max={df['symptom_count'].max()}")


if __name__ == "__main__":
    df = generate_dataset(n_total=3000, output_path="data/dr_friend_dataset.csv")
    validate_dataset(df)
