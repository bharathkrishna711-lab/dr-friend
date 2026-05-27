"""
Data Preprocessing Pipeline
Author: Bharath Krishna
Project: Dr. Friend
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from imblearn.over_sampling import SMOTE
from config import *

# ============================================================================
# STEP 1: LOAD DATA
# ============================================================================

def load_data():
    """Load the dataset."""
    print("=" * 80)
    print("LOADING DATASET")
    print("=" * 80)
    
    df = pd.read_csv(DATASET_PATH)
    print(f"Dataset loaded: {df.shape[0]} rows, {df.shape[1]} columns")
    print(f"Target variable: {TARGET_COL}")
    print(f"Classes: {df[TARGET_COL].nunique()}")
    
    return df

# ============================================================================
# STEP 2: HANDLE MISSING VALUES
# ============================================================================

def handle_missing_values(df):
    """Impute missing values with median for numerical columns."""
    print("\n" + "=" * 80)
    print("HANDLING MISSING VALUES")
    print("=" * 80)
    
    # Check missing values before
    missing_before = df.isnull().sum().sum()
    print(f"Missing values before: {missing_before}")
    
    # Impute numerical columns with median
    numerical_cols = df.select_dtypes(include=[np.number]).columns
    for col in numerical_cols:
        if df[col].isnull().sum() > 0:
            median_val = df[col].median()
            df[col].fillna(median_val, inplace=True)
            print(f" {col}: Filled {df[col].isnull().sum()} with median {median_val:.2f}")
    
    # Fill categorical missing with 'none'
    categorical_cols = df.select_dtypes(include=['object']).columns
    for col in categorical_cols:
        if col != TARGET_COL and df[col].isnull().sum() > 0:
            df[col].fillna('none', inplace=True)
            print(f" {col}: Filled with 'none'")
    
    missing_after = df.isnull().sum().sum()
    print(f"\nMissing values after: {missing_after}")
    
    return df

# ============================================================================
# STEP 3: FEATURE ENGINEERING
# ============================================================================

def engineer_features(df):
    """Create derived features."""
    print("\n" + "=" * 80)
    print("FEATURE ENGINEERING")
    print("=" * 80)
    
    # Age groups
    df['age_group'] = pd.cut(df['age'], bins=AGE_BINS, labels=AGE_LABELS)
    print(" Created age_group")
    
    # Fever category
    df['fever_category'] = pd.cut(df['temperature_c'], bins=TEMP_BINS, labels=TEMP_LABELS)
    print(" Created fever_category")
    
    # Symptom count
    symptom_cols = [col for col in df.columns if col.startswith('sym_')]
    df['symptom_count'] = df[symptom_cols].sum(axis=1)
    print(f" Created symptom_count (avg: {df['symptom_count'].mean():.2f})")
    
    # Vital risk score (composite)
    risk_score = 0
    risk_score += (df['temperature_c'] > 38.5).astype(int) * 2
    risk_score += (df['spo2_percent'] < 92).astype(int) * 3
    risk_score += (df['heart_rate_bpm'] > 100).astype(int) * 1
    risk_score += (df['systolic_bp'] > 140).astype(int) * 1
    df['vital_risk_score'] = risk_score
    print(f" Created vital_risk_score (avg: {df['vital_risk_score'].mean():.2f})")
    
    return df

# ============================================================================
# STEP 4: ENCODE CATEGORICAL VARIABLES
# ============================================================================

def encode_categorical(df):
    """Encode categorical variables."""
    print("\n" + "=" * 80)
    print("ENCODING CATEGORICAL VARIABLES")
    print("=" * 80)
    
    # Label encode target first
    le_target = LabelEncoder()
    df[TARGET_COL] = le_target.fit_transform(df[TARGET_COL])
    print(f" Target encoded: {len(le_target.classes_)} classes")
    
    # Find all object (string) columns
    object_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    
    # Remove TARGET_COL and row_id if they somehow got included
    object_cols = [col for col in object_cols if col not in [TARGET_COL, 'row_id']]
    
    if object_cols:
        print(f" Encoding columns: {object_cols}")
        # One-hot encode all categorical columns
        df = pd.get_dummies(df, columns=object_cols, drop_first=True, dtype=int)
        print(f"  One-hot encoding complete")
    
    # Ensure all columns are numeric
    for col in df.columns:
        if df[col].dtype == 'object':
            print(f"  ⚠️  Converting {col} to numeric")
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
    
    print(f"  Final feature count: {len(df.columns)}")
    
    return df, le_target

# ============================================================================
# STEP 5: SPLIT DATA
# ============================================================================

def split_data(df):
    """Split into train and test sets."""
    print("\n" + "=" * 80)
    print("TRAIN/TEST SPLIT")
    print("=" * 80)
    
    # Separate features and target
    X = df.drop(columns=[TARGET_COL])

    # Drop only broad_category - directly derived from disease label (leakage) (faced data leakage evident from low confidence level)
    # comorbidity is kept - assigned independently and modifies symptoms realistically
    leaky_cols = [col for col in X.columns 
                if col.startswith('broad_category_')]
    X = X.drop(columns=leaky_cols)
    print(f"  Dropped {len(leaky_cols)} leaky feature columns")


    y = df[TARGET_COL]
    
    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        test_size=TEST_SIZE, 
        random_state=RANDOM_STATE,
        stratify=y
    )
    
    print(f"  Train set: {X_train.shape[0]} samples")
    print(f"  Test set: {X_test.shape[0]} samples")
    print(f"  Features: {X_train.shape[1]}")
    
    return X_train, X_test, y_train, y_test

# ============================================================================
# STEP 6: SCALE FEATURES
# ============================================================================

def scale_features(X_train, X_test):
    """Scale features using StandardScaler."""
    print("\n" + "=" * 80)
    print("FEATURE SCALING")
    print("=" * 80)
    
    scaler = StandardScaler()
    
    # Get column names before scaling
    train_columns = X_train.columns if hasattr(X_train, 'columns') else None
    
    # Fit and transform
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    print("  Features scaled (mean=0, std=1)")
    
    # Convert back to DataFrame if columns exist
    if train_columns is not None:
        X_train_scaled = pd.DataFrame(X_train_scaled, columns=train_columns)
        X_test_scaled = pd.DataFrame(X_test_scaled, columns=train_columns)
    
    return X_train_scaled, X_test_scaled, scaler

# ============================================================================
# STEP 7: APPLY SMOTE
# ============================================================================

def apply_smote(X_train, y_train):
    """Apply SMOTE to balance classes."""
    print("\n" + "=" * 80)
    print("APPLYING SMOTE")
    print("=" * 80)
    
    print("Class distribution before SMOTE:")
    print(y_train.value_counts().sort_index())
    
    smote = SMOTE(random_state=RANDOM_STATE)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
    
    print("\nClass distribution after SMOTE:")
    print(pd.Series(y_train_resampled).value_counts().sort_index())
    
    print(f"\n  SMOTE applied")
    print(f"  Training samples: {len(y_train)} → {len(y_train_resampled)}")
    
    return X_train_resampled, y_train_resampled

# ============================================================================
# COMPLETE PIPELINE
# ============================================================================

def preprocess_pipeline():
    """Complete preprocessing pipeline."""
    print("\n" + "=" * 80)
    print("DR. FRIEND - DATA PREPROCESSING PIPELINE")
    print("=" * 80)
    
    # Step 1: Load
    df = load_data()
    
    # Step 2: Handle missing values
    df = handle_missing_values(df)
    
    # Step 3: Feature engineering
    df = engineer_features(df)
    
    # Step 4: Encode categorical
    df, le_target = encode_categorical(df)
    
    # Step 5: Split
    X_train, X_test, y_train, y_test = split_data(df)
    
    # Step 6: Scale
    X_train_scaled, X_test_scaled, scaler = scale_features(X_train, X_test)
    
    # Step 7: SMOTE (only on training data)
    X_train_final, y_train_final = apply_smote(X_train_scaled, y_train)
    
    print("\n" + "=" * 80)
    print("PREPROCESSING COMPLETE")
    print("=" * 80)
    
    return X_train_final, X_test_scaled, y_train_final, y_test, scaler, le_target

# ============================================================================
# TEST THE PIPELINE
# ============================================================================

if __name__ == "__main__":
    X_train, X_test, y_train, y_test, scaler, le_target = preprocess_pipeline()
    
    print(f"\nFinal shapes:")
    print(f"   X_train: {X_train.shape}")
    print(f"   X_test: {X_test.shape}")
    print(f"   y_train: {y_train.shape}")
    print(f"   y_test: {y_test.shape}")