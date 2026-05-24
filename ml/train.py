"""
Model Training Pipeline for Dr. Friend
Trains baseline ML models for disease prediction:
- Logistic Regression
- Random Forest
- XGBoost
- LightGBM
"""


# Standard library imports
import numpy as np
import pandas as pd
from pathlib import Path
# Scikit-learn imports
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)
import joblib
import mlflow
import mlflow.sklearn

# Import our preprocessing pipeline
from preprocess import preprocess_pipeline
from config import *

def load_preprocessed_data():
    """
    Load preprocessed training and test data.
    Returns: X_train, X_test, y_train, y_test, scaler, label_encoder
    """

    print("=" * 80)
    print("LOADING PREPROCESSED DATA")
    print("=" * 80)

    # Call preprocessing pipeline (THIS WAS MISSING!)
    X_train, X_test, y_train, y_test, scaler, le_target = preprocess_pipeline()

    


    print(f"\n Data loaded successfully!")
    print(f"   Training samples: {X_train.shape[0]}")
    print(f"   Test samples: {X_test.shape[0]}")
    print(f"   Features: {X_train.shape[1]}")
    print(f"   Classes: {len(np.unique(y_train))}")

    return X_train, X_test, y_train, y_test, scaler, le_target 

#================================================================
 #TRAIN LOGISTIC REGRESSION MODEL
#================================================================

def train_logistic_regression(X_train, y_train, X_test, y_test):
    """
    Train baseline Logistic Regression model with MLflow tracking.
    Returns: trained model, test predictions
    """
    print("\n" + "=" * 80)
    print("TRAINING LOGISTIC REGRESSION (BASELINE)")
    print("=" * 80)
    
    # Start MLflow run
    with mlflow.start_run(run_name="Logistic_Regression_Baseline"):
        C_value = 0.01
         
        
        # Log parameters
        params = {
            'C': C_value,
            'model': 'LogisticRegression',
            'max_iter': 1000,
            'solver': 'lbfgs',
            'random_state': RANDOM_STATE
        }
        mlflow.log_params(params)

         
        
        # Initialize model
        model = LogisticRegression(
            C=C_value,
            max_iter=1000,
            random_state=RANDOM_STATE,
            solver='lbfgs'
        )
        
        # Train the model
        print("\nTraining model...")
        model.fit(X_train, y_train)
        print("Training complete!")
        
        # Make predictions
        y_pred_train = model.predict(X_train)
        y_pred_test = model.predict(X_test)
        
        # Calculate metrics
        train_acc = accuracy_score(y_train, y_pred_train)
        test_acc = accuracy_score(y_test, y_pred_test)
        precision = precision_score(y_test, y_pred_test, average='weighted')
        recall = recall_score(y_test, y_pred_test, average='weighted')
        f1 = f1_score(y_test, y_pred_test, average='weighted')
        
        # Log metrics to MLflow
        mlflow.log_metric("train_accuracy", train_acc)
        mlflow.log_metric("test_accuracy", test_acc)
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall)
        mlflow.log_metric("f1_score", f1)
        mlflow.log_metric("overfitting_gap", train_acc - test_acc)
        
        # Print performance
        print(f"\nModel Performance:")
        print(f"   Training Accuracy: {train_acc:.4f} ({train_acc*100:.2f}%)")
        print(f"   Test Accuracy: {test_acc:.4f} ({test_acc*100:.2f}%)")
        
        # Detailed metrics
        print(f"\nDetailed Test Metrics:")
        print(f"   Precision: {precision:.4f}")
        print(f"   Recall: {recall:.4f}")
        print(f"   F1-Score: {f1:.4f}")
        
        # Log model to MLflow
        mlflow.sklearn.log_model(model, "logistic_regression_model")
        
        print("\n Metrics logged to MLflow")
        
        return model, y_pred_test



def setup_mlflow():
    """Initialize MLflow experiment"""
    mlflow.set_experiment("dr_friend_disease_prediction")
    print(" MLflow experiment initialized")


# ============================================================================
# TEST: RUN THE CODE
# ============================================================================

if __name__ == "__main__":
    # Setup MLflow
    setup_mlflow()
    
    # Load data
    X_train, X_test, y_train, y_test, scaler, le_target = load_preprocessed_data()
    
    # Train model
    model, y_pred = train_logistic_regression(X_train, y_train, X_test, y_test)
    
    print("\n" + "=" * 80)
    print(" TRAINING COMPLETE!")
    print("=" * 80)
    print("\nTo view MLflow UI, run: mlflow ui")
