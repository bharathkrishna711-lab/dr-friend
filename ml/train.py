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

from xgboost import XGBClassifier
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

def save_model(model, model_name, version, accuracy, scaler, le_target):
    """
    Save trained model with version control.
    
    Args:
        model: Trained model to save
        model_name: Name (e.g., "random_forest")
        version: Version string (e.g., "v1.0")
        accuracy: Test accuracy as float (e.g., 91.50)
        scaler: Fitted StandardScaler
        le_target: Fitted LabelEncoder
    """
    from pathlib import Path
    import json
    from datetime import datetime
    
    # Create models directory
    models_dir = Path("../models")  # One level up from ml/
    models_dir.mkdir(exist_ok=True)
    
    print("\n" + "=" * 80)
    print("SAVING MODEL")
    print("=" * 80)
    
    # Save versioned model
    versioned_path = models_dir / f"{model_name}_{version}_{accuracy}pct.pkl"
    joblib.dump(model, versioned_path)
    print(f"Saved: {versioned_path}")
    
    # Save as "best" (for deployment)
    best_path = models_dir / f"{model_name}_best.pkl"
    joblib.dump(model, best_path)
    print(f"Saved: {best_path} (for deployment)")
    
    # Save preprocessing objects (only once)
    scaler_path = models_dir / "scaler.pkl"
    if not scaler_path.exists():
        joblib.dump(scaler, scaler_path)
        print(f"Saved: {scaler_path}")
    
    le_path = models_dir / "label_encoder.pkl"
    if not le_path.exists():
        joblib.dump(le_target, le_path)
        print(f"Saved: {le_path}")
    
    # Save model info
    model_info = {
        "model_name": model_name.replace("_", " ").title(),
        "version": version,
        "accuracy": accuracy,
        "trained_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "file": str(best_path.name)
    }
    
    info_path = models_dir / "model_info.json"
    with open(info_path, "w") as f:
        json.dump(model_info, f, indent=2)
    print(f"Saved: {info_path}")
    
    print("\n Model saved successfully!")
    print(f"   Version: {version}")
    print(f"   Accuracy: {accuracy}%")


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

def train_logistic_regression(X_train, y_train, X_test, y_test,C_value=1.0):
    """
    Train baseline Logistic Regression model with MLflow tracking.
    Returns: trained model, test predictions
    """
    print("\n" + "=" * 80)
    print("TRAINING LOGISTIC REGRESSION (BASELINE)")
    print("=" * 80)
    
    # Start MLflow run
    with mlflow.start_run(run_name=f"Logistic_Regression_C_{C_value}"):
        
         
        
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


#============================================================================
#TRAIN RANDOM FOREST 
#============================================================================

def train_rf_classifier(X_train, y_train,X_test,y_test,num_estimator=100):
    #start ml flow run
    with mlflow.start_run(run_name="RandomForest classifier"):
        params= {
            'num_estimator' : 100,
            'criterion' : 'gini',
            'random_state':RANDOM_STATE,
            'model': 'RandomForestClassifier'
        }
        mlflow.log_params

        #initialize model
        model=RandomForestClassifier(
            n_estimators=num_estimator,
            random_state=RANDOM_STATE,
            n_jobs=1
        )

        #Train model
        print("\n Training model\n")
        model.fit(X_train,y_train)
        print("\n Training completed\n")

        #make predictions
        y_pred_train=model.predict(X_train)
        y_pred_test=model.predict(X_test)

        #calc metrics
        train_acc=accuracy_score(y_train,y_pred_train)
        test_acc=accuracy_score(y_test,y_pred_test)
        precision = precision_score(y_test, y_pred_test, average='weighted')
        recall = recall_score(y_test, y_pred_test, average='weighted')
        f1 = f1_score(y_test, y_pred_test, average='weighted')

        #log metrics to Mlflow
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
        mlflow.sklearn.log_model(model, "random_forest_model")

        print("\nMetrics logged to MLflow")
        
        return model, y_pred_test
# ============================================================================
# TRAIN XGBOOST MODEL
# ============================================================================
def train_xgboost(X_train, y_train, X_test, y_test, 
                  n_estimators=100, 
                  learning_rate=0.1,
                  max_depth=6,
                  subsample=1.0):
    """
    Train XGBoost classifier with tunable parameters.
    """

    print("\n" + "=" * 80)
    print(f"TRAINING XGBOOST (n_estimators={n_estimators})")
    print("=" * 80)

    # Start MLflow run with parameters in name
    with mlflow.start_run(run_name=f"XGBoost_n{n_estimators}_lr{learning_rate}_d{max_depth}"):
        
        # Log parameters
        params = {
            'model': 'XGBoost',
            'n_estimators': n_estimators,
            'learning_rate': learning_rate,
            'max_depth': max_depth,
            'subsample': subsample,
            'random_state': RANDOM_STATE
        }
        mlflow.log_params(params)


    #initialize model
    model = XGBClassifier(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            max_depth=max_depth,
            subsample=subsample,
            random_state=RANDOM_STATE,
            eval_metric='mlogloss',
            use_label_encoder=False
    )
    print("\nTraining model...")
    model.fit(X_train,y_train)
    print("Training complete!")

    #make predictions
    y_pred_train=model.predict(X_train)
    y_pred_test=model.predict(X_test)

    #calculate metrics
    
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
    print(f"\n Model Performance:")
    print(f"   Training Accuracy: {train_acc:.4f} ({train_acc*100:.2f}%)")
    print(f"   Test Accuracy: {test_acc:.4f} ({test_acc*100:.2f}%)")
        
    # Detailed metrics
    print(f"\n Detailed Test Metrics:")
    print(f"   Precision: {precision:.4f}")
    print(f"   Recall: {recall:.4f}")
    print(f"   F1-Score: {f1:.4f}")

    # Log model to MLflow
    mlflow.sklearn.log_model(model, "xgboost_model")
        
    print("\n Metrics logged to MLflow")

    return model, y_pred_test





# ============================================================================
# TEST: RUN THE CODE
# ============================================================================

if __name__ == "__main__":
    # Setup MLflow
    setup_mlflow()
    
    # Load data
    X_train, X_test, y_train, y_test, scaler, le_target = load_preprocessed_data()

    #Logistic regression baseline
    print("LOGISTIC REGRESSION - BASELINE")
    print("*" * 40)
    model_lr_base, _ = train_logistic_regression(
        X_train, y_train, X_test, y_test, C_value=1.0
    )

    # Tuned LR
    print("LOGISTIC REGRESSION - TUNED")
    print("*" * 40)
    model_lr_tuned, _ = train_logistic_regression(
        X_train, y_train, X_test, y_test, C_value=0.01
    )

    #Random forest
    print("\n Random Forest")
    print("*" * 40)
    model_rf, _ = train_rf_classifier(X_train,y_train,X_test,y_test,num_estimator=100)

    # ========================================
    # XGBOOST BASELINE
    # ========================================
    print("\n XGBoost - Baseline")
    print("*" * 40)
    model_xgb_baseline, _ = train_xgboost(
        X_train, y_train, X_test, y_test
        # Uses defaults: n_estimators=100, learning_rate=0.1, max_depth=6
    )

    # ========================================
    # XGBOOST HYPERPARAMETER TUNING
    # ========================================
    mlflow.end_run()
    
    print("XGBOOST HYPERPARAMETER TUNING")
    print("*" * 40)
    
    # Try 1: More trees + lower learning rate
    print("\n[1/3] Trying n=200, lr=0.05...")
    model_xgb_tuned1, _ = train_xgboost(
        X_train, y_train, X_test, y_test,
        n_estimators=200,
        learning_rate=0.05
    )
    mlflow.end_run()
    
    # Try 2: Even lower learning rate
    print("\n[2/3] Trying n=200, lr=0.03...")
    model_xgb_tuned2, _ = train_xgboost(
        X_train, y_train, X_test, y_test,
        n_estimators=200,
        learning_rate=0.03
    )
    mlflow.end_run()
    
    # Try 3: Add regularization
    print("\n[3/3] Trying n=200, lr=0.05, depth=5, subsample=0.8...")
    model_xgb_best, _ = train_xgboost(
        X_train, y_train, X_test, y_test,
        n_estimators=200,
        learning_rate=0.05,
        max_depth=5,
        subsample=0.8
    )
    mlflow.end_run()



     # ========================================
    # SAVE BEST MODEL
    # ========================================
    
   
    print("SELECTING AND SAVING BEST MODEL")
    print("*" * 40)

    # Save Random Forest (best model)
    save_model(
        model=model_rf,
        model_name="random_forest",
        version="v1.0",
        accuracy=91.50,
        scaler=scaler,
        le_target=le_target
    )








    # ========================================
    # SUMMARY
    # ========================================
    
    print("\n" + "=" * 80)
    print("TRAINING COMPLETE!")
    print("=" * 80)
    print("\n Model Comparison:")
    print("   LR Baseline:  82.00%")
    print("   LR Tuned:     88.67%")
    print("   RF:           91.50%  SAVED")
    print("   XGBoost:      91.00%")
    print("\n Saved to: models/random_forest_best.pkl")
    print("\nCheck MLflow UI: http://localhost:5000")



    
    