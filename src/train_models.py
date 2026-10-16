"""
Model Training and Comparison Engine
Project: 31. ABISHEK M - Medical Appointment No-Show Prediction
Domain: Healthcare administration / Patient scheduling

Trains, tunes, and evaluates:
1. Logistic Regression
2. Decision Tree Classifier
3. Random Forest Classifier
4. XGBoost Classifier

Generates full performance benchmarks, ROC/PR data, confusion matrices, and feature importances.
"""

import os
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    classification_report,
    roc_curve,
    precision_recall_curve
)

from data_pipeline import prepare_and_save_data, get_feature_matrix, load_raw_data, clean_and_engineer_features


def compute_metrics(y_true, y_pred, y_prob):
    """Computes comprehensive clinical and statistical performance metrics."""
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    
    accuracy = float(accuracy_score(y_true, y_pred))
    precision = float(precision_score(y_true, y_pred, zero_division=0))
    recall = float(recall_score(y_true, y_pred, zero_division=0))
    f1 = float(f1_score(y_true, y_pred, zero_division=0))
    specificity = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0
    
    roc_auc = float(roc_auc_score(y_true, y_prob)) if y_prob is not None else 0.0
    pr_auc = float(average_precision_score(y_true, y_prob)) if y_prob is not None else 0.0
    
    return {
        "accuracy": round(accuracy, 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "specificity": round(specificity, 4),
        "f1_score": round(f1, 4),
        "roc_auc": round(roc_auc, 4),
        "pr_auc": round(pr_auc, 4),
        "tp": int(tp),
        "fp": int(fp),
        "tn": int(tn),
        "fn": int(fn)
    }


def train_and_evaluate_all():
    os.makedirs("models", exist_ok=True)
    
    # 1. Load cleaned data or prepare
    clean_csv_path = "data/cleaned_appointments.csv"
    if not os.path.exists(clean_csv_path):
        print("Cleaned data not found. Running preparation...")
        prepare_and_save_data()
        
    print(f"Loading cleaned data from {clean_csv_path}...")
    df = pd.read_csv(clean_csv_path)
    X, y, feature_cols = get_feature_matrix(df)
    
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    
    print(f"Features ({len(feature_cols)}): {feature_cols}")
    print(f"Train samples: {len(X_train)}, Test samples: {len(X_test)}")
    
    # Standard Scaler for Logistic Regression (and distance/linear baselines)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    joblib.dump(scaler, "models/scaler.joblib")
    
    scale_pos_weight = (len(y_train) - y_train.sum()) / y_train.sum()
    print(f"Calculated negative-to-positive ratio (scale_pos_weight): {scale_pos_weight:.2f}")

    # Define the 4 specified models
    models_dict = {
        "Logistic Regression": {
            "model": LogisticRegression(
                C=0.5,
                max_iter=1000,
                class_weight='balanced',
                random_state=42,
                solver='lbfgs'
            ),
            "use_scaled": True,
            "filename": "logistic_regression.joblib"
        },
        "Decision Tree": {
            "model": DecisionTreeClassifier(
                max_depth=7,
                min_samples_split=20,
                min_samples_leaf=15,
                class_weight='balanced',
                random_state=42
            ),
            "use_scaled": False,
            "filename": "decision_tree.joblib"
        },
        "Random Forest": {
            "model": RandomForestClassifier(
                n_estimators=150,
                max_depth=12,
                min_samples_split=15,
                min_samples_leaf=6,
                class_weight='balanced',
                random_state=42,
                n_jobs=-1
            ),
            "use_scaled": False,
            "filename": "random_forest.joblib"
        },
        "XGBoost": {
            "model": XGBClassifier(
                n_estimators=200,
                max_depth=5,
                learning_rate=0.08,
                scale_pos_weight=scale_pos_weight,
                subsample=0.85,
                colsample_bytree=0.85,
                eval_metric='auc',
                random_state=42,
                n_jobs=-1
            ),
            "use_scaled": False,
            "filename": "xgboost.joblib"
        }
    }

    results = {}
    curves_data = {}
    feature_importances = {}

    for name, config in models_dict.items():
        print(f"\n==================================================")
        print(f"Training {name}...")
        clf = config["model"]
        use_scaled = config["use_scaled"]
        
        train_X = X_train_scaled if use_scaled else X_train
        test_X = X_test_scaled if use_scaled else X_test
        
        # Fit
        clf.fit(train_X, y_train)
        
        # Predict on Test Set
        y_pred = clf.predict(test_X)
        y_prob = clf.predict_proba(test_X)[:, 1]
        
        # Compute Metrics
        metrics = compute_metrics(y_test, y_pred, y_prob)
        results[name] = metrics
        
        print(f"Results for {name}:")
        print(f"  Accuracy:    {metrics['accuracy']:.4f}")
        print(f"  Recall:      {metrics['recall']:.4f} (Sensitivity to No-Shows)")
        print(f"  Precision:   {metrics['precision']:.4f}")
        print(f"  Specificity: {metrics['specificity']:.4f}")
        print(f"  F1 Score:    {metrics['f1_score']:.4f}")
        print(f"  ROC-AUC:     {metrics['roc_auc']:.4f}")
        print(f"  PR-AUC:      {metrics['pr_auc']:.4f}")
        
        # Save model
        joblib.dump(clf, os.path.join("models", config["filename"]))
        
        # Compute ROC & PR Curve points (subsampled to 200 points for light JSON payload)
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        prec, rec, _ = precision_recall_curve(y_test, y_prob)
        
        step_roc = max(1, len(fpr) // 150)
        step_pr = max(1, len(prec) // 150)
        
        curves_data[name] = {
            "fpr": [round(float(v), 4) for v in fpr[::step_roc]] + [1.0],
            "tpr": [round(float(v), 4) for v in tpr[::step_roc]] + [1.0],
            "precision": [round(float(v), 4) for v in prec[::step_pr]],
            "recall": [round(float(v), 4) for v in rec[::step_pr]]
        }
        
        # Feature Importances / Coefficients
        if hasattr(clf, "feature_importances_"):
            importances = clf.feature_importances_
            feature_importances[name] = {
                feat: round(float(imp), 4)
                for feat, imp in sorted(zip(feature_cols, importances), key=lambda x: x[1], reverse=True)
            }
        elif hasattr(clf, "coef_"):
            coefs = clf.coef_[0]
            feature_importances[name] = {
                feat: round(float(coef), 4)
                for feat, coef in sorted(zip(feature_cols, coefs), key=lambda x: abs(x[1]), reverse=True)
            }

    # Save summary dataframe
    df_results = pd.DataFrame(results).T
    df_results.index.name = "Model"
    df_results.to_csv("models/model_comparison.csv")
    print("\n==================================================")
    print("Final Model Benchmark Table:")
    print(df_results[['accuracy', 'recall', 'precision', 'specificity', 'f1_score', 'roc_auc', 'pr_auc']])

    # Save JSON metadata
    metadata = {
        "features": feature_cols,
        "metrics": results,
        "feature_importances": feature_importances,
        "test_samples": len(y_test),
        "test_noshows": int(y_test.sum()),
        "test_shows": int((y_test == 0).sum())
    }
    with open("models/model_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    with open("models/roc_pr_curves.json", "w") as f:
        json.dump(curves_data, f, indent=2)

    print("\nSaved all models and performance metadata to /models successfully.")
    return df_results


if __name__ == "__main__":
    train_and_evaluate_all()
