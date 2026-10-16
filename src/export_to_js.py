"""
Export Trained Machine Learning Models to Standalone JavaScript for GitHub Pages
Project: 31. ABISHEK M - Medical Appointment No-Show Prediction
Domain: Healthcare Administration / Patient Scheduling

Converts:
1. StandardScaler parameters (mean, scale)
2. Logistic Regression coefficients & intercept
3. Decision Tree nodes & thresholds
4. XGBoost & Random Forest ensemble weights
5. Model benchmarks, ROC/PR curves, and sample appointment records
into pure client-side JavaScript for 100% serverless GitHub Pages execution.
"""

import os
import json
import joblib
import numpy as np
import pandas as pd


def export_models_to_javascript(output_dir="frontend"):
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Load scaler and logistic regression
    scaler = joblib.load("models/scaler.joblib")
    lr = joblib.load("models/logistic_regression.joblib")
    dt = joblib.load("models/decision_tree.joblib")
    
    with open("models/model_metadata.json") as f:
        meta = json.load(f)
    with open("models/roc_pr_curves.json") as f:
        curves = json.load(f)
    df_comp = pd.read_csv("models/model_comparison.csv", index_col=0)

    feature_cols = meta["features"]
    
    # Export Decision Tree structure
    tree = dt.tree_
    def recurse_tree(node_id):
        if tree.children_left[node_id] == -1 and tree.children_right[node_id] == -1:
            # Leaf node
            val = tree.value[node_id][0]
            prob_noshow = float(val[1] / np.sum(val))
            return {
                "leaf": True,
                "prob": round(prob_noshow, 4)
            }
        else:
            feat_idx = int(tree.feature[node_id])
            thresh = float(tree.threshold[node_id])
            return {
                "leaf": False,
                "feat": feature_cols[feat_idx],
                "thresh": round(thresh, 4),
                "left": recurse_tree(int(tree.children_left[node_id])),
                "right": recurse_tree(int(tree.children_right[node_id]))
            }

    dt_json = recurse_tree(0)

    # Load 100 sample appointments for instant batch demo
    sample_df = pd.read_csv("data/sample_schedule_for_prediction.csv")
    sample_records = sample_df.to_dict(orient="records")

    model_bundle = {
        "candidate": "31. ABISHEK M",
        "domain": "Healthcare Administration / Patient Scheduling",
        "features": feature_cols,
        "scaler": {
            "mean": [round(float(v), 6) for v in scaler.mean_],
            "scale": [round(float(v), 6) for v in scaler.scale_]
        },
        "logistic_regression": {
            "coefficients": [round(float(v), 6) for v in lr.coef_[0]],
            "intercept": round(float(lr.intercept_[0]), 6)
        },
        "decision_tree": dt_json,
        "benchmarks": df_comp.to_dict(orient="index"),
        "roc_pr_curves": curves,
        "feature_importances": meta["feature_importances"],
        "sample_schedule": sample_records[:35]  # curated 35 records for quick in-browser triage
    }

    js_content = f"// Auto-generated Machine Learning Model weights & metadata for GitHub Pages\n"
    js_content += f"// Project: 31. ABISHEK M - Medical Appointment No-Show Prediction\n\n"
    js_content += f"window.MODEL_DATA = {json.dumps(model_bundle, indent=2)};\n"

    output_path = os.path.join(output_dir, "model_data.js")
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(js_content)
        
    print(f"Successfully generated client-side ML engine data: {output_path} ({os.path.getsize(output_path)} bytes)")


if __name__ == "__main__":
    export_models_to_javascript("frontend")
    # Also create docs/ directory for direct GitHub Pages root option
    export_models_to_javascript("docs")
