"""
Inference & Prediction Pipeline for Medical Appointment No-Show Prediction
Project: 31. ABISHEK M - Medical Appointment No-Show Prediction
Domain: Healthcare administration / Patient scheduling

Provides single-patient scoring and batch scheduling triage.
"""

import os
import json
import joblib
import pandas as pd
import numpy as np


class AppointmentPredictor:
    def __init__(self, models_dir="models"):
        self.models_dir = models_dir
        self.scaler = joblib.load(os.path.join(models_dir, "scaler.joblib"))
        
        self.models = {
            "Logistic Regression": joblib.load(os.path.join(models_dir, "logistic_regression.joblib")),
            "Decision Tree": joblib.load(os.path.join(models_dir, "decision_tree.joblib")),
            "Random Forest": joblib.load(os.path.join(models_dir, "random_forest.joblib")),
            "XGBoost": joblib.load(os.path.join(models_dir, "xgboost.joblib"))
        }
        
        with open(os.path.join(models_dir, "model_metadata.json")) as f:
            self.metadata = json.load(f)
            
        self.feature_cols = self.metadata["features"]

    def _prepare_features(self, input_dict):
        """
        Converts human-friendly patient appointment details into model input vector.
        """
        age = float(input_dict.get("Age", 35))
        gender_f = 1 if str(input_dict.get("Gender", "F")).upper() in ["F", "FEMALE", "1"] else 0
        scholarship = int(input_dict.get("Scholarship", 0))
        hypertension = int(input_dict.get("Hypertension", input_dict.get("Hipertension", 0)))
        diabetes = int(input_dict.get("Diabetes", 0))
        alcoholism = int(input_dict.get("Alcoholism", 0))
        handicap = int(input_dict.get("Handicap", input_dict.get("Handcap", 0)))
        has_handicap = 1 if handicap > 0 else 0
        sms_received = int(input_dict.get("SMS_received", 0))
        
        # Lead time & dates
        lead_days = input_dict.get("LeadDays")
        if lead_days is None:
            # If dates provided instead of direct lead days
            sched_dt = pd.to_datetime(input_dict.get("ScheduledDay", pd.Timestamp.now()))
            appt_dt = pd.to_datetime(input_dict.get("AppointmentDay", sched_dt))
            lead_days = max(0, (appt_dt.normalize() - sched_dt.normalize()).days)
            sched_hour = sched_dt.hour
            appt_dow = appt_dt.dayofweek
        else:
            lead_days = max(0, int(lead_days))
            sched_hour = int(input_dict.get("Scheduled_Hour", 10))
            appt_dow = int(input_dict.get("Appt_DayOfWeek", 2))
            
        same_day = 1 if lead_days == 0 else 0
        num_conditions = hypertension + diabetes + alcoholism + has_handicap
        
        # Historical Patient features
        prev_appts = int(input_dict.get("Prev_Appointments", 0))
        prev_noshows = int(input_dict.get("Prev_NoShows", 0))
        prev_rate = float(input_dict.get("Prev_NoShow_Rate", (prev_noshows / prev_appts) if prev_appts > 0 else 0.0))
        is_repeat = 1 if prev_appts > 0 else 0
        
        row = {
            'Age': age,
            'Gender_F': gender_f,
            'Scholarship': scholarship,
            'Hypertension': hypertension,
            'Diabetes': diabetes,
            'Alcoholism': alcoholism,
            'Handicap': handicap,
            'Has_Handicap': has_handicap,
            'SMS_received': sms_received,
            'LeadDays': lead_days,
            'SameDay_Appt': same_day,
            'Scheduled_Hour': sched_hour,
            'Appt_DayOfWeek': appt_dow,
            'Num_Conditions': num_conditions,
            'Prev_Appointments': prev_appts,
            'Prev_NoShows': prev_noshows,
            'Prev_NoShow_Rate': prev_rate,
            'Is_Repeat_Patient': is_repeat
        }
        
        df_row = pd.DataFrame([row])[self.feature_cols]
        return df_row, row

    def predict_single(self, patient_dict, model_name="XGBoost", threshold=0.5):
        """
        Performs inference for an individual appointment booking.
        """
        if model_name not in self.models:
            model_name = "XGBoost"
            
        clf = self.models[model_name]
        df_feats, processed_dict = self._prepare_features(patient_dict)
        
        if model_name == "Logistic Regression":
            X_input = self.scaler.transform(df_feats)
        else:
            X_input = df_feats
            
        prob = float(clf.predict_proba(X_input)[0, 1])
        prediction = 1 if prob >= threshold else 0
        
        # Clinical Risk Tiering
        if prob < 0.25:
            risk_tier = "Low Risk"
            risk_color = "#10B981"  # Emerald Green
            recommendations = [
                "Standard automated appointment reminder via SMS/Email 24 hours prior.",
                "Routine clinic scheduling slot; no special intervention required."
            ]
        elif prob < 0.50:
            risk_tier = "Moderate Risk"
            risk_color = "#F59E0B"  # Amber/Yellow
            recommendations = [
                "Dispatch 2-step automated reminder (SMS 3 days prior + WhatsApp/SMS 24h prior).",
                "Enable 1-click self-service reschedule link in the SMS to encourage cancellation notice.",
                "Flag for standby waitlist patient backfill if slot opens."
            ]
        else:
            risk_tier = "High Risk"
            risk_color = "#EF4444"  # Rose Red
            recommendations = [
                "High priority: Direct telephone confirmation call by patient care coordinator 48 hours prior.",
                "Screen for social determinants: Assess transport barriers, childcare, or mobility assistance.",
                "Consider pairing with a telehealth / virtual consultation option if travel is an obstacle.",
                "Clinical operational safeguard: Eligible for strategic double-booking or rapid standby queue."
            ]

        # Top local risk factors
        risk_factors = []
        if processed_dict['LeadDays'] > 14:
            risk_factors.append(f"Extended booking lead time of {processed_dict['LeadDays']} days increases lapse risk.")
        if processed_dict['SameDay_Appt'] == 1:
            risk_factors.append("Same-day appointment significantly decreases no-show likelihood.")
        if processed_dict['Prev_NoShows'] > 0:
            risk_factors.append(f"Patient has missed {processed_dict['Prev_NoShows']} appointment(s) previously.")
        if processed_dict['SMS_received'] == 0 and processed_dict['LeadDays'] > 2:
            risk_factors.append("No SMS confirmation received yet for a multi-day advance booking.")
        if processed_dict['Age'] < 25:
            risk_factors.append("Younger patient demographic historically exhibits higher scheduling volatility.")
        if processed_dict['Scholarship'] == 1:
            risk_factors.append("Social assistance program enrollee; potential socioeconomic access constraints.")

        if not risk_factors:
            risk_factors.append("Standard patient profile without elevated risk indicators.")

        return {
            "model_used": model_name,
            "no_show_probability": round(prob, 4),
            "predicted_no_show": bool(prediction),
            "attendance_prediction": "Will Miss Appointment (No-Show)" if prediction == 1 else "Will Attend Appointment",
            "risk_tier": risk_tier,
            "risk_color": risk_color,
            "threshold_used": threshold,
            "recommendations": recommendations,
            "key_drivers": risk_factors,
            "processed_features": processed_dict
        }

    def predict_batch(self, df_input, model_name="XGBoost", threshold=0.5):
        """
        Scores an entire batch of upcoming patient appointments.
        """
        results = []
        for idx, row in df_input.iterrows():
            res = self.predict_single(row.to_dict(), model_name=model_name, threshold=threshold)
            results.append({
                "Index": idx,
                "NoShow_Probability": res["no_show_probability"],
                "Predicted_NoShow": res["predicted_no_show"],
                "Risk_Tier": res["risk_tier"],
                "Primary_Recommendation": res["recommendations"][0]
            })
            
        df_scored = pd.concat([df_input.reset_index(drop=True), pd.DataFrame(results).drop(columns=['Index'])], axis=1)
        # Sort descending by risk score
        df_scored = df_scored.sort_values("NoShow_Probability", ascending=False).reset_index(drop=True)
        return df_scored


if __name__ == "__main__":
    predictor = AppointmentPredictor()
    
    # Test case 1: High risk patient (long lead time, previous no-shows, youth)
    patient_high_risk = {
        "Age": 21,
        "Gender": "F",
        "LeadDays": 25,
        "Scholarship": 1,
        "Hypertension": 0,
        "Diabetes": 0,
        "Alcoholism": 0,
        "Handicap": 0,
        "SMS_received": 0,
        "Prev_Appointments": 4,
        "Prev_NoShows": 3,
        "Appt_DayOfWeek": 4
    }
    
    # Test case 2: Low risk patient (same-day booking, older, perfect prior attendance)
    patient_low_risk = {
        "Age": 68,
        "Gender": "M",
        "LeadDays": 0,
        "Scholarship": 0,
        "Hypertension": 1,
        "Diabetes": 1,
        "Alcoholism": 0,
        "Handicap": 0,
        "SMS_received": 1,
        "Prev_Appointments": 6,
        "Prev_NoShows": 0,
        "Appt_DayOfWeek": 1
    }
    
    print("=== High Risk Test Case ===")
    print(predictor.predict_single(patient_high_risk, model_name="XGBoost"))
    
    print("\n=== Low Risk Test Case ===")
    print(predictor.predict_single(patient_low_risk, model_name="XGBoost"))
