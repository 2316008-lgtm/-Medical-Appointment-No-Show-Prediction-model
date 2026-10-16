"""
Data Pipeline for Medical Appointment No-Show Prediction
Project: 31. ABISHEK M - Medical Appointment No-Show Prediction
Domain: Healthcare administration / Patient scheduling

Performs:
1. Data ingestion & validation
2. Data cleansing (invalid ages, negative lead times)
3. Leak-free chronological feature engineering (Lead time, weekday, prior appointment history)
4. Train/Test splitting (stratified)
5. Export of processed datasets
"""

import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split


def load_raw_data(filepath="data/medical_noshows.csv"):
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Raw data file not found at {filepath}")
    df = pd.read_csv(filepath)
    return df


def clean_and_engineer_features(df):
    """
    Cleans raw dataset and constructs features strictly using pre-appointment information.
    Ensures zero data leakage.
    """
    data = df.copy()

    # 1. Standardize column names (fix typos from original Kaggle dataset)
    # Hipertension -> Hypertension, Handcap -> Handicap
    column_renames = {
        'Hipertension': 'Hypertension',
        'Handcap': 'Handicap'
    }
    data.rename(columns=column_renames, inplace=True)

    # 2. Filter invalid data:
    # Age: Keep between 0 and 110 (one row has Age == -1)
    data = data[(data['Age'] >= 0) & (data['Age'] <= 110)].copy()

    # 3. Datetime conversions
    data['ScheduledDay_dt'] = pd.to_datetime(data['ScheduledDay'])
    data['AppointmentDay_dt'] = pd.to_datetime(data['AppointmentDay'])

    # 4. Lead time calculation:
    # AppointmentDay has time set to 00:00:00, ScheduledDay has exact time.
    # We normalize both to calendar date to find the lead days.
    data['LeadDays'] = (
        data['AppointmentDay_dt'].dt.normalize() - data['ScheduledDay_dt'].dt.normalize()
    ).dt.days

    # Filter invalid lead days (< 0: 5 errant records in Kaggle dataset)
    data = data[data['LeadDays'] >= 0].copy()

    # 5. Target variable: No-show ('Yes' = 1 [missed appointment], 'No' = 0 [attended])
    data['NoShow_Binary'] = (data['No-show'] == 'Yes').astype(int)

    # 6. Chronological sorting for LEAK-FREE historical patient features
    # Sort strictly by PatientId, AppointmentDay_dt, and ScheduledDay_dt
    data = data.sort_values(
        ['PatientId', 'AppointmentDay_dt', 'ScheduledDay_dt']
    ).reset_index(drop=True)

    # Cumulative prior appointments and no-shows using shift(1) to avoid leaking the current outcome
    data['Prev_Appointments'] = data.groupby('PatientId').cumcount()
    data['Prev_NoShows'] = (
        data.groupby('PatientId')['NoShow_Binary']
        .transform(lambda s: s.shift().cumsum())
        .fillna(0)
        .astype(int)
    )
    
    # Previous no-show rate (0.0 if first time patient)
    data['Prev_NoShow_Rate'] = np.where(
        data['Prev_Appointments'] > 0,
        (data['Prev_NoShows'] / data['Prev_Appointments']).round(4),
        0.0
    )
    data['Is_Repeat_Patient'] = (data['Prev_Appointments'] > 0).astype(int)

    # 7. Operational & Temporal features
    data['Appt_DayOfWeek'] = data['AppointmentDay_dt'].dt.dayofweek  # 0=Monday, 4=Friday, 5=Saturday
    data['Appt_DayName'] = data['AppointmentDay_dt'].dt.day_name()
    data['Scheduled_Hour'] = data['ScheduledDay_dt'].dt.hour
    data['SameDay_Appt'] = (data['LeadDays'] == 0).astype(int)
    
    # Lead time bins
    data['LeadTimeCategory'] = pd.cut(
        data['LeadDays'],
        bins=[-1, 0, 3, 7, 14, 30, 999],
        labels=['SameDay', 'Short_1_3d', 'Medium_4_7d', 'Fortnight_8_14d', 'Month_15_30d', 'Long_Over30d']
    ).astype(str)

    # 8. Demographic & Clinical features
    data['Gender_F'] = (data['Gender'] == 'F').astype(int)
    data['Has_Handicap'] = (data['Handicap'] > 0).astype(int)
    data['Num_Conditions'] = (
        data['Hypertension'] + data['Diabetes'] + data['Alcoholism'] + data['Has_Handicap']
    )
    
    # Age groups
    data['AgeGroup'] = pd.cut(
        data['Age'],
        bins=[-1, 12, 24, 45, 65, 120],
        labels=['Child', 'Youth', 'Adult', 'MiddleAge', 'Senior']
    ).astype(str)

    return data


def get_feature_matrix(df):
    """
    Selects numerical & encoded features suitable for scikit-learn & XGBoost.
    Excludes PatientId and AppointmentID from predictors as mandated.
    """
    feature_cols = [
        'Age',
        'Gender_F',
        'Scholarship',
        'Hypertension',
        'Diabetes',
        'Alcoholism',
        'Handicap',
        'Has_Handicap',
        'SMS_received',
        'LeadDays',
        'SameDay_Appt',
        'Scheduled_Hour',
        'Appt_DayOfWeek',
        'Num_Conditions',
        'Prev_Appointments',
        'Prev_NoShows',
        'Prev_NoShow_Rate',
        'Is_Repeat_Patient'
    ]
    
    X = df[feature_cols].copy()
    y = df['NoShow_Binary'].copy()
    
    return X, y, feature_cols


def prepare_and_save_data(raw_path="data/medical_noshows.csv", output_dir="data"):
    os.makedirs(output_dir, exist_ok=True)
    print("Step 1: Ingesting raw dataset...")
    df_raw = load_raw_data(raw_path)
    print(f"Loaded {len(df_raw)} records from {raw_path}")

    print("Step 2: Cleaning and engineering leak-free features...")
    df_clean = clean_and_engineer_features(df_raw)
    print(f"Cleaned dataset has {len(df_clean)} records and {df_clean.shape[1]} columns")

    # Save cleaned dataset
    clean_csv = os.path.join(output_dir, "cleaned_appointments.csv")
    df_clean.to_csv(clean_csv, index=False)
    print(f"Saved processed dataset to {clean_csv}")

    # Step 3: Train / Test Split (Stratified 80/20)
    X, y, feature_cols = get_feature_matrix(df_clean)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    print(f"Training set: {X_train.shape[0]} samples, {y_train.sum()} no-shows ({y_train.mean():.2%})")
    print(f"Testing set:  {X_test.shape[0]} samples, {y_test.sum()} no-shows ({y_test.mean():.2%})")

    # Save a small sample batch for web app batch prediction demonstrations
    sample_batch = df_clean.sample(n=100, random_state=42)[
        ['AppointmentID', 'PatientId', 'Gender', 'Age', 'Scholarship', 'Hypertension',
         'Diabetes', 'Alcoholism', 'Handicap', 'SMS_received', 'LeadDays', 'SameDay_Appt',
         'Scheduled_Hour', 'Appt_DayOfWeek', 'Num_Conditions', 'Prev_Appointments',
         'Prev_NoShows', 'Prev_NoShow_Rate', 'Is_Repeat_Patient', 'No-show']
    ]
    sample_csv = os.path.join(output_dir, "sample_schedule_for_prediction.csv")
    sample_batch.to_csv(sample_csv, index=False)
    print(f"Saved sample batch for demo to {sample_csv}")

    return X_train, X_test, y_train, y_test, feature_cols, df_clean


if __name__ == "__main__":
    prepare_and_save_data()
