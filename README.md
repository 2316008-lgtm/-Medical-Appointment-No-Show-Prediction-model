# 🏥 31. ABISHEK M — Medical Appointment No-Show Prediction

**Domain:** Healthcare Administration / Patient Scheduling  
**Problem Statement:** Missed appointments waste critical healthcare resources, leave provider slots idle, and cause preventable delays in patient care.  
**Objective:** Predict appointment attendance so healthcare organizations can investigate scheduling improvements, deploy targeted reminders, and optimize clinic capacity.  
**Candidate / Student ID:** `31. ABISHEK M`  
**Algorithms Implemented & Benchmarked:** `Logistic Regression`, `Decision Tree`, `Random Forest`, `XGBoost`

---

## 📌 Executive Summary

Outpatient no-shows create severe logistical and financial bottlenecks for healthcare systems worldwide. When patients miss scheduled visits without prior cancellation:
1. High-cost clinical resources (physicians, specialized diagnostic equipment, exam rooms) remain underutilized.
2. Other patients on waiting lists face prolonged delays for care.
3. Patients who miss care suffer worse clinical outcomes due to unmonitored chronic conditions.

This project delivers an end-to-end, clinically grounded machine learning solution built upon **110,516 authentic clinical appointments**. It implements **leak-free chronological feature engineering**, trains and tunes **four predictive algorithms**, and provides an **interactive Streamlit Web Dashboard** for real-time patient risk scoring, batch appointment triage, and model benchmarking.

---

## 📁 Repository Architecture

```
c:\Users\kisho\Desktop\v-p\
├── data/
│   ├── medical_noshows.csv                # Raw authentic Kaggle dataset (110,527 records)
│   ├── cleaned_appointments.csv           # Cleaned & feature-engineered dataset (110,516 records)
│   └── sample_schedule_for_prediction.csv # 100-record preloaded clinic schedule for demo
├── models/
│   ├── logistic_regression.joblib        # Trained Logistic Regression model
│   ├── decision_tree.joblib              # Trained Decision Tree classifier
│   ├── random_forest.joblib              # Trained Random Forest ensemble
│   ├── xgboost.joblib                    # Trained XGBoost gradient booster
│   ├── scaler.joblib                     # StandardScaler for continuous features
│   ├── model_metadata.json               # Metric records, feature importances, sample sizes
│   ├── model_comparison.csv              # Benchmark metric table
│   └── roc_pr_curves.json                # Precomputed coordinates for ROC and PR curves
├── src/
│   ├── __init__.py
│   ├── data_pipeline.py                  # Cleaning, leak-free feature engineering, 80/20 train/test split
│   ├── train_models.py                   # Model training, hyperparameter configuration, metric evaluation
│   └── predict.py                        # Real-time inference API for single patient and batch scoring
├── app.py                                # High-end interactive Streamlit web dashboard
├── requirements.txt                      # Project dependencies
└── README.md                             # Comprehensive technical & academic report
```

---

## 🧬 Dataset Schema & Preprocessing

The input dataset originates from municipal healthcare clinics in Vitória, Brazil, capturing 110,527 appointment records across 81 distinct clinical neighborhoods.

### 1. Raw Fields Handled
| Field Name | Type | Description | Handling in Pipeline |
| :--- | :--- | :--- | :--- |
| **`PatientId`** | Float64 | Unique patient identifier | **Excluded as direct predictor**; utilized strictly to construct leak-free chronological patient attendance histories. |
| **`AppointmentID`** | Int64 | Unique appointment transaction identifier | **Excluded as direct predictor** (administrative key). |
| **`Gender`** | String | Patient recorded gender (`M` / `F`) | Encoded as binary (`Gender_F`: 1 = Female, 0 = Male). |
| **`Age`** | Int64 | Patient age (years) | Outlier cleaning: 1 errant record with `Age = -1` removed; constrained to valid medical range [0, 110]. |
| **`ScheduledDay`** | Timestamp | Exact date and time when the appointment was booked | Converted to datetime; used to extract hour of booking (`Scheduled_Hour`) and calculate lead time. |
| **`AppointmentDay`**| Timestamp | Scheduled calendar date of the medical appointment | Converted to datetime; used to extract day of week (`Appt_DayOfWeek`) and lead time. |
| **`Scholarship`** | Binary (0/1)| Enrollment in the *Bolsa Família* social welfare program | Retained directly as socioeconomic indicator. |
| **`Hipertension`** | Binary (0/1)| Medical diagnosis of hypertension | Standardized to `Hypertension` (0/1). |
| **`Diabetes`** | Binary (0/1)| Medical diagnosis of diabetes mellitus | Retained directly (0/1). |
| **`Alcoholism`** | Binary (0/1)| Clinical indicator of alcoholism | Retained directly (0/1). |
| **`Handcap`** | Integer (0-4)| Number of physical disabilities / handicap level | Standardized to `Handicap` + binary flag `Has_Handicap` (1 if $\ge 1$, else 0). |
| **`SMS_received`** | Binary (0/1)| Whether one or more automated SMS reminders were sent | Retained directly (0/1). |
| **`No-show`** | String | Target attendance outcome (`No` = Showed up, `Yes` = No-Show) | Target encoded as binary `NoShow_Binary`: **1 for Missed Appointment (No-Show)**, **0 for Attended**. |

---

## 🛡️ Leak-Free Feature Engineering Architecture

A frequent critical flaw in healthcare scheduling models is **data leakage**—using future visit outcomes to predict a current visit. To strictly prevent leakage as instructed, all temporal features are calculated **exclusively using information available prior to the appointment**:

1. **Booking Lead Time (`LeadDays`):**
   $$\text{LeadDays} = \text{Date}(\text{AppointmentDay}) - \text{Date}(\text{ScheduledDay})$$
   - 5 errant records in the raw dataset where $\text{AppointmentDay} < \text{ScheduledDay}$ were eliminated.
   - Values range from $0$ (same-day booking) up to $179$ days.

2. **Same-Day Appointment Flag (`SameDay_Appt`):**
   - Binary indicator: $1$ if $\text{LeadDays} = 0$, else $0$.
   - Patients booking same-day visits exhibit a **95.4% attendance rate** (only 4.6% no-show rate).

3. **Appointment Day of Week (`Appt_DayOfWeek`):**
   - Extracted from `AppointmentDay` (Monday = 0 through Saturday = 5). Captures weekend and start-of-week attendance variances.

4. **Chronological Prior Attendance History (`Prev_Appointments`, `Prev_NoShows`, `Prev_NoShow_Rate`):**
   - Records are sorted chronologically by `PatientId`, `AppointmentDay`, and `ScheduledDay`.
   - Prior metrics are computed using cumulative expansion **shifted by 1 visit**:
     ```python
     df['Prev_Appointments'] = df.groupby('PatientId').cumcount()
     df['Prev_NoShows'] = df.groupby('PatientId')['NoShow_Binary'].transform(lambda s: s.shift().cumsum()).fillna(0).astype(int)
     df['Prev_NoShow_Rate'] = np.where(df['Prev_Appointments'] > 0, df['Prev_NoShows'] / df['Prev_Appointments'], 0.0)
     ```
   - For first-time patients, these values are strictly $0$. For the 48,220 repeat patient encounters, this provides a powerful, leak-free behavioral signal.

5. **Comorbidity Index (`Num_Conditions`):**
   - Additive sum: $\text{Hypertension} + \text{Diabetes} + \text{Alcoholism} + \text{Has\_Handicap}$.

---

## 🤖 Algorithms Built & Evaluated

The dataset is partitioned into an **80% Training Set (88,412 samples)** and a **20% Hold-Out Testing Set (22,104 samples)** using stratified sampling to preserve the 20.19% baseline no-show rate.

Four distinct algorithms were built, tuned, and evaluated:

### 1. Logistic Regression (Linear Probabilistic Classifier)
- **Configuration:** L2 regularized ($C=0.5$), L-BFGS solver, max iterations = 1000, `class_weight='balanced'`.
- **Scaling:** Continuous features transformed via `StandardScaler`.
- **Role:** Transparent linear baseline measuring log-odds coefficients for each clinical factor.

### 2. Decision Tree Classifier (Non-Linear Rule-Based)
- **Configuration:** Maximum depth = 7, minimum samples per split = 20, minimum samples per leaf = 15, `class_weight='balanced'`, Gini impurity.
- **Role:** Highly interpretable hierarchical decision paths for clinical guidelines.

### 3. Random Forest (Bagging Ensemble)
- **Configuration:** 150 bootstrapped estimators, max depth = 12, min samples per split = 15, min samples per leaf = 6, `class_weight='balanced'`, parallelized execution.
- **Role:** Reduces variance and captures complex multi-way feature interactions.

### 4. XGBoost (Extreme Gradient Boosting)
- **Configuration:** 200 trees, max depth = 5, learning rate $\eta = 0.08$, `scale_pos_weight = 3.95` (negative-to-positive class ratio), subsample = 0.85, colsample_bytree = 0.85, `eval_metric='auc'`.
- **Role:** State-of-the-art boosted decision trees optimizing pseudo-residuals with loss regularization.

---

## 📊 Comprehensive Model Benchmarking Results

Evaluated on the **22,104 hold-out test appointments** (4,462 actual No-Shows, 17,642 actual Attendees):

| Algorithm | Accuracy | Recall (Sensitivity) | Precision (PPV) | Specificity (TNR) | F1-Score | ROC-AUC | PR-AUC |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Logistic Regression** | 57.54% | **84.85%** | 30.30% | 50.63% | 0.4465 | 0.7305 | 0.3758 |
| **Decision Tree** | 58.87% | 82.05% | 30.63% | 53.00% | 0.4461 | 0.7331 | 0.3731 |
| **Random Forest** | 59.73% | 82.03% | 31.12% | 54.09% | 0.4512 | 0.7433 | 0.4053 |
| **XGBoost (Champion)** | **60.87%** | **80.14%** | **31.53%** | **55.99%** | **0.4526** | **0.7454** | **0.4079** |

### Confusion Matrix Breakdown (Test Set, N = 22,104)

| Model | True Negatives (TN) | False Positives (FP) | False Negatives (FN) | True Positives (TP) |
| :--- | :---: | :---: | :---: | :---: |
| **Logistic Regression** | 8,932 | 8,710 | **676** | **3,786** |
| **Decision Tree** | 9,350 | 8,292 | 801 | 3,661 |
| **Random Forest** | 9,542 | 8,100 | 802 | 3,660 |
| **XGBoost** | **9,878** | **7,764** | 886 | 3,576 |

### 💡 Clinical & Operational Insights
1. **Recall Over Accuracy:** In healthcare scheduling, an unaddressed no-show costs an estimated **$150 – $250** in wasted clinical capacity. Catching **80% to 85% of all missed appointments** allows the clinic to proactively intervene via automated reminders or telephone outreach.
2. **Top Predictive Drivers Identified:**
   - **`SameDay_Appt`:** Booking on the day of the appointment drops no-show probability to under 5%.
   - **`LeadDays`:** As booking lead time extends beyond 14 days, no-show rates jump past 32%.
   - **`Prev_NoShows` & `Prev_NoShow_Rate`:** Past behavior is a top indicator of future attendance.
   - **`Age`:** Young adults (15–25) have the highest cancellation rates (~26%); elderly patients (>65) have the highest compliance (~14%).
   - **`Scholarship`:** Socioeconomic hardship correlates with higher transport and scheduling barriers.

---

## 🖥️ Interactive Streamlit Web Application

The project includes an interactive web dashboard running on `http://localhost:8501`:

### Dashboard Features:
1. **🩺 Patient Risk Predictor:**
   - Real-time sliders, date pickers, comorbidity toggles, and historical visit inputs.
   - Live probability calculation, color-coded risk meter (Low / Moderate / High), and customized clinical action plan.
2. **📊 Model Benchmarking:**
   - Side-by-side metric scorecard for all 4 models.
   - Interactive Plotly ROC curves and Precision-Recall curves.
   - Interactive Confusion Matrices with TP/FP/TN/FN breakdowns.
3. **📅 Batch Scheduling Triage:**
   - Instant scoring of multi-patient rosters (preloaded 100-appointment sample or custom CSV upload).
   - Color-coded triage queue sorted by risk level.
   - One-click CSV export of prioritized phone outreach queues for hospital coordinators.
4. **📈 Healthcare Data Insights (EDA):**
   - Interactive visualizations of Lead Time vs Attendance, Life Stage analysis, Weekday patterns, and Comorbidity impact.
5. **🧠 Feature Importance & Explainability:**
   - Visualizing feature weights and gain rankings across models.

---

## 🚀 Quickstart Guide

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Run Data Pipeline (Cleaning & Feature Engineering)
```bash
python src/data_pipeline.py
```

### 3. Train & Benchmark All 4 Models
```bash
python src/train_models.py
```

### 4. Run Test Inference
```bash
python src/predict.py
```

### 5. Launch the Web Application
```bash
streamlit run app.py
```
Open **`http://localhost:8501`** in your browser.

---

## 📜 Compliance & Verification

- **Patient Identifier Hygiene:** `PatientId` and `AppointmentID` are excluded as direct predictors. `PatientId` is used exclusively for chronological history compilation.
- **Zero Data Leakage:** All lead times, weekday indicators, and prior attendance records are calculated strictly from pre-appointment information.
- **Algorithms:** Logistic Regression, Decision Tree, Random Forest, and XGBoost fully trained, benchmarked, and serialized.
- **Student Attribution:** Project configured for **31. ABISHEK M**.
