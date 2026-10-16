# 🎤 Project Presentation Guide & Viva Walkthrough

**Project Title:** Medical Appointment No-Show Prediction  
**Student / Candidate:** 31. ABISHEK M  
**Domain:** Healthcare Administration / Patient Scheduling  
**Live Application:** Deployed on GitHub Pages & Streamlit  

---

## ⏱️ 2-Minute Spoken Pitch (Word-for-Word Presentation Script)

> *"Good morning/afternoon, Respected Evaluators,*  
>  
> *I am **Abishek M**, and today I am presenting **Medical Appointment No-Show Prediction**, a machine learning decision support system developed for healthcare administration and patient scheduling.*  
>  
> *In outpatient clinics, missed appointments waste billions of dollars annually. When a patient misses an appointment without canceling, high-cost physician time and diagnostic rooms sit idle, while patients on waiting lists face delayed care. Our objective is to predict appointment attendance in advance so clinics can deploy targeted interventions—such as automated reminders or strategic waitlist backfills.*  
>  
> *We analyzed **110,516 authentic patient appointments**. To ensure clinical integrity, we strictly enforced **zero data leakage**: patient IDs and appointment IDs were excluded as direct predictors, and all historical features—such as prior no-shows and booking lead times—were calculated strictly using information available **prior** to the scheduled visit.*  
>  
> *We developed and benchmarked four core algorithms:*  
> 1. *Logistic Regression with standard scaling and balanced class weights,*  
> 2. *Decision Tree Classifier,*  
> 3. *Random Forest Ensemble,*  
> 4. *XGBoost with weighted gradient boosting.*  
>  
> *On our 22,104 hold-out test set, **XGBoost achieved champion performance with an ROC-AUC of 0.7454 and 80.14% Recall**, successfully identifying 4 out of every 5 patients at risk of missing their appointment.*  
>  
> *Finally, we deployed a **100% serverless, client-side web application on GitHub Pages**, where all machine learning weights run directly in the browser in under 5 milliseconds with zero backend required.*  
>  
> *Thank you, and I am now ready for the live demonstration and questions."*

---

## 📑 Slide-by-Slide Presentation Structure

### Slide 1: Problem Statement & Clinical Objective
- **Problem:** Approximately 20.2% of outpatient appointments are missed without notice.
- **Impact:** Each idle slot costs ~$150–$250 in wasted provider and facility time, and delays care for other patients.
- **Objective:** Build a machine learning classification engine that stratifies scheduled patients into **Low, Moderate, and High Risk** tiers to drive targeted interventions.

### Slide 2: Dataset Schema & Data Hygiene
- **Dataset:** 110,516 records from municipal clinics in Vitória, Brazil.
- **Identifier Hygiene:** `PatientId` and `AppointmentID` strictly **excluded as direct predictors**.
- **Data Cleansing:** Errant records with negative ages and negative lead times were removed.
- **Target Encoding:** Binary attendance outcome: `1 = No-Show (Missed)`, `0 = Show (Attended)`.

### Slide 3: Zero Data Leakage Feature Engineering
*Why this design is clinically rigorous:*
1. **Lead Time (`LeadDays`):** Calendar days between booking date and scheduled visit date ($0$ to $179$ days).
2. **Same-Day Flag (`SameDay_Appt`):** Highlights acute same-day visits (attendance $> 95\%$).
3. **Chronological Prior History (`Prev_NoShows`, `Prev_NoShow_Rate`):**
   - Computed per patient using chronological expansion shifted by one visit (`shift(1).cumsum()`).
   - Ensures an appointment's prediction **only uses prior completed visits**, completely eliminating data leakage.
4. **Comorbidity Index (`Num_Conditions`):** Composite indicator summing Hypertension, Diabetes, Alcoholism, and Physical Disability.

---

## 📊 Slide 4: Model Benchmarking (Test Set: $N = 22,104$)

Evaluated on a 20% hold-out test set preserving the 20.19% baseline no-show rate:

| Algorithm | ROC-AUC | Recall (Sensitivity) | Precision | Specificity | F1-Score | PR-AUC |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Logistic Regression** | 0.7305 | **84.85%** | 30.30% | 50.63% | 0.4465 | 0.3758 |
| **Decision Tree** | 0.7331 | 82.05% | 30.63% | 53.00% | 0.4461 | 0.3731 |
| **Random Forest** | 0.7433 | 82.03% | 31.12% | 54.09% | 0.4512 | 0.4053 |
| **XGBoost (Champion)** | **0.7454** | **80.14%** | **31.53%** | **55.99%** | **0.4526** | **0.4079** |

### Confusion Matrix Breakdown (Test Set)
- **Logistic Regression:** $\text{TP} = 3,786$ caught | $\text{FN} = 676$ missed | $\text{TN} = 8,932$ | $\text{FP} = 8,710$
- **XGBoost:** $\text{TP} = 3,576$ caught | $\text{FN} = 886$ missed | $\text{TN} = 9,878$ | $\text{FP} = 7,764$

> **Clinical Decision Rationale:** In healthcare scheduling, **Recall is prioritized over raw accuracy**. A false negative (missing a no-show) leaves a $200 doctor slot empty, whereas a false positive merely triggers an inexpensive automated SMS or confirmation call.

---

## 🔍 Slide 5: Top 3 Predictive Drivers Discovered

1. **Same-Day Appointment Flag (`SameDay_Appt`):**
   - The single strongest negative predictor of missing a visit ($\beta = -1.037$ in Logistic Regression; top feature gain in XGBoost).
   - Same-day bookings have a **95.4% attendance rate** (only 4.6% no-show rate).
2. **Booking Lead Time (`LeadDays`):**
   - Appointments scheduled $> 14$ days in advance experience no-show rates climbing above **32%**.
3. **Prior No-Show History (`Prev_NoShows`, `Prev_NoShow_Rate`):**
   - Strong behavioral pattern: past no-shows are the strongest positive indicator of future missed visits.

---

## 🖥️ Slide 6: Live Deployment & Architecture

1. **Interactive Streamlit Web Dashboard:**
   - Real-time patient risk scorer, batch appointment triage queue, and downloadable outreach CSV.
2. **100% Serverless GitHub Pages Deployment:**
   - Trained model coefficients, scaler parameters, and decision tree logic compiled into client-side JavaScript (`models_engine.js`).
   - Runs directly in any web browser in $< 5\text{ ms}$ with zero hosting costs.

---

## 🎯 Viva & Evaluator Q&A Preparation

#### Q1: "Why did you choose XGBoost as the champion model?"
> **Answer:** *"XGBoost achieved the highest ROC-AUC (0.7454) and Precision-Recall AUC (0.4079). It effectively captures non-linear interactions between extended lead times and prior patient no-shows while maintaining an 80.14% recall."*

#### Q2: "How did you prevent data leakage when engineering prior patient history?"
> **Answer:** *"If we used a patient's total no-shows across all records, we would leak future outcomes into past appointments. Instead, we sorted all appointments chronologically and applied a shifted cumulative expansion (`shift(1).cumsum()`). That way, every visit is scored strictly using information available prior to that specific appointment."*

#### Q3: "Why is accuracy around 60% instead of 90%?"
> **Answer:** *"The dataset has an 80/20 class imbalance. A naive model that always predicts 'Show' would achieve 80% accuracy, but have 0% Recall—catching zero no-shows! We applied balanced class weighting (`scale_pos_weight = 3.95`), which deliberately prioritizes **Recall (80%–85%)** so the clinic actually catches the patients who need reminders."*

#### Q4: "How does the frontend run on GitHub Pages without a Python backend?"
> **Answer:** *"GitHub Pages only serves static files. We compiled our trained model weights, normalization parameters, and decision tree structures into a pure client-side JavaScript engine (`models_engine.js`). All inference is calculated locally in the browser in less than 5 milliseconds."*
