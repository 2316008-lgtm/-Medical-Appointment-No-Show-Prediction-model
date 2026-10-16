"""
Interactive Healthcare Administration Dashboard & Appointment Attendance Predictor
Project: 31. ABISHEK M - Medical Appointment No-Show Prediction
Domain: Healthcare Administration / Patient Scheduling
Algorithms: Logistic Regression, Decision Tree, Random Forest, XGBoost
"""

import os
import sys
import json
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from datetime import date, timedelta

# Append src to path
sys.path.append(os.path.join(os.path.dirname(__file__), "src"))
from src.predict import AppointmentPredictor

# Page Configuration
st.set_page_config(
    page_title="Medical Appointment No-Show Prediction | Abishek M",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom High-End CSS Styling
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', sans-serif;
    }
    
    .main-header {
        background: linear-gradient(135deg, #0F172A 0%, #1E293B 100%);
        border: 1px solid #334155;
        border-radius: 16px;
        padding: 24px 32px;
        margin-bottom: 24px;
        box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.3);
    }
    
    .badge-pill {
        display: inline-block;
        padding: 4px 14px;
        border-radius: 9999px;
        font-size: 0.78rem;
        font-weight: 700;
        letter-spacing: 0.05em;
        text-transform: uppercase;
    }
    
    .badge-blue {
        background: rgba(59, 130, 246, 0.18);
        color: #60A5FA;
        border: 1px solid rgba(59, 130, 246, 0.3);
    }
    
    .card-metric {
        background: #1E293B;
        border: 1px solid #334155;
        border-radius: 12px;
        padding: 18px 20px;
        transition: transform 0.2s ease, border-color 0.2s ease;
    }
    .card-metric:hover {
        border-color: #3B82F6;
        transform: translateY(-2px);
    }
    
    .risk-banner {
        border-radius: 14px;
        padding: 20px 24px;
        color: white;
        margin-bottom: 20px;
    }
    
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px 8px 0px 0px;
        padding: 10px 20px;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_predictor_engine():
    return AppointmentPredictor(models_dir="models")


@st.cache_data
def load_metadata_and_curves():
    with open("models/model_metadata.json") as f:
        meta = json.load(f)
    with open("models/roc_pr_curves.json") as f:
        curves = json.load(f)
    df_comp = pd.read_csv("models/model_comparison.csv", index_col=0)
    return meta, curves, df_comp


@st.cache_data
def load_eda_data():
    if os.path.exists("data/cleaned_appointments.csv"):
        return pd.read_csv("data/cleaned_appointments.csv")
    return None


predictor = load_predictor_engine()
metadata, curves_data, df_comparison = load_metadata_and_curves()
df_eda = load_eda_data()

# ----------------- SIDEBAR -----------------
with st.sidebar:
    st.markdown("### 🏥 Project Metadata")
    st.markdown("""
    **Student / Candidate:**  
    **31. ABISHEK M**  
    
    **Domain:**  
    Healthcare Administration / Patient Scheduling  
    
    **Problem Statement:**  
    Missed appointments waste valuable healthcare resources, leave provider slots idle, and delay critical patient interventions.
    
    **Dataset:**  
    110,516 Patient Appointments (Kaggle Official)
    """)
    st.divider()
    
    st.markdown("### ⚙️ Global Model Settings")
    active_model = st.selectbox(
        "Active Prediction Algorithm",
        options=["XGBoost", "Random Forest", "Decision Tree", "Logistic Regression"],
        index=0,
        help="Select which trained algorithm powers inference calculations."
    )
    
    decision_threshold = st.slider(
        "Classification Decision Threshold",
        min_value=0.10,
        max_value=0.90,
        value=0.50,
        step=0.05,
        help="Standard threshold is 0.50. Lower thresholds (e.g. 0.35 - 0.40) increase Recall to catch more no-shows for proactive phone reminders."
    )
    
    if decision_threshold < 0.45:
        st.info("💡 **Clinical Tip:** Lowering threshold prioritizes **Recall**, ensuring fewer missed appointments slip through at the expense of extra reminder calls.")
    elif decision_threshold > 0.55:
        st.warning("⚠️ **Note:** Higher threshold prioritizes **Precision**, reserving proactive calls for the most certain no-show risks.")

    st.divider()
    st.markdown("### 📋 Required Algorithms")
    st.caption("✅ Logistic Regression (Standardized)\n\n✅ Decision Tree (Interpretable)\n\n✅ Random Forest (Bagging Ensemble)\n\n✅ XGBoost (Gradient Boosting)")


# ----------------- MAIN HEADER -----------------
st.markdown("""
<div class="main-header">
    <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap;">
        <div>
            <span class="badge-pill badge-blue">Healthcare Analytics & Operations</span>
            <h1 style="color: #F8FAFC; margin: 8px 0 6px 0; font-size: 2.1rem; font-weight: 800;">
                Medical Appointment No-Show Prediction
            </h1>
            <p style="color: #94A3B8; margin: 0; font-size: 1.0rem;">
                Machine Learning Decision Support System for Reducing Idle Clinic Capacity & Preventing Delayed Patient Care
            </p>
        </div>
        <div style="text-align: right; margin-top: 10px;">
            <span style="font-size: 0.9rem; color: #CBD5E1; font-weight: 600;">Student ID: 31. ABISHEK M</span><br>
            <span style="font-size: 0.8rem; color: #64748B;">Python 3.11 • Scikit-Learn • XGBoost</span>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)


# ----------------- NAVIGATION TABS -----------------
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🩺 Patient Risk Predictor",
    "📊 Model Benchmarking",
    "📅 Batch Scheduling Triage",
    "📈 Healthcare Data Insights",
    "🧠 Feature Importance & Explainability"
])


# ==============================================================================
# TAB 1: INDIVIDUAL PATIENT PREDICTOR
# ==============================================================================
with tab1:
    st.markdown("### 🩺 Real-Time Patient Attendance Risk Scorer")
    st.caption("Input patient attributes, scheduling timeframe, and prior history to predict attendance probability and receive targeted operational recommendations.")
    
    col_input1, col_input2, col_input3 = st.columns([1.1, 1.1, 1.2])
    
    with col_input1:
        st.markdown("#### 1. Demographic & Clinical")
        age = st.slider("Patient Age (Years)", min_value=0, max_value=105, value=38, step=1)
        gender = st.selectbox("Recorded Gender", options=["F (Female)", "M (Male)"], index=0)
        gender_code = "F" if "F" in gender else "M"
        
        scholarship = st.checkbox("Bolsa Família Assistance (Scholarship)", value=False,
                                  help="Indicates enrollment in social welfare assistance program.")
        
        st.markdown("**Chronic Diagnoses & Conditions:**")
        col_cond1, col_cond2 = st.columns(2)
        with col_cond1:
            hypertension = st.checkbox("Hypertension", value=False)
            diabetes = st.checkbox("Diabetes", value=False)
        with col_cond2:
            alcoholism = st.checkbox("Alcoholism", value=False)
            has_handicap = st.checkbox("Disability / Handicap", value=False)
        handicap_level = 1 if has_handicap else 0

    with col_input2:
        st.markdown("#### 2. Booking & Appointment Logistics")
        today = date.today()
        scheduled_date = st.date_input("Booking / Scheduled Date", value=today - timedelta(days=7))
        appointment_date = st.date_input("Appointment Date", value=today)
        
        lead_days_calc = max(0, (appointment_date - scheduled_date).days)
        st.metric("Calculated Lead Time", f"{lead_days_calc} Days",
                  delta=f"{'Same-Day Booking' if lead_days_calc == 0 else 'Advance Scheduling'}",
                  delta_color="normal")
        
        scheduled_hour = st.slider("Time of Booking (Hour of Day, 24h)", min_value=7, max_value=20, value=10)
        appt_weekday = appointment_date.weekday()
        weekday_names = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
        st.caption(f"Appointment falls on: **{weekday_names[appt_weekday]}**")
        
        sms_sent = st.radio("SMS Notification Sent / Received?", options=["Yes (SMS Received)", "No (No SMS Sent)"], index=0)
        sms_val = 1 if "Yes" in sms_sent else 0

    with col_input3:
        st.markdown("#### 3. Patient History (Leak-Free)")
        is_repeat = st.checkbox("Repeat Patient (Has Prior Clinic Visits)?", value=True)
        
        if is_repeat:
            prior_appts = st.number_input("Prior Scheduled Appointments", min_value=1, max_value=50, value=3, step=1)
            prior_noshows = st.number_input("Prior Missed Appointments (No-Shows)", min_value=0, max_value=int(prior_appts), value=1, step=1)
            prev_rate = round(prior_noshows / prior_appts, 3) if prior_appts > 0 else 0.0
            st.metric("Historical No-Show Rate", f"{prev_rate:.1%}")
        else:
            prior_appts = 0
            prior_noshows = 0
            prev_rate = 0.0
            st.caption("First-time patient; zero historical records exist.")

        st.markdown("---")
        calc_button = st.button("⚡ Calculate Attendance Probability", use_container_width=True, type="primary")

    # Perform Prediction
    patient_payload = {
        "Age": age,
        "Gender": gender_code,
        "Scholarship": 1 if scholarship else 0,
        "Hypertension": 1 if hypertension else 0,
        "Diabetes": 1 if diabetes else 0,
        "Alcoholism": 1 if alcoholism else 0,
        "Handicap": handicap_level,
        "SMS_received": sms_val,
        "LeadDays": lead_days_calc,
        "Scheduled_Hour": scheduled_hour,
        "Appt_DayOfWeek": appt_weekday,
        "Prev_Appointments": prior_appts,
        "Prev_NoShows": prior_noshows,
        "Prev_NoShow_Rate": prev_rate
    }
    
    pred_res = predictor.predict_single(patient_payload, model_name=active_model, threshold=decision_threshold)
    
    st.markdown("---")
    st.markdown("### 📊 Prediction Result & Clinical Action Plan")
    
    res_col1, res_col2, res_col3 = st.columns([1.2, 1.4, 1.4])
    
    with res_col1:
        prob = pred_res["no_show_probability"]
        tier = pred_res["risk_tier"]
        color = pred_res["risk_color"]
        
        st.markdown(f"""
        <div style="background: #0F172A; border: 2px solid {color}; border-radius: 14px; padding: 22px; text-align: center;">
            <span style="font-size: 0.85rem; color: #94A3B8; font-weight: 600; text-transform: uppercase;">Predicted No-Show Risk</span>
            <div style="font-size: 3.2rem; font-weight: 800; color: {color}; margin: 8px 0;">{prob:.1%}</div>
            <div style="background: {color}22; color: {color}; border: 1px solid {color}55; padding: 6px 16px; border-radius: 999px; display: inline-block; font-weight: 700; font-size: 0.95rem;">
                {tier}
            </div>
            <p style="color: #CBD5E1; margin-top: 14px; font-size: 0.9rem;">
                <strong>Classification:</strong> {pred_res['attendance_prediction']}<br>
                <small style="color: #64748B;">Algorithm: {active_model} (Threshold: {decision_threshold:.2f})</small>
            </p>
        </div>
        """, unsafe_allow_html=True)
        
    with res_col2:
        st.markdown("#### 🔍 Primary Risk Drivers for this Booking")
        for driver in pred_res["key_drivers"]:
            st.markdown(f"- 🔸 {driver}")
            
        st.markdown("<br>", unsafe_allow_html=True)
        # Quick gauge bar
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=prob * 100,
            number={'suffix': "%", 'font': {'color': color, 'size': 26}},
            gauge={
                'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "#475569"},
                'bar': {'color': color, 'thickness': 0.28},
                'bgcolor': "#1E293B",
                'borderwidth': 0,
                'steps': [
                    {'range': [0, 25], 'color': '#064E3B'},
                    {'range': [25, 50], 'color': '#78350F'},
                    {'range': [50, 100], 'color': '#7F1D1D'}
                ],
                'threshold': {
                    'line': {'color': "#FFFFFF", 'width': 3},
                    'thickness': 0.8,
                    'value': decision_threshold * 100
                }
            }
        ))
        fig_gauge.update_layout(
            height=180,
            margin=dict(l=20, r=20, t=10, b=10),
            paper_bgcolor='rgba(0,0,0,0)',
            font={'family': 'Plus Jakarta Sans'}
        )
        st.plotly_chart(fig_gauge, use_container_width=True)

    with res_col3:
        st.markdown("#### 📋 Recommended Operational Protocol")
        for rec in pred_res["recommendations"]:
            st.markdown(f"- 🎯 {rec}")
            
        st.markdown("""
        <div style="background: #1E293B; border-left: 4px solid #3B82F6; padding: 12px 16px; border-radius: 6px; margin-top: 14px;">
            <strong style="color: #93C5FD;">Resource Impact:</strong><br>
            <span style="font-size: 0.85rem; color: #CBD5E1;">
            Targeted intervention on high-risk patients recovers up to 60-70% of potential slot loss via automated phone confirmations and waitlist backfills.
            </span>
        </div>
        """, unsafe_allow_html=True)


# ==============================================================================
# TAB 2: MODEL BENCHMARKING & COMPARISON
# ==============================================================================
with tab2:
    st.markdown("### 📊 Comprehensive Model Benchmarking & Performance Comparison")
    st.caption("All 4 required models evaluated on 22,104 hold-out test appointments (stratified 20.19% no-show rate, identical real-world test set).")
    
    # 4 Metric Cards across Top
    col_m1, col_m2, col_m3, col_m4 = st.columns(4)
    with col_m1:
        st.markdown(f"""
        <div class="card-metric">
            <span style="color: #94A3B8; font-size: 0.85rem; font-weight: 600;">LOGISTIC REGRESSION</span>
            <div style="font-size: 1.8rem; font-weight: 800; color: #60A5FA;">{df_comparison.loc['Logistic Regression', 'roc_auc']:.4f} <span style="font-size: 0.9rem; color: #94A3B8;">AUC</span></div>
            <span style="color: #34D399; font-size: 0.85rem;">Recall: <strong>{df_comparison.loc['Logistic Regression', 'recall']:.1%}</strong></span>
        </div>
        """, unsafe_allow_html=True)
    with col_m2:
        st.markdown(f"""
        <div class="card-metric">
            <span style="color: #94A3B8; font-size: 0.85rem; font-weight: 600;">DECISION TREE</span>
            <div style="font-size: 1.8rem; font-weight: 800; color: #A78BFA;">{df_comparison.loc['Decision Tree', 'roc_auc']:.4f} <span style="font-size: 0.9rem; color: #94A3B8;">AUC</span></div>
            <span style="color: #34D399; font-size: 0.85rem;">Recall: <strong>{df_comparison.loc['Decision Tree', 'recall']:.1%}</strong></span>
        </div>
        """, unsafe_allow_html=True)
    with col_m3:
        st.markdown(f"""
        <div class="card-metric">
            <span style="color: #94A3B8; font-size: 0.85rem; font-weight: 600;">RANDOM FOREST</span>
            <div style="font-size: 1.8rem; font-weight: 800; color: #F472B6;">{df_comparison.loc['Random Forest', 'roc_auc']:.4f} <span style="font-size: 0.9rem; color: #94A3B8;">AUC</span></div>
            <span style="color: #34D399; font-size: 0.85rem;">Recall: <strong>{df_comparison.loc['Random Forest', 'recall']:.1%}</strong></span>
        </div>
        """, unsafe_allow_html=True)
    with col_m4:
        st.markdown(f"""
        <div class="card-metric" style="border-color: #3B82F6;">
            <span style="color: #38BDF8; font-size: 0.85rem; font-weight: 700;">★ XGBOOST (CHAMPION)</span>
            <div style="font-size: 1.8rem; font-weight: 800; color: #38BDF8;">{df_comparison.loc['XGBoost', 'roc_auc']:.4f} <span style="font-size: 0.9rem; color: #94A3B8;">AUC</span></div>
            <span style="color: #34D399; font-size: 0.85rem;">Recall: <strong>{df_comparison.loc['XGBoost', 'recall']:.1%}</strong></span>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    
    # Detailed Comparison Table
    st.markdown("#### 📋 Clinical Metric Scorecard")
    display_df = df_comparison[['accuracy', 'recall', 'precision', 'specificity', 'f1_score', 'roc_auc', 'pr_auc']].copy()
    display_df.columns = ["Accuracy", "Recall (Sensitivity)", "Precision (PPV)", "Specificity (TNR)", "F1-Score", "ROC-AUC", "PR-AUC"]
    
    st.dataframe(
        display_df.style.format("{:.4f}").highlight_max(axis=0, color="#1E3A8A"),
        use_container_width=True
    )

    col_plot1, col_plot2 = st.columns(2)
    
    # ROC Curves
    with col_plot1:
        st.markdown("#### 📈 Receiver Operating Characteristic (ROC) Curves")
        fig_roc = go.Figure()
        
        colors_map = {
            "Logistic Regression": "#3B82F6",
            "Decision Tree": "#8B5CF6",
            "Random Forest": "#EC4899",
            "XGBoost": "#06B6D4"
        }
        
        for model_name, curve_vals in curves_data.items():
            auc_val = df_comparison.loc[model_name, 'roc_auc']
            fig_roc.add_trace(go.Scatter(
                x=curve_vals["fpr"],
                y=curve_vals["tpr"],
                mode='lines',
                name=f"{model_name} (AUC = {auc_val:.3f})",
                line=dict(color=colors_map.get(model_name, "#CBD5E1"), width=2.5)
            ))
            
        fig_roc.add_trace(go.Scatter(
            x=[0, 1], y=[0, 1],
            mode='lines',
            name="Random Guess (AUC = 0.500)",
            line=dict(color='#64748B', dash='dash')
        ))
        
        fig_roc.update_layout(
            xaxis_title="False Positive Rate (1 - Specificity)",
            yaxis_title="True Positive Rate (Recall / Sensitivity)",
            template="plotly_dark",
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(15,23,42,0.6)',
            legend=dict(yanchor="bottom", y=0.05, xanchor="right", x=0.98),
            height=420
        )
        st.plotly_chart(fig_roc, use_container_width=True)

    # PR Curves
    with col_plot2:
        st.markdown("#### 🎯 Precision-Recall (PR) Curves")
        fig_pr = go.Figure()
        
        for model_name, curve_vals in curves_data.items():
            pr_auc_val = df_comparison.loc[model_name, 'pr_auc']
            fig_pr.add_trace(go.Scatter(
                x=curve_vals["recall"],
                y=curve_vals["precision"],
                mode='lines',
                name=f"{model_name} (PR-AUC = {pr_auc_val:.3f})",
                line=dict(color=colors_map.get(model_name, "#CBD5E1"), width=2.5)
            ))
            
        baseline_rate = metadata["test_noshows"] / metadata["test_samples"]
        fig_pr.add_trace(go.Scatter(
            x=[0, 1], y=[baseline_rate, baseline_rate],
            mode='lines',
            name=f"No-Skill Baseline ({baseline_rate:.1%})",
            line=dict(color='#64748B', dash='dash')
        ))
        
        fig_pr.update_layout(
            xaxis_title="Recall (Sensitivity)",
            yaxis_title="Precision (Positive Predictive Value)",
            template="plotly_dark",
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(15,23,42,0.6)',
            legend=dict(yanchor="top", y=0.98, xanchor="right", x=0.98),
            height=420
        )
        st.plotly_chart(fig_pr, use_container_width=True)

    st.markdown("---")
    st.markdown("#### 🧩 Confusion Matrices (Test Set: N = 22,104)")
    
    col_cm1, col_cm2, col_cm3, col_cm4 = st.columns(4)
    cms = [col_cm1, col_cm2, col_cm3, col_cm4]
    
    for i, model_name in enumerate(["Logistic Regression", "Decision Tree", "Random Forest", "XGBoost"]):
        with cms[i]:
            m = metadata["metrics"][model_name]
            tp, fp, tn, fn = m['tp'], m['fp'], m['tn'], m['fn']
            
            z_data = [[tn, fp], [fn, tp]]
            fig_cm = px.imshow(
                z_data,
                text_auto=True,
                labels=dict(x="Predicted Attendance", y="Actual Attendance", color="Count"),
                x=['Attended (0)', 'No-Show (1)'],
                y=['Attended (0)', 'No-Show (1)'],
                color_continuous_scale="Blues"
            )
            fig_cm.update_layout(
                title=f"<b>{model_name}</b>",
                height=260,
                margin=dict(l=20, r=20, t=40, b=20),
                paper_bgcolor='rgba(0,0,0,0)',
                coloraxis_showscale=False
            )
            st.plotly_chart(fig_cm, use_container_width=True)
            st.caption(f"**TP (Caught):** {tp:,} | **FN (Missed):** {fn:,}")


# ==============================================================================
# TAB 3: BATCH SCHEDULING TRIAGE
# ==============================================================================
with tab3:
    st.markdown("### 📅 Batch Scheduling Triage & Prioritized Outreach Queue")
    st.caption("Upload upcoming appointment rosters to instantly triage patients, identify high-risk slots, and deploy proactive staff outreach.")
    
    col_batch_ctl1, col_batch_ctl2 = st.columns([1.5, 1])
    
    with col_batch_ctl1:
        batch_source = st.radio(
            "Batch Data Source:",
            options=["Preloaded Clinical Schedule (100 Sample Appointments)", "Upload Custom Clinic CSV File"],
            horizontal=True
        )
        
    with col_batch_ctl2:
        export_placeholder = st.empty()

    if batch_source == "Upload Custom Clinic CSV File":
        uploaded_file = st.file_uploader("Upload CSV containing appointments", type=["csv"])
        if uploaded_file is not None:
            df_batch_raw = pd.read_csv(uploaded_file)
        else:
            st.info("Awaiting file upload. You can switch to the preloaded sample schedule above to test immediately.")
            df_batch_raw = None
    else:
        sample_path = "data/sample_schedule_for_prediction.csv"
        if os.path.exists(sample_path):
            df_batch_raw = pd.read_csv(sample_path)
        else:
            st.error("Sample CSV not found. Please run the data pipeline.")
            df_batch_raw = None

    if df_batch_raw is not None:
        st.markdown(f"Loaded **{len(df_batch_raw)}** appointments for scheduling analysis.")
        
        with st.spinner("Scoring appointment roster with Machine Learning..."):
            df_scored = predictor.predict_batch(
                df_batch_raw,
                model_name=active_model,
                threshold=decision_threshold
            )
            
        # Summary metrics
        total_pts = len(df_scored)
        high_risk_count = (df_scored["Risk_Tier"] == "High Risk").sum()
        mod_risk_count = (df_scored["Risk_Tier"] == "Moderate Risk").sum()
        low_risk_count = (df_scored["Risk_Tier"] == "Low Risk").sum()
        est_noshows = df_scored["Predicted_NoShow"].sum()
        
        st.markdown("<br>", unsafe_allow_html=True)
        col_kpi1, col_kpi2, col_kpi3, col_kpi4 = st.columns(4)
        col_kpi1.metric("Total Appointments", f"{total_pts}")
        col_kpi2.metric("Predicted No-Shows", f"{est_noshows}", f"{(est_noshows/total_pts):.1%}")
        col_kpi3.metric("🚨 High Risk (Urgent Calls)", f"{high_risk_count}", f"{(high_risk_count/total_pts):.1%}")
        col_kpi4.metric("📱 Moderate Risk (Automated SMS)", f"{mod_risk_count}", f"{(mod_risk_count/total_pts):.1%}")
        
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("#### 📋 Prioritized Patient Outreach Queue (Ranked by Risk)")
        
        display_cols = [
            "AppointmentID", "PatientId", "Age", "Gender", "LeadDays",
            "Prev_NoShows", "SMS_received", "NoShow_Probability",
            "Risk_Tier", "Primary_Recommendation"
        ]
        available_cols = [c for c in display_cols if c in df_scored.columns]
        
        def highlight_tier(val):
            if val == "High Risk":
                return "background-color: rgba(239, 68, 68, 0.25); color: #FCA5A5; font-weight: bold;"
            elif val == "Moderate Risk":
                return "background-color: rgba(245, 158, 11, 0.25); color: #FCD34D;"
            else:
                return "background-color: rgba(16, 185, 129, 0.25); color: #6EE7B7;"
                
        st.dataframe(
            df_scored[available_cols].style
            .format({"NoShow_Probability": "{:.1%}", "PatientId": "{:.0f}"})
            .applymap(highlight_tier, subset=["Risk_Tier"] if "Risk_Tier" in available_cols else []),
            use_container_width=True,
            height=380
        )
        
        # Download button
        csv_data = df_scored.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Download Prioritized Outreach CSV",
            data=csv_data,
            file_name="prioritized_appointment_outreach.csv",
            mime="text/csv",
            type="primary"
        )


# ==============================================================================
# TAB 4: HEALTHCARE DATA INSIGHTS & EDA
# ==============================================================================
with tab4:
    st.markdown("### 📈 Exploratory Healthcare Data Insights & Epidemiological Trends")
    st.caption("Deep-dive analysis of appointment attendance patterns across 110,516 clinical encounters.")
    
    if df_eda is not None:
        col_eda1, col_eda2 = st.columns(2)
        
        # 1. Lead Time vs No Show Rate
        with col_eda1:
            st.markdown("#### ⏳ Lead Time (Booking-to-Appointment Days) vs No-Show Rate")
            # Group by LeadDays up to 35 days
            lead_summary = df_eda[df_eda['LeadDays'] <= 35].groupby('LeadDays')['NoShow_Binary'].agg(['count', 'mean']).reset_index()
            
            fig_lead = px.area(
                lead_summary,
                x='LeadDays',
                y='mean',
                labels={'LeadDays': 'Booking Lead Time (Days)', 'mean': 'No-Show Rate'},
                title="Impact of Scheduling Lead Time on Missed Appointments"
            )
            fig_lead.update_traces(line_color="#38BDF8", fillcolor="rgba(56, 189, 248, 0.2)")
            fig_lead.add_hline(y=0.2019, line_dash="dash", line_color="#EF4444", annotation_text="Baseline No-Show (20.2%)")
            fig_lead.update_layout(
                template="plotly_dark",
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(15,23,42,0.6)',
                yaxis_tickformat='.0%',
                height=340
            )
            st.plotly_chart(fig_lead, use_container_width=True)
            st.caption("💡 **Finding:** Same-day bookings have only a **4.6%** no-show rate. When bookings exceed 14 days, the no-show rate jumps above **32%**.")

        # 2. Age Distribution vs Attendance
        with col_eda2:
            st.markdown("#### 👥 Patient Age vs No-Show Propensity")
            age_summary = df_eda.groupby('Age')['NoShow_Binary'].agg(['count', 'mean']).reset_index()
            age_summary = age_summary[age_summary['count'] > 100]
            
            fig_age = px.line(
                age_summary,
                x='Age',
                y='mean',
                labels={'Age': 'Patient Age (Years)', 'mean': 'No-Show Rate'},
                title="No-Show Probability Across Patient Life Stages"
            )
            fig_age.update_traces(line_color="#F43F5E", line_width=2.5)
            fig_age.add_hline(y=0.2019, line_dash="dash", line_color="#94A3B8")
            fig_age.update_layout(
                template="plotly_dark",
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(15,23,42,0.6)',
                yaxis_tickformat='.0%',
                height=340
            )
            st.plotly_chart(fig_age, use_container_width=True)
            st.caption("💡 **Finding:** Peak no-show rates occur in youth/young adults (ages 14-25, ~25-26%). Senior citizens (>60) exhibit highest attendance compliance (~15%).")

        col_eda3, col_eda4 = st.columns(2)
        
        # 3. Day of Week Attendance
        with col_eda3:
            st.markdown("#### 📅 Attendance by Day of the Week")
            dow_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday"]
            dow_summary = df_eda.groupby('Appt_DayName')['NoShow_Binary'].agg(['count', 'mean']).reindex(dow_order).reset_index()
            
            fig_dow = px.bar(
                dow_summary,
                x='Appt_DayName',
                y='mean',
                color='mean',
                color_continuous_scale="Viridis",
                labels={'Appt_DayName': 'Appointment Day', 'mean': 'No-Show Rate'},
                title="No-Show Rate Across Appointment Weekdays"
            )
            fig_dow.update_layout(
                template="plotly_dark",
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(15,23,42,0.6)',
                yaxis_tickformat='.0%',
                height=320,
                coloraxis_showscale=False
            )
            st.plotly_chart(fig_dow, use_container_width=True)
            st.caption("💡 **Finding:** Friday and Monday show slightly higher cancellation/no-show rates compared to mid-week appointments.")

        # 4. Chronic Conditions & Scholarship Impact
        with col_eda4:
            st.markdown("#### 🩺 Clinical Diagnosis & Assistance Program Breakdown")
            cond_metrics = {
                "Hypertension": df_eda[df_eda['Hypertension'] == 1]['NoShow_Binary'].mean(),
                "Diabetes": df_eda[df_eda['Diabetes'] == 1]['NoShow_Binary'].mean(),
                "Alcoholism": df_eda[df_eda['Alcoholism'] == 1]['NoShow_Binary'].mean(),
                "Disability (Handicap)": df_eda[df_eda['Has_Handicap'] == 1]['NoShow_Binary'].mean(),
                "Social Scholarship (Bolsa)": df_eda[df_eda['Scholarship'] == 1]['NoShow_Binary'].mean(),
                "Baseline (Overall)": 0.2019
            }
            df_cond = pd.DataFrame(list(cond_metrics.items()), columns=["Category", "NoShow_Rate"]).sort_values("NoShow_Rate")
            
            fig_cond = px.bar(
                df_cond,
                x='NoShow_Rate',
                y='Category',
                orientation='h',
                color='NoShow_Rate',
                color_continuous_scale="Reds",
                title="No-Show Rate by Health Condition & Assistance"
            )
            fig_cond.update_layout(
                template="plotly_dark",
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(15,23,42,0.6)',
                xaxis_tickformat='.0%',
                height=320,
                coloraxis_showscale=False
            )
            st.plotly_chart(fig_cond, use_container_width=True)
            st.caption("💡 **Finding:** Patients with chronic diseases (Hypertension, Diabetes) attend more reliably, whereas socioeconomic hardship (Scholarship) correlates with higher missed appointments.")
    else:
        st.info("Cleaned dataset not found. Run pipeline to view EDA.")


# ==============================================================================
# TAB 5: FEATURE IMPORTANCE & EXPLAINABILITY
# ==============================================================================
with tab5:
    st.markdown("### 🧠 Model Explainability & Feature Importance Rankings")
    st.caption("Inspect the exact driving signals discovered by each machine learning algorithm.")
    
    col_feat1, col_feat2 = st.columns(2)
    
    with col_feat1:
        st.markdown(f"#### 🌲 Feature Importance: **{active_model}**")
        raw_imps = metadata["feature_importances"].get(active_model, {})
        
        df_imp = pd.DataFrame(list(raw_imps.items()), columns=["Feature", "Importance"])
        if active_model == "Logistic Regression":
            df_imp["AbsImportance"] = df_imp["Importance"].abs()
            df_imp = df_imp.sort_values("AbsImportance", ascending=True).tail(12)
        else:
            df_imp = df_imp.sort_values("Importance", ascending=True).tail(12)
            
        fig_imp = px.bar(
            df_imp,
            x="Importance",
            y="Feature",
            orientation="h",
            color="Importance",
            color_continuous_scale="Teal",
            title=f"Top Predictive Drivers in {active_model}"
        )
        fig_imp.update_layout(
            template="plotly_dark",
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(15,23,42,0.6)',
            height=420,
            coloraxis_showscale=False
        )
        st.plotly_chart(fig_imp, use_container_width=True)

    with col_feat2:
        st.markdown("#### 🔬 Healthcare Operational Interpretation")
        st.markdown("""
        1. **Same-Day Appointment Flag (`SameDay_Appt`):**
           - The single strongest negative predictor of no-shows across all models.
           - Patients booking on the same day attend at a **~95.4%** rate.
           - *Clinic Recommendation:* Reserve a portion of provider capacity for same-day acute booking to maximize schedule density.

        2. **Booking Lead Time (`LeadDays`):**
           - Strongest continuous predictor. Longer lead times allow forgetfulness, scheduling conflicts, or symptom resolution.
           - *Clinic Recommendation:* Multi-touch remindercadence (SMS at T-7 days, WhatsApp at T-2 days, telephone call at T-24h).

        3. **Historical No-Show Profile (`Prev_NoShows`, `Prev_NoShow_Rate`):**
           - Formulated with zero data leakage (only historical prior appointments count).
           - Strong habitual behavioral marker: past no-shows strongly predict future no-shows.
           - *Clinic Recommendation:* Flag chronic no-show patients for coordinator check-in and barrier assessment.

        4. **Patient Age (`Age`):**
           - Younger cohorts exhibit higher mobility and schedule changes.
           - Elderly patients adhere strictly to medical appointments.

        5. **SMS Notification (`SMS_received`):**
           - Highly influential in tree splits for multi-day bookings.
        """)

    st.markdown("---")
    st.markdown("### 📚 Summary of Algorithms Built")
    
    col_alg1, col_alg2 = st.columns(2)
    with col_alg1:
        st.markdown("""
        **1. Logistic Regression (L2 Regularized with Balanced Class Weights)**
        - Baseline linear probabilistic classifier.
        - Scales features with `StandardScaler` to measure direct log-odds shifts.
        - Provides highest raw recall (**84.85%**) due to balanced penalty.
        
        **2. Decision Tree Classifier**
        - Non-linear tree partitioning with Gini impurity splitting.
        - High clinical interpretability with direct rule extractions (e.g. `LeadDays > 3`).
        """)
    with col_alg2:
        st.markdown("""
        **3. Random Forest (Bagging Ensemble)**
        - Ensembles 150 bootstrapped trees to mitigate variance and overfitting.
        - Achieves strong generalization (**0.7433 ROC-AUC**).
        
        **4. XGBoost (Extreme Gradient Boosting)**
        - Champion algorithm with highest ROC-AUC (**0.7454**) and PR-AUC (**0.4079**).
        - Employs weighted gradient descent with `scale_pos_weight=3.95` to balance clinic priority.
        """)


# ----------------- FOOTER -----------------
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #64748B; font-size: 0.85rem; padding-bottom: 24px;">
    <strong>31. ABISHEK M — Medical Appointment No-Show Prediction</strong> • Healthcare Administration & Patient Scheduling Machine Learning System<br>
    Built with Python, Scikit-Learn, XGBoost, and Streamlit.
</div>
""", unsafe_allow_html=True)
