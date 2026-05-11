"""
Smart Hotel Prediction - Modern Dashboard
Premium Analytics UI with Hero Header and Metric Cards
"""

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import time

# ============================================================
# PAGE CONFIGURATION
# ============================================================
st.set_page_config(
    page_title="Smart Hotel | Analytics",
    page_icon="🏨",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ============================================================
# CUSTOM CSS - PREMIUM MODERN THEME
# ============================================================
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;800&display=swap');

    html, body, [data-testid="stAppViewContainer"] {
        font-family: 'Inter', sans-serif;
        background-color: #f8fafc;
        color: #1e293b;
    }

    /* Force all text to black for better contrast */
    [data-testid="stMarkdownContainer"] p, [data-testid="stMarkdownContainer"] span {
        color: #1e293b !important;
    }

    /* ===== HERO HEADER ===== */
    .hero-section {
        background: #0f3460; /* Deep Navy Blue */
        padding: 60px 50px;
        border-radius: 24px;
        margin-bottom: 30px;
        color: white !important;
        position: relative;
        overflow: hidden;
    }
    .hero-section h1 {
        font-size: 3.2rem;
        font-weight: 800;
        color: white !important;
        margin-bottom: 20px;
        letter-spacing: -1px;
    }
    .hero-section p {
        font-size: 1.1rem;
        color: rgba(255,255,255,0.8) !important;
        max-width: 800px;
        line-height: 1.6;
    }
    .hero-badge {
        background: rgba(255,255,255,0.1);
        padding: 5px 15px;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: 600;
        color: white !important;
        text-transform: uppercase;
        margin-bottom: 20px;
        display: inline-block;
        border: 1px solid rgba(255,255,255,0.2);
    }

    /* ===== METRIC CARDS ===== */
    .metric-row {
        display: flex;
        gap: 20px;
        margin-bottom: 30px;
        flex-wrap: wrap;
    }
    .metric-card {
        background: white;
        padding: 25px;
        border-radius: 18px;
        flex: 1;
        min-width: 200px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05), 0 2px 4px -1px rgba(0, 0, 0, 0.03);
        border: 1px solid #f1f5f9;
    }
    .metric-label {
        font-size: 0.75rem;
        font-weight: 700;
        color: #64748b !important;
        text-transform: uppercase;
        margin-bottom: 10px;
    }
    .metric-value {
        font-size: 1.4rem;
        font-weight: 800;
        color: #0f172a !important;
    }

    /* ===== INFO BOX ===== */
    .info-box {
        background: #e0f2fe;
        border-left: 5px solid #0ea5e9;
        padding: 20px;
        border-radius: 12px;
        margin-bottom: 40px;
        font-size: 0.95rem;
        color: #0369a1 !important;
    }

    /* ===== INPUT CARDS ===== */
    .input-card {
        background: white;
        border-radius: 20px;
        padding: 30px;
        margin-bottom: 25px;
        border: 1px solid #e2e8f0;
        box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.05);
    }
    .input-header {
        font-weight: 800;
        color: #0f172a !important;
        font-size: 1.1rem;
        margin-bottom: 20px;
        display: flex;
        align-items: center;
        gap: 10px;
    }

    /* ===== SPLASH SCREEN ===== */
    .splash-overlay {
        position: fixed;
        top: 0; left: 0; right: 0; bottom: 0;
        background: #0f3460;
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        z-index: 99999;
    }
    .splash-title { font-size: 3rem; color: white; font-weight: 800; }
    .splash-bar-container { width: 300px; height: 8px; background: rgba(255,255,255,0.1); border-radius: 10px; margin-top: 30px; overflow: hidden; }
    .splash-bar { height: 100%; background: #0ea5e9; animation: splashLoad 4s ease-in-out forwards; }
    @keyframes splashLoad { 0% { width: 0%; } 100% { width: 100%; } }

    /* BUTTONS */
    .stButton>button {
        background: #0ea5e9;
        color: white;
        font-weight: 700;
        padding: 12px 24px;
        border-radius: 12px;
        width: 100%;
        border: none;
        transition: all 0.3s;
    }
    .stButton>button:hover { background: #0284c7; transform: translateY(-2px); }
</style>
""", unsafe_allow_html=True)

# ============================================================
# SESSION STATE & SPLASH SCREEN
# ============================================================
if 'visited' not in st.session_state:
    splash_placeholder = st.empty()
    splash_placeholder.markdown("""
    <div class="splash-overlay">
        <div class="splash-icon" style="font-size: 6rem;">🏨</div>
        <div class="splash-title">Smart Hotel Prediction</div>
        <div class="splash-bar-container"><div class="splash-bar"></div></div>
    </div>
    """, unsafe_allow_html=True)
    time.sleep(3)
    splash_placeholder.empty()
    st.session_state.visited = True

# ============================================================
# LOAD ARTIFACTS
# ============================================================
@st.cache_resource
def load_artifacts():
    with open("model.pkl", "rb") as f:
        all_models = pickle.load(f)
    with open("scaler_v2.pkl", "rb") as f:
        feature_scaler = pickle.load(f)
    with open("model_config.pkl", "rb") as f:
        config = pickle.load(f)
    return all_models, feature_scaler, config

try:
    loaded_models, loaded_scaler, model_config = load_artifacts()
    feature_names = model_config["feature_names"]
except Exception as e:
    st.error(f"Initialization Error: {e}")
    st.stop()

# ============================================================
# HERO HEADER
# ============================================================
st.markdown("""
<div class="hero-section">
    <div class="hero-badge">Benchmark Dashboard | Prediction Console | SHAP Explainability</div>
    <h1>Hotel Cancellation Intelligence</h1>
    <p>Professional dashboard for comparing all trained models, reviewing saved holdout and 3-fold validation metrics, inspecting confusion matrices and SHAP explanations, and running live booking predictions from the saved deployment model.</p>
</div>
""", unsafe_allow_html=True)

# ============================================================
# METRIC CARDS
# ============================================================
st.markdown("""
<div class="metric-row">
    <div class="metric-card">
        <div class="metric-label">Best Benchmark</div>
        <div class="metric-value">Random Forest</div>
    </div>
    <div class="metric-card">
        <div class="metric-label">Cloud Model</div>
        <div class="metric-value">Random Forest</div>
    </div>
    <div class="metric-card">
        <div class="metric-label">Train / Test</div>
        <div class="metric-value">80% / 20%</div>
    </div>
    <div class="metric-card">
        <div class="metric-label">Cross-Validation</div>
        <div class="metric-value">5-fold</div>
    </div>
    <div class="metric-card">
        <div class="metric-label">Runtime</div>
        <div class="metric-value">Py 3.14 / XGB</div>
    </div>
</div>
""", unsafe_allow_html=True)

st.markdown("""
<div class="info-box">
    Current app mode uses the notebook-aligned reservation benchmark only. The form, evaluation tables, deployed model, and reports all come from the same reservation artifact set.
</div>
""", unsafe_allow_html=True)

# ============================================================
# MAIN APPLICATION TABS
# ============================================================
tab_overview, tab_prediction = st.tabs(["📊 Overview", "🔮 Live Prediction"])

with tab_overview:
    st.info("Select the 'Live Prediction' tab to input guest data and get a real-time status forecast.")
    # You could add charts here later

with tab_prediction:
    st.markdown("<div class='input-card'>", unsafe_allow_html=True)
    st.markdown("<div class='input-header'>📍 Booking Overview</div>", unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    with c1: lead_time = st.number_input("Days Before Arrival", 0, 500, 45)
    with c2: avg_price = st.slider("Price per Night ($)", 0, 500, 150)
    with c3: requests = st.slider("Guest Special Requests", 0, 5, 1)
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<div class='input-card'>", unsafe_allow_html=True)
    st.markdown("<div class='input-header'>👥 Guest & Stay Profile</div>", unsafe_allow_html=True)
    c4, c5, c6 = st.columns(3)
    with c4:
        stay_cat = st.selectbox("Total Stay Length", ["Day Use (0 nights)", "Short Stay (1-3)", "Week Stay (4-7)", "Two Weeks (8-14)", "Long Stay (15+)"])
        stay_map = {"Day Use (0 nights)": 0, "Short Stay (1-3)": 1, "Week Stay (4-7)": 2, "Two Weeks (8-14)": 3, "Long Stay (15+)": 4}
    with c5:
        arrival_day = st.selectbox("Day of Arrival", ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"])
        day_map = {"Monday": 0, "Tuesday": 1, "Wednesday": 2, "Thursday": 3, "Friday": 4, "Saturday": 5, "Sunday": 6}
    with c6: cancel_ratio = st.slider("Previous Cancellations", 0.0, 1.0, 0.0)
    
    c7, c8 = st.columns(2)
    with c7: parking = st.selectbox("Requires Parking?", ["No", "Yes"])
    with c8: first_time = st.selectbox("Is New Guest?", ["No", "Yes"])
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<div class='input-card'>", unsafe_allow_html=True)
    st.markdown("<div class='input-header'>📦 Service Selection</div>", unsafe_allow_html=True)
    c9, c10, c11, c12 = st.columns(4)
    with c9:
        meal_display = {"Breakfast Only": "Meal Plan 1", "Half Board": "Meal Plan 2", "Full Board": "Meal Plan 3", "None": "Not Selected"}
        meal_choice = st.selectbox("Meal Package", list(meal_display.keys()))
    with c10:
        room_display = {"Standard": "Room_Type 1", "Deluxe": "Room_Type 2", "Junior Suite": "Room_Type 3", "Executive": "Room_Type 4", "Premium": "Room_Type 5", "Presidential": "Room_Type 6", "Luxury Villa": "Room_Type 7"}
        room_choice = st.selectbox("Room Category", list(room_display.keys()))
    with c11: market_choice = st.selectbox("Market Segment", ["Aviation", "Complementary", "Corporate", "Offline", "Online"])
    with c12: guest_choice = st.selectbox("Group Size", ["1", "2", "3", "4", "5", "Group"])
    st.markdown("</div>", unsafe_allow_html=True)

    if st.button("RUN ANALYSIS"):
        def get_features():
            p_val = 1 if parking == "Yes" else 0
            f_val = 1 if first_time == "Yes" else 0
            if lead_time <= 1: lt_cat = 0
            elif lead_time <= 7: lt_cat = 1
            elif lead_time <= 30: lt_cat = 2
            elif lead_time <= 365: lt_cat = 3
            else: lt_cat = 4
            base = [p_val, lt_cat, avg_price, requests, stay_map[stay_cat], day_map[arrival_day], 6, 2017, cancel_ratio, f_val]
            encoded = []
            curr_meal = meal_display[meal_choice]
            for m in ["Meal Plan 2", "Meal Plan 3", "Not Selected"]: encoded.append(1 if curr_meal == m else 0)
            curr_room = room_display[room_choice]
            for r in ["Room_Type 2", "Room_Type 3", "Room_Type 4", "Room_Type 5", "Room_Type 6", "Room_Type 7"]: encoded.append(1 if curr_room == r else 0)
            for s in ["Complementary", "Corporate", "Offline", "Online"]: encoded.append(1 if market_choice == s else 0)
            for g in ["2", "3", "4", "5", "Group"]: encoded.append(1 if guest_choice == g else 0)
            return base + encoded

        feat_row = get_features()
        input_df = pd.DataFrame([feat_row], columns=feature_names)
        input_df["average_price"] = loaded_scaler.transform(input_df[["average_price"]].values)

        results = []
        model_info = {"logistic_regression": "Logistic", "random_forest": "Forest", "knn": "KNN", "xgboost": "XGBoost", "svm": "SVM", "decision_tree": "Tree"}

        st.write("---")
        res_cols = st.columns(len(model_info))
        i = 0
        for m_key, m_name in model_info.items():
            if m_key in loaded_models:
                mod = loaded_models[m_key]
                pred = mod.predict(input_df.values)[0]
                conf = mod.predict_proba(input_df.values)[0][1] if hasattr(mod, "predict_proba") else 0.5
                with res_cols[i]:
                    status = "STAY" if pred == 1 else "CANCEL"
                    color = "#16a34a" if status == "STAY" else "#dc2626"
                    st.markdown(f"""
                    <div style="background:white; padding:15px; border-radius:12px; border:1px solid #e2e8f0; text-align:center;">
                        <div style="font-size:0.7rem; font-weight:700; color:#64748b;">{m_name}</div>
                        <div style="font-size:1.1rem; font-weight:800; color:{color};">{status}</div>
                        <div style="font-size:0.7rem; color:#94a3b8;">{conf:.0%} Conf.</div>
                    </div>
                    """, unsafe_allow_html=True)
                results.append(pred)
                i += 1

        vote_stay = sum(1 for r in results if r == 1)
        final = "NOT CANCELED" if vote_stay >= (len(results)/2) else "CANCELED"
        st.markdown(f"""
        <div style="background:#0f3460; color:white; padding:30px; border-radius:20px; text-align:center; margin-top:30px;">
            <div style="font-size:2.2rem; font-weight:800;">{final}</div>
            <div style="font-size:1rem; opacity:0.8;">Ensemble Agreement: {vote_stay}/{len(results)} Models</div>
        </div>
        """, unsafe_allow_html=True)
