"""
Hotel Booking Cancellation Predictor - Streamlit Application
Premium UI with modern design, animated cards, and optimized input features.
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
    page_title="Hotel Analytics | Cancellation Predictor",
    page_icon="🏨",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ============================================================
# CUSTOM CSS - PREMIUM THEME
# ============================================================
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;800&display=swap');

    html, body, [data-testid="stAppViewContainer"] {
        font-family: 'Inter', sans-serif;
        background-color: #0f172a;
        color: #f8fafc;
    }

    /* ===== GLOBAL STYLES ===== */
    .main-header {
        text-align: center;
        padding: 40px 0 20px 0;
    }
    .main-header h1 {
        font-size: 3rem;
        background: linear-gradient(135deg, #38bdf8 0%, #818cf8 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 800;
        margin-bottom: 10px;
        letter-spacing: -1px;
    }
    .main-header p {
        color: #94a3b8;
        font-size: 1.2rem;
    }

    /* ===== START SCREEN OVERLAY ===== */
    .splash-overlay {
        position: fixed;
        top: 0; left: 0; right: 0; bottom: 0;
        background: #020617;
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        z-index: 99999;
    }
    .splash-icon {
        font-size: 6rem;
        margin-bottom: 20px;
        filter: drop-shadow(0 0 20px rgba(56, 189, 248, 0.5));
        animation: float 3s ease-in-out infinite;
    }
    @keyframes float {
        0%, 100% { transform: translateY(0px); }
        50% { transform: translateY(-20px); }
    }
    .splash-title {
        font-size: 3rem;
        color: #fff;
        font-weight: 800;
        letter-spacing: 4px;
        background: linear-gradient(90deg, #38bdf8, #818cf8);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .splash-bar-container {
        width: 300px;
        height: 6px;
        background: rgba(255,255,255,0.05);
        border-radius: 10px;
        margin-top: 40px;
        overflow: hidden;
    }
    .splash-bar {
        height: 100%;
        background: linear-gradient(90deg, #38bdf8, #818cf8);
        animation: splashLoad 4.5s ease-in-out forwards;
    }
    @keyframes splashLoad { 0% { width: 0%; } 100% { width: 100%; } }

    /* ===== ANIMATED CROSSING LINES ===== */
    .crossing-lines-bg {
        position: relative;
        overflow: hidden;
        padding: 20px 0;
    }
    .crossing-lines-bg::before, .crossing-lines-bg::after {
        content: '';
        position: absolute;
        top: 50%; left: 50%;
        width: 200%; height: 1px;
        background: linear-gradient(90deg, transparent, rgba(56, 189, 248, 0.2), transparent);
        z-index: -1;
    }
    .crossing-lines-bg::before { transform: translate(-50%, -50%) rotate(-15deg); animation: moveLine 8s linear infinite; }
    .crossing-lines-bg::after { transform: translate(-50%, -50%) rotate(15deg); animation: moveLine 8s linear infinite reverse; }
    @keyframes moveLine { 0% { transform: translate(-50%, -50%) rotate(-15deg) translateX(-20%); } 100% { transform: translate(-50%, -50%) rotate(-15deg) translateX(20%); } }

    /* ===== MODEL CARDS ===== */
    .model-card {
        background: rgba(30, 41, 59, 0.7);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 20px;
        padding: 24px;
        margin: 12px 0;
        transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1);
        backdrop-filter: blur(12px);
    }
    .model-card:hover {
        transform: translateY(-5px);
        border-color: #38bdf8;
        box-shadow: 0 20px 40px rgba(0,0,0,0.4), 0 0 20px rgba(56, 189, 248, 0.2);
    }
    .card-title { font-weight: 800; font-size: 1.1rem; color: #fff; margin: 10px 0 5px 0; }
    .card-metric { font-size: 0.95rem; color: #94a3b8; }
    .card-accent { color: #38bdf8; font-weight: 700; }

    /* ===== INPUT SECTION CARDS ===== */
    .input-group-card {
        background: #1e293b;
        border-radius: 16px;
        padding: 25px;
        border-left: 4px solid #38bdf8;
        margin-bottom: 20px;
    }
    .input-label {
        font-weight: 700;
        color: #38bdf8;
        margin-bottom: 15px;
        text-transform: uppercase;
        letter-spacing: 1px;
        font-size: 0.85rem;
    }

    /* ===== RESULT BOX ===== */
    .result-box {
        background: linear-gradient(135deg, #1e40af 0%, #1e3a8a 100%);
        border-radius: 24px;
        padding: 40px;
        text-align: center;
        margin-top: 30px;
        border: 1px solid rgba(56, 189, 248, 0.3);
        box-shadow: 0 30px 60px rgba(0,0,0,0.5);
    }
    .result-icon { font-size: 4rem; margin-bottom: 10px; }
    .result-text { font-size: 2.5rem; font-weight: 800; color: #fff; }
    .result-sub { font-size: 1.1rem; color: #94a3b8; margin-top: 10px; }

    /* STREAMLIT OVERRIDES */
    .stButton>button {
        background: linear-gradient(90deg, #38bdf8, #818cf8);
        color: white;
        border: none;
        padding: 15px 30px;
        border-radius: 12px;
        font-weight: 800;
        transition: all 0.3s;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    .stButton>button:hover {
        transform: scale(1.02);
        box-shadow: 0 0 20px rgba(56, 189, 248, 0.4);
    }
    div[data-testid="stExpander"] { background: transparent; border: none; }
</style>
""", unsafe_allow_html=True)

# ============================================================
# SESSION STATE & SPLASH SCREEN
# ============================================================
if 'visited' not in st.session_state:
    splash_placeholder = st.empty()
    splash_placeholder.markdown("""
    <div class="splash-overlay">
        <div class="splash-icon">🏨</div>
        <div class="splash-title">KHALED ANALYTICS</div>
        <div class="splash-bar-container"><div class="splash-bar"></div></div>
    </div>
    """, unsafe_allow_html=True)
    time.sleep(5)
    splash_placeholder.empty()
    st.session_state.visited = True

# ============================================================
# LOAD ARTIFACTS
# ============================================================
@st.cache_resource
def load_artifacts():
    with open("model.pkl", "rb") as f:
        all_models = pickle.load(f)
    with open("scaler.pkl", "rb") as f:
        feature_scaler = pickle.load(f)
    with open("model_config.pkl", "rb") as f:
        config = pickle.load(f)
    return all_models, feature_scaler, config

try:
    loaded_models, loaded_scaler, model_config = load_artifacts()
    feature_names = model_config["feature_names"]
except Exception as e:
    st.error(f"Error loading models: {e}")
    st.stop()

# ============================================================
# UI HEADER
# ============================================================
st.markdown("""
<div class="main-header">
    <h1>Hotel Reservation Predictor</h1>
    <p>Advanced Machine Learning Suite for Booking Status Forecasting</p>
</div>
""", unsafe_allow_html=True)

# ============================================================
# INPUT SECTION
# ============================================================
with st.container():
    st.markdown("<div class='input-group-card'>", unsafe_allow_html=True)
    st.markdown("<div class='input-label'>🔹 Core Stay Details</div>", unsafe_allow_html=True)
    
    c1, c2, c3 = st.columns(3)
    with c1:
        lead_time = st.number_input("Advance Booking Days", 0, 500, 30)
    with c2:
        avg_price = st.slider("Daily Rate ($)", 0, 500, 120)
    with c3:
        requests = st.slider("Special Requests", 0, 5, 1)
        
    c4, c5, c6 = st.columns(3)
    with c4:
        stay_cat = st.selectbox("Stay Duration", ["Day Use (0 nights)", "Short Stay (1-3)", "Week Stay (4-7)", "Two Weeks (8-14)", "Long Stay (15+)"])
        stay_map = {"Day Use (0 nights)": 0, "Short Stay (1-3)": 1, "Week Stay (4-7)": 2, "Two Weeks (8-14)": 3, "Long Stay (15+)": 4}
    with c5:
        arrival_day = st.selectbox("Arrival Day", ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"])
        day_map = {"Monday": 0, "Tuesday": 1, "Wednesday": 2, "Thursday": 3, "Friday": 4, "Saturday": 5, "Sunday": 6}
    with c6:
        cancel_ratio = st.slider("Guest Cancel History", 0.0, 1.0, 0.0)
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<div class='input-group-card'>", unsafe_allow_html=True)
    st.markdown("<div class='input-label'>🔹 Reservation Timing & Profile</div>", unsafe_allow_html=True)
    c7, c8, c9, c10 = st.columns(4)
    with c7:
        res_month = st.selectbox("Arrival Month", list(range(1, 13)), index=5)
    with c8:
        res_year = st.selectbox("Arrival Year", [2015, 2016, 2017, 2018], index=2)
    with c9:
        parking = st.radio("Car Parking?", ["No", "Yes"], horizontal=True)
    with c10:
        first_time = st.radio("New Guest?", ["No", "Yes"], horizontal=True)
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<div class='input-group-card'>", unsafe_allow_html=True)
    st.markdown("<div class='input-label'>🔹 Service & Market Preferences</div>", unsafe_allow_html=True)
    c11, c12, c13, c14 = st.columns(4)
    with c11:
        meal_display = {"Breakfast Only": "Meal Plan 1", "Half Board (BF+D)": "Meal Plan 2", "Full Board (All)": "Meal Plan 3", "No Meal Plan": "Not Selected"}
        meal_choice = st.selectbox("Meal Package", list(meal_display.keys()))
    with c12:
        room_display = {"Standard": "Room_Type 1", "Deluxe": "Room_Type 2", "Junior Suite": "Room_Type 3", "Executive": "Room_Type 4", "Premium": "Room_Type 5", "Presidential": "Room_Type 6", "Luxury Villa": "Room_Type 7"}
        room_choice = st.selectbox("Room Category", list(room_display.keys()))
    with c13:
        market_choice = st.selectbox("Booking Source", ["Aviation", "Complementary", "Corporate", "Offline", "Online"])
    with c14:
        guest_choice = st.selectbox("Guest Group Size", ["1", "2", "3", "4", "5", "Group"])
    st.markdown("</div>", unsafe_allow_html=True)

# ============================================================
# FEATURE ENGINEERING (Fixing the 28 Feature Bug)
# ============================================================
def get_features():
    # 1. Binary/Numeric Basics (8 features)
    p_val = 1 if parking == "Yes" else 0
    f_val = 1 if first_time == "Yes" else 0
    
    # Lead time category
    if lead_time <= 1: lt_cat = 0
    elif lead_time <= 7: lt_cat = 1
    elif lead_time <= 30: lt_cat = 2
    elif lead_time <= 365: lt_cat = 3
    else: lt_cat = 4

    base_features = [
        p_val, lt_cat, avg_price, requests,
        stay_map[stay_cat], day_map[arrival_day], 
        res_month, res_year, cancel_ratio, f_val
    ]

    # 2. One-Hot Encoded (Total 18 additional features to reach 28)
    encoded = []
    
    # Meal (3 features, drop_first 'Meal Plan 1')
    current_meal = meal_display[meal_choice]
    for m in ["Meal Plan 2", "Meal Plan 3", "Not Selected"]:
        encoded.append(1 if current_meal == m else 0)
        
    # Room (6 features, drop_first 'Room_Type 1')
    current_room = room_display[room_choice]
    for r in ["Room_Type 2", "Room_Type 3", "Room_Type 4", "Room_Type 5", "Room_Type 6", "Room_Type 7"]:
        encoded.append(1 if current_room == r else 0)
        
    # Market (4 features, drop_first 'Aviation')
    for s in ["Complementary", "Corporate", "Offline", "Online"]:
        encoded.append(1 if market_choice == s else 0)
        
    # Guests (5 features, drop_first '1')
    for g in ["2", "3", "4", "5", "Group"]:
        encoded.append(1 if guest_choice == g else 0)
        
    return base_features + encoded

# ============================================================
# PREDICTION ENGINE
# ============================================================
if st.button("🚀 EXECUTE PREDICTION", use_container_width=True):
    with st.spinner("Analyzing data patterns..."):
        feat_row = get_features()
        
        # DEBUG: Verification of 28 features
        if len(feat_row) != 28:
            st.error(f"Feature mismatch: Expected 28, got {len(feat_row)}")
            st.stop()
            
        input_df = pd.DataFrame([feat_row], columns=feature_names)
        input_df["average_price"] = loaded_scaler.transform(input_df[["average_price"]])

        results = []
        model_info = {
            "logistic_regression": "📈 Logistic Regression",
            "random_forest": "🌲 Random Forest",
            "knn": "📍 K-Nearest Neighbors",
            "xgboost": "⚡ XGBoost",
            "svm": "🔲 SVM (RBF)",
            "decision_tree": "🌳 Decision Tree"
        }

        for m_key, m_name in model_info.items():
            if m_key in loaded_models:
                mod = loaded_models[m_key]
                pred = mod.predict(input_df)[0]
                conf = mod.predict_proba(input_df)[0][1] if hasattr(mod, "predict_proba") else 0.5
                results.append({"Model": m_name, "Pred": "Stay" if pred == 1 else "Cancel", "Conf": conf})

        # Display Results
        st.markdown("<div class='crossing-lines-bg'>", unsafe_allow_html=True)
        cols = st.columns(len(results))
        for i, res in enumerate(results):
            with cols[i]:
                st.markdown(f"""
                <div class="model-card">
                    <div class="card-title">{res['Model']}</div>
                    <div class="card-accent" style="color: {'#22c55e' if res['Pred'] == 'Stay' else '#ef4444'}">{res['Pred']}</div>
                    <div class="card-metric">Confidence: {res['Conf']:.1%}</div>
                </div>
                """, unsafe_allow_html=True)

        # Ensemble Final
        vote_stay = sum(1 for r in results if r['Pred'] == 'Stay')
        final = "NOT CANCELED" if vote_stay >= (len(results)/2) else "CANCELED"
        st.markdown(f"""
        <div class="result-box">
            <div class="result-icon">{'✅' if final == 'NOT CANCELED' else '❌'}</div>
            <div class="result-text">{final}</div>
            <div class="result-sub">Unbiased Ensemble Agreement: {vote_stay}/{len(results)} Models</div>
        </div>
        """, unsafe_allow_html=True)
