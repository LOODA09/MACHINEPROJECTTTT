"""
Hotel Booking Cancellation Predictor - Streamlit Application
Baby Blue Theme | Optimized Input | Fixed Scaling Bug
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
    page_title="Hotel Analytics | Predictor",
    page_icon="🏨",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ============================================================
# CUSTOM CSS - BABY BLUE THEME
# ============================================================
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Segoe+UI:wght@400;600;800&display=swap');

    html, body, [data-testid="stAppViewContainer"] {
        font-family: 'Segoe UI', sans-serif;
        background-color: #f0f9ff; /* Very Light Baby Blue */
        color: #000000 !important;
    }

    /* Force all text to black */
    .stMarkdown, .stText, .stHeader, h1, h2, h3, p, span, label {
        color: #000000 !important;
    }

    /* ===== HERO HEADER SECTION ===== */
    .hero-container {
        display: flex;
        justify-content: center;
        align-items: center;
        gap: 50px;
        background: linear-gradient(135deg, #bae6fd 0%, #e0f2fe 100%);
        padding: 60px 40px;
        border-radius: 0 0 50px 50px;
        margin-bottom: 40px;
        position: relative;
        overflow: hidden;
        box-shadow: 0 10px 30px rgba(0,0,0,0.05);
    }

    .hero-col {
        position: relative;
        text-align: center;
        z-index: 2;
    }

    .hero-title {
        font-size: 3.5rem;
        font-weight: 900;
        color: #0369a1 !important;
        margin: 0;
        letter-spacing: -2px;
    }

    /* Revolving Rings Animation */
    .revolving-ring {
        position: absolute;
        top: 50%;
        left: 50%;
        transform: translate(-50%, -50%);
        border: 2px dashed rgba(14, 165, 233, 0.3);
        border-radius: 50%;
        z-index: 1;
    }
    .ring-1 { width: 300px; height: 300px; animation: rotate 10s linear infinite; }
    .ring-2 { width: 250px; height: 250px; animation: rotate 15s linear infinite reverse; }

    @keyframes rotate { from { transform: translate(-50%, -50%) rotate(0deg); } to { transform: translate(-50%, -50%) rotate(360deg); } }

    /* ===== UI CARD ANIMATIONS ===== */
    .input-group-card {
        background-color: rgba(255, 255, 255, 0.8) !important;
        border-radius: 20px;
        padding: 25px;
        margin-bottom: 25px;
        border: 1px solid #bae6fd;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05);
        transition: all 0.3s ease;
    }
    .input-group-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 12px 20px rgba(14, 165, 233, 0.1);
        border-color: #7dd3fc;
    }
    .splash-icon { font-size: 6rem; margin-bottom: 20px; color: white; }
    .splash-title { font-size: 2.5rem; color: #fff; font-weight: 800; }
    .splash-bar-container {
        width: 250px;
        height: 8px;
        background: rgba(255,255,255,0.2);
        border-radius: 10px;
        margin-top: 30px;
        overflow: hidden;
    }
    .splash-bar {
        height: 100%;
        background: white;
        animation: splashLoad 4s ease-in-out forwards;
    }
    @keyframes splashLoad { 0% { width: 0%; } 100% { width: 100%; } }

    /* ===== CROSSING LINES ===== */
    .crossing-lines-bg { position: relative; padding: 20px 0; }
    .crossing-lines-bg::before, .crossing-lines-bg::after {
        content: ''; position: absolute; top: 50%; left: 50%; width: 200%; height: 2px;
        background: rgba(14, 165, 233, 0.1); z-index: -1;
    }
    .crossing-lines-bg::before { transform: translate(-50%, -50%) rotate(-20deg); }
    .crossing-lines-bg::after { transform: translate(-50%, -50%) rotate(20deg); }

    /* ===== CARDS ===== */
    .model-card {
        background: white;
        border: 1px solid #e0f2fe;
        border-radius: 20px;
        padding: 20px;
        margin: 10px 0;
        box-shadow: 0 4px 15px rgba(0,0,0,0.05);
        text-align: center;
    }
    .card-title { font-weight: 700; font-size: 1rem; color: #0369a1; }
    .card-accent { font-size: 1.2rem; font-weight: 800; margin-top: 5px; }

    /* ===== INPUT CARDS ===== */
    .input-group-card {
        background-color: rgba(255, 255, 255, 0.8) !important;
        border-radius: 20px;
        padding: 25px;
        margin-bottom: 25px;
        border: 1px solid #bae6fd;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05);
    }
    .input-label { 
        font-weight: 800 !important; 
        color: #0369a1 !important; 
        margin-bottom: 15px; 
        font-size: 1.1rem !important; 
        text-transform: uppercase;
        display: block;
        border-bottom: 2px solid #0ea5e9;
        padding-bottom: 5px;
    }

    /* ===== RESULT BOX ===== */
    .result-box {
        background: #0ea5e9;
        border-radius: 25px;
        padding: 30px;
        text-align: center;
        color: white;
        margin-top: 20px;
        box-shadow: 0 10px 30px rgba(14, 165, 233, 0.3);
    }
    .result-text { font-size: 2.2rem; font-weight: 800; }

    /* STREAMLIT BUTTON */
    .stButton>button {
        background: #0ea5e9;
        color: white;
        border-radius: 12px;
        font-weight: 700;
        width: 100%;
        padding: 10px;
        border: none;
    }
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
        <div class="splash-title">Smart hotel prediction</div>
        <div class="splash-bar-container"><div class="splash-bar"></div></div>
    </div>
    """, unsafe_allow_html=True)
    time.sleep(4)
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
    st.error(f"Error loading system: {e}")
    st.stop()

# ============================================================
# UI HEADER
# ============================================================
st.markdown("""
<div class="hero-container">
    <div class="revolving-ring ring-1"></div>
    <div class="revolving-ring ring-2"></div>
    <div class="hero-col">
        <h1 class="hero-title">Smart Hotel</h1>
    </div>
    <div class="hero-col">
        <h1 class="hero-title">Prediction</h1>
    </div>
</div>
""", unsafe_allow_html=True)

# ============================================================
# INPUT SECTION
# ============================================================
with st.container():
    st.markdown("<div class='input-group-card'>", unsafe_allow_html=True)
    st.markdown("<div class='input-label'>📍 Booking Overview</div>", unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    with c1:
        lead_time = st.number_input("Days Before Arrival", 0, 500, 45)
    with c2:
        avg_price = st.slider("Price per Night ($)", 0, 500, 150)
    with c3:
        requests = st.slider("Guest Special Requests", 0, 5, 1)
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<div class='input-group-card'>", unsafe_allow_html=True)
    st.markdown("<div class='input-label'>👥 Guest & Stay Profile</div>", unsafe_allow_html=True)
    c4, c5, c6 = st.columns(3)
    with c4:
        stay_cat = st.selectbox("Total Stay Length", ["Day Use (0 nights)", "Short Stay (1-3)", "Week Stay (4-7)", "Two Weeks (8-14)", "Long Stay (15+)"])
        stay_map = {"Day Use (0 nights)": 0, "Short Stay (1-3)": 1, "Week Stay (4-7)": 2, "Two Weeks (8-14)": 3, "Long Stay (15+)": 4}
    with c5:
        arrival_day = st.selectbox("Day of Arrival", ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"])
        day_map = {"Monday": 0, "Tuesday": 1, "Wednesday": 2, "Thursday": 3, "Friday": 4, "Saturday": 5, "Sunday": 6}
    with c6:
        cancel_ratio = st.slider("Previous Cancellations", 0.0, 1.0, 0.0)
    
    c7, c8 = st.columns(2)
    with c7:
        parking = st.selectbox("Requires Parking?", ["No", "Yes"])
    with c8:
        first_time = st.selectbox("Is New Guest?", ["No", "Yes"])
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<div class='input-group-card'>", unsafe_allow_html=True)
    st.markdown("<div class='input-label'>📦 Service Selection</div>", unsafe_allow_html=True)
    c9, c10, c11, c12 = st.columns(4)
    with c9:
        meal_display = {"Breakfast Only": "Meal Plan 1", "Half Board": "Meal Plan 2", "Full Board": "Meal Plan 3", "None": "Not Selected"}
        meal_choice = st.selectbox("Meal Package", list(meal_display.keys()))
    with c10:
        room_display = {"Standard": "Room_Type 1", "Deluxe": "Room_Type 2", "Junior Suite": "Room_Type 3", "Executive": "Room_Type 4", "Premium": "Room_Type 5", "Presidential": "Room_Type 6", "Luxury Villa": "Room_Type 7"}
        room_choice = st.selectbox("Room Category", list(room_display.keys()))
    with c11:
        market_choice = st.selectbox("Market Segment", ["Aviation", "Complementary", "Corporate", "Offline", "Online"])
    with c12:
        guest_choice = st.selectbox("Group Size", ["1", "2", "3", "4", "5", "Group"])
    st.markdown("</div>", unsafe_allow_html=True)

# ============================================================
# FEATURE ENGINEERING (Fixed Scaling & Internal Month/Year)
# ============================================================
def get_features():
    p_val = 1 if parking == "Yes" else 0
    f_val = 1 if first_time == "Yes" else 0
    
    if lead_time <= 1: lt_cat = 0
    elif lead_time <= 7: lt_cat = 1
    elif lead_time <= 30: lt_cat = 2
    elif lead_time <= 365: lt_cat = 3
    else: lt_cat = 4

    # Use internal defaults to maintain 28-feature compatibility without cluttered UI
    res_month = 6 
    res_year = 2017

    base = [p_val, lt_cat, avg_price, requests, stay_map[stay_cat], day_map[arrival_day], res_month, res_year, cancel_ratio, f_val]

    encoded = []
    curr_meal = meal_display[meal_choice]
    for m in ["Meal Plan 2", "Meal Plan 3", "Not Selected"]: encoded.append(1 if curr_meal == m else 0)
    curr_room = room_display[room_choice]
    for r in ["Room_Type 2", "Room_Type 3", "Room_Type 4", "Room_Type 5", "Room_Type 6", "Room_Type 7"]: encoded.append(1 if curr_room == r else 0)
    for s in ["Complementary", "Corporate", "Offline", "Online"]: encoded.append(1 if market_choice == s else 0)
    for g in ["2", "3", "4", "5", "Group"]: encoded.append(1 if guest_choice == g else 0)
        
    return base + encoded

# ============================================================
# PREDICTION ENGINE
# ============================================================
if st.button("RUN ANALYSIS"):
    feat_row = get_features()
    input_df = pd.DataFrame([feat_row], columns=feature_names)
    
    # FIXED SCALING: Using .values to bypass feature name strictness
    input_df["average_price"] = loaded_scaler.transform(input_df[["average_price"]].values)

    results = []
    model_info = {
        "logistic_regression": "📈 Logistic", "random_forest": "🌲 Forest",
        "knn": "📍 KNN", "xgboost": "⚡ XGBoost",
        "svm": "🔲 SVM", "decision_tree": "🌳 Tree"
    }

    for m_key, m_name in model_info.items():
        if m_key in loaded_models:
            mod = loaded_models[m_key]
            # Use .values to avoid "feature names mismatch" error
            pred = mod.predict(input_df.values)[0]
            conf = mod.predict_proba(input_df.values)[0][1] if hasattr(mod, "predict_proba") else 0.5
            results.append({"Model": m_name, "Pred": "Stay" if pred == 1 else "Cancel", "Conf": conf})

    st.markdown("<div class='crossing-lines-bg'>", unsafe_allow_html=True)
    cols = st.columns(len(results))
    for i, res in enumerate(results):
        with cols[i]:
            st.markdown(f"""
            <div class="model-card">
                <div class="card-title">{res['Model']}</div>
                <div class="card-accent" style="color: {'#16a34a' if res['Pred'] == 'Stay' else '#dc2626'}">{res['Pred']}</div>
                <div class="card-metric">Prob: {res['Conf']:.0%}</div>
            </div>
            """, unsafe_allow_html=True)

    vote_stay = sum(1 for r in results if r['Pred'] == 'Stay')
    final = "NOT CANCELED" if vote_stay >= (len(results)/2) else "CANCELED"
    st.markdown(f"""
    <div class="result-box">
        <div class="result-text">{final}</div>
        <div class="result-sub">Aggregate Decision: {vote_stay}/{len(results)} Models</div>
    </div>
    """, unsafe_allow_html=True)
