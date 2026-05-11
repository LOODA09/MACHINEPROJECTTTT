"""
Smart Hotel Prediction - Professional Dashboard
Optimized for 14-Feature Ensemble Models | Fixed Scaling & CSS
"""

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import plotly.graph_objects as go
import plotly.express as px
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
# PREMIUM CSS
# ============================================================
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Manrope:wght@400;500;600;700;800&display=swap');

    html, body, [class*="css"] {
        font-family: 'Manrope', sans-serif;
    }

    .stApp {
        background:
            radial-gradient(circle at 8% 10%, rgba(14, 165, 233, 0.10), transparent 24%),
            radial-gradient(circle at 92% 12%, rgba(245, 158, 11, 0.12), transparent 26%),
            radial-gradient(circle at 50% 100%, rgba(13, 148, 136, 0.10), transparent 36%),
            linear-gradient(180deg, #f8fafc 0%, #edf7f5 100%);
    }

    /* Force labels and sub-text to black */
    .metric-tile span, .section-title, label, .stMarkdown p, .hero-topline {
        color: #000000 !important;
        font-weight: 700;
    }

    /* ===== WELCOME OVERLAY ===== */
    .welcome-overlay {
        position: fixed; inset: 0; z-index: 999999;
        display: flex; align-items: center; justify-content: center;
        background: linear-gradient(135deg, #04111f 0%, #0b3954 48%, #0f766e 100%);
        animation: overlayFade 3.4s ease forwards;
        pointer-events: none;
    }
    .welcome-card {
        text-align: center; color: white; padding: 30px 34px; border-radius: 28px;
        background: rgba(255,255,255,.08); border: 1px solid rgba(255,255,255,.16);
        backdrop-filter: blur(10px); box-shadow: 0 30px 80px rgba(0,0,0,.20);
    }
    @keyframes overlayFade { 0%, 70% { opacity: 1; visibility: visible; } 100% { opacity: 0; visibility: hidden; } }

    /* ===== HERO HEADER ===== */
    .hero-shell {
        position: relative; overflow: hidden; padding: 40px 38px 36px;
        border-radius: 30px; margin-bottom: 22px; color: white;
        background: linear-gradient(-45deg, #07111f, #0f3d56, #0f766e, #064e3b);
        background-size: 400% 400%;
        box-shadow: 0 34px 90px rgba(7, 17, 31, 0.20);
        animation: fadeLift .8s ease both, heroGradientWave 12s ease infinite;
    }
    .hero-topline {
        position: relative; z-index: 2; display: inline-flex; align-items: center; gap: 10px;
        padding: 8px 14px; border-radius: 999px;
        background: rgba(255,255,255,0.9); /* Changed to white background for black text visibility */
        font-size: .84rem; letter-spacing: .04em; text-transform: uppercase;
    }
    .hero-shell h1 { font-size: 2.7rem; color: white !important; }
    .hero-shell p { color: rgba(255,255,255,.86) !important; }

    /* ===== METRICS RIBBON ===== */
    .metrics-ribbon { display: grid; grid-template-columns: repeat(6, minmax(0, 1fr)); gap: 14px; margin: 18px 0 24px; }
    .metric-tile {
        background: white; border: 1px solid rgba(7, 17, 31, 0.08);
        box-shadow: 0 18px 44px rgba(7, 17, 31, 0.08);
        border-radius: 22px; padding: 16px 18px; text-align: center;
    }
    .metric-tile span { color: #000000 !important; }
    .metric-tile strong { color: #07111f; font-size: 1.1rem; font-weight: 800; }

    /* ===== SECTION CARDS ===== */
    .section-card {
        background: white; border-radius: 24px; padding: 25px;
        box-shadow: 0 10px 30px rgba(0,0,0,0.05); margin-bottom: 25px;
    }
    .section-title { font-size: 1.2rem; border-bottom: 3px solid #0f766e; padding-bottom: 5px; margin-bottom: 20px; }

    /* ===== RESULT BOX ===== */
    .live-result { border-radius: 26px; padding: 24px; text-align: center; margin-bottom: 25px; }
    .risk-high { background: #fee2e2; border: 2px solid #ef4444; }
    .risk-low { background: #dcfce7; border: 2px solid #22c55e; }

    @keyframes fadeLift { from { opacity: 0; transform: translateY(14px); } to { opacity: 1; transform: translateY(0); } }
    @keyframes heroGradientWave { 0% { background-position: 0% 50%; } 50% { background-position: 100% 50%; } 100% { background-position: 0% 50%; } }
</style>
""", unsafe_allow_html=True)

# ============================================================
# SPLASH SCREEN
# ============================================================
if "welcome_seen" not in st.session_state:
    st.markdown("""<div class="welcome-overlay"><div class="welcome-card"><h1>Smart Hotel Prediction</h1><p>Initializing Ensemble Intelligence...</p></div></div>""", unsafe_allow_html=True)
    st.session_state["welcome_seen"] = True
    time.sleep(2.5)
    st.rerun()

# ============================================================
# LOAD ARTIFACTS
# ============================================================
@st.cache_resource
def load_artifacts():
    with open("model.pkl", "rb") as f:
        all_models = pickle.load(f)
    with open("scaler_v2.pkl", "rb") as f:
        feature_scaler = pickle.load(f)
    return all_models, feature_scaler

try:
    loaded_models, loaded_scaler = load_artifacts()
    # 14 Core Features in correct order
    feature_names = [
        'number of adults', 'number of children', 'number of weekend nights', 
        'number of week nights', 'type of meal', 'car parking space', 
        'room type', 'lead time', 'market segment type', 'repeated', 
        'P-C', 'P-not-C', 'average price ', 'special requests'
    ]
except Exception as e:
    st.error(f"Error: {e}")
    st.stop()

# ============================================================
# UI HEADER
# ============================================================
st.markdown("""
<div class="hero-shell">
    <div class="hero-topline">Ensemble Prediction Console | 14 Integrated Features</div>
    <h1>Hotel Cancellation Intelligence</h1>
    <p>Professional ensemble analytics using optimized ML models to evaluate reservation risks.</p>
</div>
""", unsafe_allow_html=True)

model_info = {"logistic_regression": "Logistic", "random_forest": "Forest", "knn": "KNN", "xgboost": "XGBoost", "svm": "SVM", "decision_tree": "Tree"}
tiles = "".join(f'<div class="metric-tile"><span>Active Model</span><strong>{name}</strong></div>' for name in model_info.values())
st.markdown(f'<div class="metrics-ribbon">{tiles}</div>', unsafe_allow_html=True)

# ============================================================
# INPUT SECTION
# ============================================================
with st.container():
    st.markdown('<div class="section-card"><div class="section-title">📍 Booking Overview</div>', unsafe_allow_html=True)
    c1, c2, c3, c4 = st.columns(4)
    with c1: adults = st.number_input("Adults", 1, 4, 2)
    with c2: children = st.number_input("Children", 0, 10, 0)
    with c3: weekend_nights = st.number_input("Weekend Nights", 0, 10, 1)
    with c4: week_nights = st.number_input("Week Nights", 0, 20, 2)
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown('<div class="section-card"><div class="section-title">👥 Guest Profile & source</div>', unsafe_allow_html=True)
    c5, c6, c7, c8 = st.columns(4)
    with c5: lead_time = st.number_input("Lead Time (Days)", 0, 500, 45)
    with c6:
        market_map = {"Aviation": 0, "Complementary": 1, "Corporate": 2, "Offline": 3, "Online": 4}
        market = st.selectbox("Market Segment", list(market_map.keys()), index=4)
    with c7: parking = st.selectbox("Car Parking?", ["No", "Yes"])
    with c8: repeated = st.selectbox("Repeated Guest?", ["No", "Yes"])
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown('<div class="section-card"><div class="section-title">📦 Service & History</div>', unsafe_allow_html=True)
    c9, c10, c11, c12 = st.columns(4)
    with c9:
        meal_map = {"Meal Plan 1": 0, "Meal Plan 2": 1, "Meal Plan 3": 2, "Not Selected": 3}
        meal = st.selectbox("Meal Plan", list(meal_map.keys()))
    with c10:
        room_map = {f"Room_Type {i}": i-1 for i in range(1, 8)}
        room = st.selectbox("Room Type", list(room_map.keys()))
    with c11: avg_price = st.slider("Average Price ($)", 0, 500, 120)
    with c12: requests = st.slider("Special Requests", 0, 5, 1)
    st.markdown("</div>", unsafe_allow_html=True)

# ============================================================
# PREDICTION ENGINE
# ============================================================
if st.button("RUN ENSEMBLE ANALYSIS", use_container_width=True):
    # Construct 14-feature row
    # Order: adults, children, weekend, week, meal, parking, room, lead, market, repeated, P-C, P-not-C, price, requests
    feat_row = [
        adults, children, weekend_nights, week_nights,
        meal_map[meal], (1 if parking == "Yes" else 0),
        room_map[room], lead_time, market_map[market],
        (1 if repeated == "Yes" else 0),
        0, 0, # P-C and P-not-C defaults
        avg_price, requests
    ]
    
    input_df = pd.DataFrame([feat_row], columns=feature_names)
    scaled_df = pd.DataFrame(loaded_scaler.transform(input_df), columns=feature_names)

    results = []
    for m_key, m_name in model_info.items():
        if m_key in loaded_models:
            mod = loaded_models[m_key]
            pred = mod.predict(scaled_df.values)[0]
            conf = mod.predict_proba(scaled_df.values)[0][1] if hasattr(mod, "predict_proba") else 0.5
            results.append({"Model": m_name, "Pred": "Stay" if pred == 1 else "Cancel", "Conf": conf})

    vote_stay = sum(1 for r in results if r['Pred'] == 'Stay')
    final = "NOT CANCELED" if vote_stay >= (len(results)/2) else "CANCELED"
    is_high_risk = final == "CANCELED"
    
    st.markdown(f"""<div class="live-result {'risk-high' if is_high_risk else 'risk-low'}"><h3>{'⚠️ High Cancellation Risk' if is_high_risk else '✅ Likely To Stay'}</h3><p>Ensemble Agreement: {vote_stay}/{len(results)} Models</p></div>""", unsafe_allow_html=True)

    detail_cols = st.columns(3)
    for i, res in enumerate(results):
        with detail_cols[i % 3]:
            st.markdown(f"""<div style="background:white; padding:15px; border-radius:16px; border:1px solid #e2e8f0; text-align:center; margin-bottom:10px;"><strong>{res['Model']}</strong><br><span style="color:{'#dc2626' if res['Pred'] == 'Cancel' else '#16a34a'}; font-size:1.4rem; font-weight:800;">{res['Pred']}</span><br>Prob: {res['Conf']:.2%}</div>""", unsafe_allow_html=True)