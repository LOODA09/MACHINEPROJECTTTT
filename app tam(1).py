"""
Hotel Booking Cancellation Predictor - Streamlit Application
Premium Dark Theme | Plotly Graphs | Fixed Scaling Bug
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
# PREMIUM CSS (Extracted & Adapted from Reference)
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

    .block-container {
        padding-top: 1.4rem;
        padding-bottom: 2rem;
    }

    /* ===== WELCOME OVERLAY ===== */
    .welcome-overlay {
        position: fixed;
        inset: 0;
        z-index: 999999;
        display: flex;
        align-items: center;
        justify-content: center;
        background:
            radial-gradient(circle at 50% 20%, rgba(56,189,248,.18), transparent 28%),
            linear-gradient(135deg, #04111f 0%, #0b3954 48%, #0f766e 100%);
        animation: overlayFade 3.4s ease forwards;
        pointer-events: none;
    }

    .welcome-card {
        text-align: center;
        color: white;
        padding: 30px 34px;
        border-radius: 28px;
        background: rgba(255,255,255,.08);
        border: 1px solid rgba(255,255,255,.16);
        backdrop-filter: blur(10px);
        box-shadow: 0 30px 80px rgba(0,0,0,.20);
    }

    .hotel-graphic {
        position: relative;
        width: 170px;
        height: 150px;
        margin: 0 auto 14px;
        animation: hotelFloat 1.8s ease-in-out infinite alternate;
    }

    .hotel-building {
        position: absolute; left: 50%; bottom: 14px; transform: translateX(-50%);
        width: 110px; height: 102px; border-radius: 14px 14px 8px 8px;
        background: linear-gradient(180deg, #f8fafc 0%, #dbeafe 100%);
        box-shadow: 0 18px 40px rgba(0,0,0,.18);
    }

    .hotel-roof {
        position: absolute; left: 50%; top: 8px; transform: translateX(-50%);
        width: 130px; height: 28px; border-radius: 18px 18px 8px 8px;
        background: linear-gradient(90deg, #f59e0b, #fb7185);
    }

    .hotel-sign {
        position: absolute; left: 50%; top: 18px; transform: translateX(-50%);
        padding: 4px 10px; border-radius: 999px; background: #0f766e;
        color: white; font-size: .72rem; font-weight: 800; letter-spacing: .14em; text-transform: uppercase;
    }

    .hotel-windows {
        position: absolute; left: 50%; top: 44px; transform: translateX(-50%);
        width: 72px; display: grid; grid-template-columns: repeat(3, 1fr); gap: 8px;
    }

    .hotel-window {
        width: 18px; height: 18px; border-radius: 4px;
        background: linear-gradient(180deg, #fde68a, #f59e0b);
        box-shadow: 0 0 16px rgba(245,158,11,.35);
        animation: windowGlow 1.6s ease-in-out infinite alternate;
    }

    .hotel-window:nth-child(2), .hotel-window:nth-child(5) { animation-delay: .4s; }
    .hotel-window:nth-child(3), .hotel-window:nth-child(6) { animation-delay: .8s; }

    .hotel-door {
        position: absolute; left: 50%; bottom: 0; transform: translateX(-50%);
        width: 26px; height: 34px; border-radius: 8px 8px 0 0;
        background: linear-gradient(180deg, #0f172a, #334155);
    }

    .hotel-base {
        position: absolute; left: 50%; bottom: 0; transform: translateX(-50%);
        width: 150px; height: 10px; border-radius: 999px; background: rgba(255,255,255,.18);
    }

    .welcome-title { font-size: 2rem; font-weight: 800; letter-spacing: -.02em; margin-bottom: 8px; color: #f8fafc; }
    .welcome-copy { color: rgba(255,255,255,.84); font-size: 1rem; }
    
    .welcome-bar {
        width: 260px; height: 10px; margin: 16px auto 0; border-radius: 999px;
        overflow: hidden; background: rgba(255,255,255,.12);
    }

    .welcome-bar::after {
        content: ""; display: block; height: 100%; width: 45%; border-radius: 999px;
        background: linear-gradient(90deg, #fde68a, #38bdf8, #34d399);
        animation: loadingSweep 2.4s ease-in-out infinite;
    }

    /* ===== HERO HEADER ===== */
    .hero-shell {
        position: relative; overflow: hidden; padding: 40px 38px 36px;
        border-radius: 30px; margin-bottom: 22px; color: white;
        background: linear-gradient(-45deg, #07111f, #0f3d56, #0f766e, #064e3b);
        background-size: 400% 400%;
        box-shadow: 0 34px 90px rgba(7, 17, 31, 0.20);
        animation: fadeLift .8s ease both, heroGradientWave 12s ease infinite;
    }

    .hero-shell::before, .hero-shell::after {
        content: ""; position: absolute; left: -10%; width: 120%; height: 300px;
        border-radius: 42%; opacity: .12; pointer-events: none;
    }

    .hero-shell::before { bottom: -220px; background: linear-gradient(90deg, #38bdf8, #0ea5e9, #34d399); animation: waveDriftA 14s linear infinite; }
    .hero-shell::after { bottom: -240px; background: linear-gradient(90deg, #fde68a, #f59e0b, #fb7185); animation: waveDriftB 18s linear infinite; }

    .hero-topline {
        position: relative; z-index: 2; display: inline-flex; align-items: center; gap: 10px;
        padding: 8px 14px; border-radius: 999px;
        background: rgba(255,255,255,.10); border: 1px solid rgba(255,255,255,.16);
        font-size: .84rem; letter-spacing: .04em; text-transform: uppercase;
    }

    .hero-shell h1 {
        position: relative; z-index: 2; margin: 14px 0 10px;
        font-size: 2.7rem; line-height: 1; letter-spacing: -.03em;
    }

    .hero-shell p {
        position: relative; z-index: 2; margin: 0; max-width: 880px;
        font-size: 1rem; line-height: 1.65; color: rgba(255,255,255,.86);
    }

    /* ===== METRICS RIBBON (Used for Model Badges) ===== */
    .metrics-ribbon {
        display: grid; grid-template-columns: repeat(6, minmax(0, 1fr)); gap: 14px; margin: 18px 0 24px;
    }

    .metric-tile {
        background: rgba(255,255,255,.82); border: 1px solid rgba(7, 17, 31, 0.08);
        box-shadow: 0 18px 44px rgba(7, 17, 31, 0.08); backdrop-filter: blur(12px);
        border-radius: 22px; padding: 16px 18px; text-align: center;
        animation: fadeLift .7s ease both;
    }

    .metric-tile span { display: block; color: #52606d; font-size: .8rem; margin-bottom: 6px; }
    .metric-tile strong { display: block; color: #07111f; font-size: 1.1rem; font-weight: 800; }

    /* ===== SECTION CARDS (Used for Inputs) ===== */
    .section-card {
        background: rgba(255,255,255,.86); border: 1px solid rgba(7, 17, 31, 0.08);
        border-radius: 24px; padding: 20px 20px 12px;
        box-shadow: 0 20px 50px rgba(7, 17, 31, 0.08); backdrop-filter: blur(12px);
        animation: fadeLift .75s ease both; margin-bottom: 25px;
    }

    .section-title { margin: 0 0 4px; color: #07111f; font-size: 1.22rem; font-weight: 800; text-transform: uppercase; border-bottom: 2px solid #0f766e; padding-bottom: 8px; display: inline-block;}

    /* ===== LIVE RESULT BOX ===== */
    .live-result {
        position: relative; overflow: hidden; border-radius: 26px;
        padding: 24px; margin-bottom: 24px;
        border: 1px solid rgba(14,165,233,.14);
        background: linear-gradient(135deg, rgba(255,255,255,.92), rgba(224,242,254,.88));
        box-shadow: 0 18px 40px rgba(14,165,233,.12);
        animation: pulseCard 2.2s ease-in-out infinite;
        text-align: center;
    }

    .live-result.risk-high {
        background: linear-gradient(135deg, rgba(255,255,255,.95), rgba(254,226,226,.92));
        border-color: rgba(239,68,68,.18);
    }

    .live-result.risk-low {
        background: linear-gradient(135deg, rgba(255,255,255,.95), rgba(220,252,231,.90));
        border-color: rgba(34,197,94,.18);
    }

    .live-result::before, .live-result::after {
        content: ""; position: absolute; left: -10%; width: 120%; height: 70px;
        border-radius: 44%; opacity: .18; pointer-events: none;
    }

    .live-result::before { bottom: -24px; background: linear-gradient(90deg, #38bdf8, #0ea5e9, #14b8a6); animation: waveDriftA 7s linear infinite; }
    .live-result::after { bottom: -34px; background: linear-gradient(90deg, #fde68a, #f59e0b, #fb7185); animation: waveDriftB 10s linear infinite; }

    .live-result h3 { margin: 0 0 6px; font-size: 2.2rem; color: #0f172a; font-weight: 800; }
    .live-result p { margin: 0; color: #475569; font-size: 1.1rem; }

    /* ===== STREAMLIT BUTTON ===== */
    div.stButton > button {
        min-height: 50px; border-radius: 15px; border: 0; font-weight: 800;
        background: linear-gradient(90deg, #0f766e, #0ea5e9);
        color: white; box-shadow: 0 16px 30px rgba(14, 165, 233, .22);
        transition: transform .16s ease, box-shadow .16s ease;
    }

    div.stButton > button:hover {
        transform: translateY(-2px); box-shadow: 0 22px 36px rgba(14, 165, 233, .30);
    }

    /* ===== ANIMATIONS ===== */
    @keyframes fadeLift { from { opacity: 0; transform: translateY(14px); } to { opacity: 1; transform: translateY(0); } }
    @keyframes hotelFloat { from { transform: translateY(0px) scale(1); } to { transform: translateY(-8px) scale(1.03); } }
    @keyframes loadingSweep { 0% { transform: translateX(-140%); } 100% { transform: translateX(320%); } }
    @keyframes windowGlow { from { opacity: .62; } to { opacity: 1; } }
    @keyframes overlayFade { 0%, 70% { opacity: 1; visibility: visible; } 100% { opacity: 0; visibility: hidden; } }
    @keyframes pulseCard { 0%, 100% { transform: translateY(0px); box-shadow: 0 18px 40px rgba(14,165,233,.12); } 50% { transform: translateY(-3px); box-shadow: 0 24px 50px rgba(14,165,233,.18); } }
    @keyframes waveDriftA { from { transform: translateX(-4%) rotate(0deg); } to { transform: translateX(4%) rotate(360deg); } }
    @keyframes waveDriftB { from { transform: translateX(5%) rotate(360deg); } to { transform: translateX(-5%) rotate(0deg); } }
    @keyframes heroGradientWave { 0% { background-position: 0% 50%; } 50% { background-position: 100% 50%; } 100% { background-position: 0% 50%; } }

    @media (max-width: 900px) { .metrics-ribbon { grid-template-columns: repeat(3, minmax(0, 1fr)); } }
    @media (max-width: 600px) { .metrics-ribbon { grid-template-columns: repeat(2, minmax(0, 1fr)); } .hero-shell h1 { font-size: 2rem; } }
</style>
""", unsafe_allow_html=True)

# ============================================================
# SESSION STATE & PREMIUM SPLASH SCREEN
# ============================================================
if "welcome_seen" not in st.session_state:
    st.markdown(
        """
        <div class="welcome-overlay">
            <div class="welcome-card">
                <div class="hotel-graphic">
                    <div class="hotel-roof"></div>
                    <div class="hotel-building">
                        <div class="hotel-sign">Hotel</div>
                        <div class="hotel-windows">
                            <div class="hotel-window"></div><div class="hotel-window"></div><div class="hotel-window"></div>
                            <div class="hotel-window"></div><div class="hotel-window"></div><div class="hotel-window"></div>
                        </div>
                        <div class="hotel-door"></div>
                    </div>
                    <div class="hotel-base"></div>
                </div>
                <div class="welcome-title">Smart Hotel Prediction</div>
                <div class="welcome-copy">Preparing intelligent risk analytics and ensemble models.</div>
                <div class="welcome-bar"></div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.session_state["welcome_seen"] = True
    time.sleep(2.8)
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
# UI HEADER (Dark Animated Hero)
# ============================================================
st.markdown(
    """
    <div class="hero-shell">
        <div class="hero-topline">Ensemble Prediction Console | 6 Optimized Models</div>
        <h1>Hotel Cancellation Intelligence</h1>
        <p>
            Professional ensemble prediction utilizing Logistic Regression, Random Forest, KNN, XGBoost, SVM, and Decision Trees to evaluate live booking risk.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

# Model Tiles
model_info = {
    "logistic_regression": "📈 Logistic", "random_forest": "🌲 Forest",
    "knn": "📍 KNN", "xgboost": "⚡ XGBoost",
    "svm": "🔲 SVM", "decision_tree": "🌳 Tree"
}
tiles_html = "".join(f'<div class="metric-tile"><span>Active Model</span><strong>{name}</strong></div>' for name in model_info.values())
st.markdown(f'<div class="metrics-ribbon">{tiles_html}</div>', unsafe_allow_html=True)

# ============================================================
# INPUT SECTION (Glassmorphism Cards)
# ============================================================
with st.container():
    st.markdown('<div class="section-card"><div class="section-title">📍 Booking Overview</div>', unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    with c1: lead_time = st.number_input("Days Before Arrival", 0, 500, 45)
    with c2: avg_price = st.slider("Price per Night ($)", 0, 500, 150)
    with c3: requests = st.slider("Guest Special Requests", 0, 5, 1)
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown('<div class="section-card"><div class="section-title">👥 Guest & Stay Profile</div>', unsafe_allow_html=True)
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

    st.markdown('<div class="section-card"><div class="section-title">📦 Service Selection</div>', unsafe_allow_html=True)
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

# ============================================================
# FEATURE ENGINEERING
# ============================================================
def get_features():
    p_val = 1 if parking == "Yes" else 0
    f_val = 1 if first_time == "Yes" else 0
    
    if lead_time <= 1: lt_cat = 0
    elif lead_time <= 7: lt_cat = 1
    elif lead_time <= 30: lt_cat = 2
    elif lead_time <= 365: lt_cat = 3
    else: lt_cat = 4

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
# PLOTLY CHART BUILDER (From Reference)
# ============================================================
def build_ensemble_probability_chart(results):
    df = pd.DataFrame(results)
    df = df.sort_values("Conf", ascending=True)
    
    fig = go.Figure()
    for _, row in df.iterrows():
        color = "#ef4444" if row['Conf'] >= 0.5 else "#22c55e"
        fig.add_trace(go.Bar(
            y=[row["Model"]],
            x=[row["Conf"]],
            orientation='h',
            marker_color=color,
            text=f"{row['Conf']:.0%}",
            textposition='auto',
            hoverinfo='skip'
        ))
        
    fig.update_layout(
        title="Individual Model Cancellation Probabilities",
        height=320,
        margin=dict(l=10, r=10, t=40, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(248,250,252,0.7)",
        xaxis=dict(range=[0, 1], tickformat='.0%', gridcolor='rgba(0,0,0,0.05)'),
        yaxis=dict(visible=False),
        barnorm='fraction',
        showlegend=False
    )
    return fig

def build_probability_gauge(prob):
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=prob * 100,
        number={"suffix": "%", "font": {"size": 36, "color": "#0f172a"}},
        gauge={
            "axis": {"range": [0, 100]},
            "bar": {"color": "#0ea5e9"},
            "steps": [
                {"range": [0, 35], "color": "#dcfce7"},
                {"range": [35, 65], "color": "#fef3c7"},
                {"range": [65, 100], "color": "#fee2e2"},
            ],
            "threshold": {"line": {"color": "#ef4444", "width": 4}, "thickness": 0.8, "value": prob * 100},
        },
        title={"text": "Ensemble Cancellation Risk"}
    ))
    fig.update_layout(height=280, margin=dict(l=10, r=10, t=50, b=10), paper_bgcolor="rgba(0,0,0,0)")
    return fig

# ============================================================
# PREDICTION ENGINE
# ============================================================
if st.button("RUN ANALYSIS", use_container_width=True):
    feat_row = get_features()
    input_df = pd.DataFrame([feat_row], columns=feature_names)
    
    # FIXED SCALING BUG
    scaled_data = loaded_scaler.transform(input_df)
    input_df = pd.DataFrame(scaled_data, columns=feature_names)

    results = []

    for m_key, m_name in model_info.items():
        if m_key in loaded_models:
            mod = loaded_models[m_key]
            pred = mod.predict(input_df.values)[0]
            conf = mod.predict_proba(input_df.values)[0][1] if hasattr(mod, "predict_proba") else 0.5
            results.append({"Model": m_name, "Pred": "Stay" if pred == 1 else "Cancel", "Conf": conf})

    vote_stay = sum(1 for r in results if r['Pred'] == 'Stay')
    final = "NOT CANCELED" if vote_stay >= (len(results)/2) else "CANCELED"
    is_high_risk = final == "CANCELED"
    risk_class = "risk-high" if is_high_risk else "risk-low"
    
    avg_cancel_prob = np.mean([r['Conf'] for r in results])

    # Render Premium Results UI
    st.markdown(
        f"""
        <div class="live-result {risk_class}">
            <h3>{'⚠️ Booking Likely To Cancel' if is_high_risk else '✅ Booking Likely To Stay'}</h3>
            <p>Aggregate Decision: {vote_stay}/{len(results)} Models Voted To Keep Reservation</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    gauge_col, chart_col = st.columns([0.8, 1.2], gap="large")
    
    with gauge_col:
        st.plotly_chart(build_probability_gauge(avg_cancel_prob), use_container_width=True)
        
    with chart_col:
        st.plotly_chart(build_ensemble_probability_chart(results), use_container_width=True)
        
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">📊 Detailed Model Breakdown</div>', unsafe_allow_html=True)
    
    detail_cols = st.columns(3)
    for i, res in enumerate(results):
        with detail_cols[i % 3]:
            border_color = "#dc2626" if res['Pred'] == 'Cancel' else "#16a34a"
            bg_color = "#fef2f2" if res['Pred'] == 'Cancel' else "#f0fdf4"
            st.markdown(
                f"""
                <div style="background:{bg_color}; padding:15px; border-radius:16px; border:1px solid {border_color}; text-align:center; margin-bottom:10px;">
                    <div style="font-size:1.1rem; font-weight:800; color:#0f172a;">{res['Model']}</div>
                    <div style="font-size:1.4rem; font-weight:800; color:{border_color}; margin:5px 0;">{res['Pred']}</div>
                    <div style="font-size:0.9rem; color:#475569;">Prob: {res['Conf']:.2%}</div>
                </div>
                """,
                unsafe_allow_html=True
            )
    st.markdown("</div>", unsafe_allow_html=True)