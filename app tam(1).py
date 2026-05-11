"""
Hotel Booking Cancellation Prediction - Streamlit Application
Modern UI with animated cards, start screen, and model comparison.
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
    page_title="Hotel Booking Cancellation Predictor",
    page_icon="🏨",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ============================================================
# CUSTOM CSS - ANIMATED CARDS WITH CROSSING LINES
# ============================================================
st.markdown("""
<style>
    /* ===== GLOBAL STYLES ===== */
    .main-header {
        text-align: center;
        padding: 30px 0 10px 0;
    }
    .main-header h1 {
        font-size: 2.5rem;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 800;
        margin-bottom: 5px;
    }
    .main-header p {
        color: #888;
        font-size: 1.1rem;
    }

    /* ===== START SCREEN OVERLAY ===== */
    .splash-overlay {
        position: fixed;
        top: 0; left: 0; right: 0; bottom: 0;
        background: linear-gradient(135deg, #0f0c29 0%, #302b63 50%, #24243e 100%);
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        z-index: 99999;
        transition: opacity 0.8s ease-out;
    }
    .splash-overlay.fade-out {
        opacity: 0;
        pointer-events: none;
    }
    .splash-icon {
        font-size: 5rem;
        animation: splashPulse 1.5s ease-in-out infinite;
    }
    .splash-title {
        font-size: 2.8rem;
        color: #fff;
        font-weight: 800;
        margin-top: 20px;
        letter-spacing: 2px;
        animation: splashSlideUp 1s ease-out;
    }
    .splash-subtitle {
        color: #aaa;
        font-size: 1.2rem;
        margin-top: 10px;
        animation: splashSlideUp 1.5s ease-out;
    }
    .splash-bar-container {
        width: 200px;
        height: 4px;
        background: rgba(255,255,255,0.15);
        border-radius: 2px;
        margin-top: 30px;
        overflow: hidden;
    }
    .splash-bar {
        height: 100%;
        background: linear-gradient(90deg, #667eea, #764ba2);
        border-radius: 2px;
        animation: splashLoad 4.5s ease-in-out forwards;
    }

    @keyframes splashPulse {
        0%, 100% { transform: scale(1); }
        50% { transform: scale(1.15); }
    }
    @keyframes splashSlideUp {
        from { opacity: 0; transform: translateY(30px); }
        to { opacity: 1; transform: translateY(0); }
    }
    @keyframes splashLoad {
        0% { width: 0%; }
        100% { width: 100%; }
    }

    /* ===== ANIMATED CROSSING LINES BACKGROUND ===== */
    .crossing-lines-bg {
        position: relative;
        overflow: hidden;
        padding: 40px 0;
    }
    .crossing-lines-bg::before,
    .crossing-lines-bg::after {
        content: '';
        position: absolute;
        top: 50%; left: 50%;
        width: 300%;
        height: 2px;
        background: linear-gradient(90deg, transparent, rgba(102,126,234,0.4), transparent);
        animation: crossLine1 6s linear infinite;
    }
    .crossing-lines-bg::before {
        transform: translate(-50%, -50%) rotate(-30deg);
    }
    .crossing-lines-bg::after {
        transform: translate(-50%, -50%) rotate(30deg);
        animation: crossLine2 6s linear infinite;
        background: linear-gradient(90deg, transparent, rgba(118,75,162,0.4), transparent);
    }

    @keyframes crossLine1 {
        0% { transform: translate(-50%, -50%) rotate(-30deg) translateX(-50%); }
        100% { transform: translate(-50%, -50%) rotate(-30deg) translateX(50%); }
    }
    @keyframes crossLine2 {
        0% { transform: translate(-50%, -50%) rotate(30deg) translateX(50%); }
        100% { transform: translate(-50%, -50%) rotate(30deg) translateX(-50%); }
    }

    /* ===== ANIMATED MODEL CARDS ===== */
    .model-card {
        background: rgba(255,255,255,0.05);
        border: 1px solid rgba(255,255,255,0.1);
        border-radius: 16px;
        padding: 20px;
        margin: 10px 0;
        position: relative;
        overflow: hidden;
        transition: transform 0.3s ease, box-shadow 0.3s ease;
        backdrop-filter: blur(10px);
    }
    .model-card:hover {
        transform: translateY(-3px);
        box-shadow: 0 8px 30px rgba(102,126,234,0.3);
    }
    .model-card::before {
        content: '';
        position: absolute;
        top: -50%;
        left: -50%;
        width: 200%;
        height: 200%;
        background: conic-gradient(transparent, rgba(102,126,234,0.15), transparent 30%);
        animation: cardSpin 4s linear infinite;
    }
    @keyframes cardSpin {
        from { transform: rotate(0deg); }
        to { transform: rotate(360deg); }
    }
    .model-card .card-content {
        position: relative;
        z-index: 1;
    }
    .model-card .card-icon {
        font-size: 2rem;
        margin-bottom: 8px;
    }
    .model-card .card-title {
        font-size: 1.1rem;
        font-weight: 700;
        color: #fff;
        margin-bottom: 4px;
    }
    .model-card .card-metric {
        font-size: 0.9rem;
        color: #bbb;
    }
    .model-card .card-accent {
        color: #667eea;
        font-weight: 600;
    }

    /* ===== FEATURE INPUT CARDS ===== */
    .feature-card {
        background: linear-gradient(135deg, rgba(102,126,234,0.08), rgba(118,75,162,0.08));
        border: 1px solid rgba(102,126,234,0.2);
        border-radius: 12px;
        padding: 20px;
        margin-bottom: 15px;
        position: relative;
        overflow: hidden;
    }
    .feature-card::after {
        content: '';
        position: absolute;
        top: 0; right: 0;
        width: 60px; height: 60px;
        background: linear-gradient(135deg, rgba(102,126,234,0.1), transparent);
        border-radius: 0 12px 0 60px;
    }
    .feature-card .feature-label {
        font-weight: 600;
        color: #667eea;
        font-size: 0.95rem;
        margin-bottom: 8px;
        display: flex;
        align-items: center;
        gap: 8px;
    }

    /* ===== PREDICTION RESULT ===== */
    .result-box {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 16px;
        padding: 30px;
        text-align: center;
        color: white;
        margin: 20px 0;
        position: relative;
        overflow: hidden;
    }
    .result-box::before {
        content: '';
        position: absolute;
        top: -50%; left: -50%;
        width: 200%; height: 200%;
        background: radial-gradient(circle, rgba(255,255,255,0.1) 0%, transparent 60%);
        animation: resultGlow 3s ease-in-out infinite;
    }
    @keyframes resultGlow {
        0%, 100% { transform: scale(1); opacity: 0.5; }
        50% { transform: scale(1.1); opacity: 1; }
    }
    .result-box .result-icon {
        font-size: 3.5rem;
        position: relative;
        z-index: 1;
    }
    .result-box .result-text {
        font-size: 1.8rem;
        font-weight: 800;
        margin-top: 10px;
        position: relative;
        z-index: 1;
    }
    .result-box .result-prob {
        font-size: 1.1rem;
        opacity: 0.9;
        margin-top: 5px;
        position: relative;
        z-index: 1;
    }

    /* ===== HIDE STREAMLIT DEFAULTS ===== */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    .stDeployButton {display: none;}
    div[data-testid="stToolbar"] {display: none;}
</style>
""", unsafe_allow_html=True)

# ============================================================
# START SCREEN (5 Seconds)
# ============================================================
splash_placeholder = st.empty()

splash_html = """
<div class="splash-overlay" id="splashOverlay">
    <div class="splash-icon">🏨</div>
    <div class="splash-title">HOTEL CANCELLATION PREDICTOR</div>
    <div class="splash-subtitle">Powered by Machine Learning & Deep Learning</div>
    <div class="splash-bar-container">
        <div class="splash-bar"></div>
    </div>
</div>
"""
splash_placeholder.markdown(splash_html, unsafe_allow_html=True)
time.sleep(5)
splash_placeholder.markdown("""
<style>
    .splash-overlay { animation: fadeOut 0.8s ease-out forwards; }
    @keyframes fadeOut { from { opacity: 1; } to { opacity: 0; pointer-events: none; } }
</style>
""", unsafe_allow_html=True)
time.sleep(0.8)
splash_placeholder.empty()

# ============================================================
# MAIN APPLICATION
# ============================================================

# Header
st.markdown("""
<div class="main-header">
    <h1>Hotel Booking Cancellation Predictor</h1>
    <p>Predict reservation outcomes using 10 machine learning & deep learning models</p>
</div>
""", unsafe_allow_html=True)

st.markdown("<div class='crossing-lines-bg'>", unsafe_allow_html=True)

# ============================================================
# LOAD MODELS & SCALER
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
except FileNotFoundError:
    st.error("Model files not found. Please run training_pipeline.py first.")
    st.stop()

# ============================================================
# FEATURE INPUT SECTION
# ============================================================
st.markdown("### Enter Booking Details")
st.markdown("---")

col_a, col_b, col_c = st.columns(3)

with col_a:
    st.markdown("""
    <div class="feature-card">
        <div class="feature-label">🅿️ Parking</div>
    </div>
    """, unsafe_allow_html=True)
    parking_availability = st.checkbox("Parking Space Available?")

with col_b:
    st.markdown("""
    <div class="feature-card">
        <div class="feature-label">🆕 Guest Type</div>
    </div>
    """, unsafe_allow_html=True)
    new_guest_flag = st.checkbox("First Time Visitor?")

with col_c:
    st.markdown("""
    <div class="feature-card">
        <div class="feature-label">📅 Booking Advance</div>
    </div>
    """, unsafe_allow_html=True)
    advance_days = st.number_input("Lead Time (days)", min_value=0, max_value=500, value=30)

# Categorize advance booking
if advance_days <= 1:
    booking_advance = 0
elif advance_days <= 7:
    booking_advance = 1
elif advance_days <= 30:
    booking_advance = 2
elif advance_days <= 365:
    booking_advance = 3
else:
    booking_advance = 4

col_d, col_e = st.columns(2)

with col_d:
    st.markdown("""
    <div class="feature-card">
        <div class="feature-label">💰 Room Rate</div>
    </div>
    """, unsafe_allow_html=True)
    room_rate = st.slider("Average Price ($)", 1, 500, 100)

with col_e:
    st.markdown("""
    <div class="feature-card">
        <div class="feature-label">📋 Special Requests</div>
    </div>
    """, unsafe_allow_html=True)
    guest_requests = st.slider("Number of Requests", 0, 5, 1)

col_f, col_g, col_h = st.columns(3)

with col_f:
    st.markdown("""
    <div class="feature-card">
        <div class="feature-label">🌙 Stay Duration</div>
    </div>
    """, unsafe_allow_html=True)
    stay_option = st.selectbox(
        "Duration Category",
        options=["Day Use (0 nights)", "Short Stay (1-3)", "Week Stay (4-7)",
                 "Two Weeks (8-14)", "Long Stay (15+)"]
    )
    stay_map = {"Day Use (0 nights)": 0, "Short Stay (1-3)": 1, "Week Stay (4-7)": 2,
                "Two Weeks (8-14)": 3, "Long Stay (15+)": 4}
    stay_duration = stay_map[stay_option]

with col_g:
    st.markdown("""
    <div class="feature-card">
        <div class="feature-label">📆 Arrival Day</div>
    </div>
    """, unsafe_allow_html=True)
    day_choice = st.selectbox(
        "Day of the Week",
        options=["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    )
    arrival_day = {"Monday": 0, "Tuesday": 1, "Wednesday": 2, "Thursday": 3,
                   "Friday": 4, "Saturday": 5, "Sunday": 6}[day_choice]

with col_h:
    st.markdown("""
    <div class="feature-card">
        <div class="feature-label">📊 Cancellation History</div>
    </div>
    """, unsafe_allow_html=True)
    cancel_history = st.slider("Cancel Ratio", 0.0, 1.0, step=0.01, value=0.0)

col_i, col_j = st.columns(2)

with col_i:
    st.markdown("""
    <div class="feature-card">
        <div class="feature-label">🍽️ Dining Plan</div>
    </div>
    """, unsafe_allow_html=True)
    dining_choice = st.selectbox("Meal Plan", ["Meal Plan 1", "Meal Plan 2", "Meal Plan 3", "Not Selected"])

with col_j:
    st.markdown("""
    <div class="feature-card">
        <div class="feature-label">🛏️ Room Type</div>
    </div>
    """, unsafe_allow_html=True)
    room_choice = st.selectbox(
        "Accommodation Class",
        ["Room_Type 1", "Room_Type 2", "Room_Type 3", "Room_Type 4", "Room_Type 5", "Room_Type 6", "Room_Type 7"]
    )

col_k, col_l = st.columns(2)

with col_k:
    st.markdown("""
    <div class="feature-card">
        <div class="feature-label">🌐 Booking Channel</div>
    </div>
    """, unsafe_allow_html=True)
    channel_choice = st.selectbox(
        "Market Segment",
        ["Aviation", "Complementary", "Corporate", "Offline", "Online"]
    )

with col_l:
    st.markdown("""
    <div class="feature-card">
        <div class="feature-label">👥 Guest Count</div>
    </div>
    """, unsafe_allow_html=True)
    guest_choice = st.selectbox(
        "Total Guests",
        ["1", "2", "3", "4", "5", "Group"]
    )

# ============================================================
# BUILD ENCODED FEATURE VECTOR
# ============================================================
encoded_vector = []

# Binary flags
parking_val = 1 if parking_availability else 0
new_guest_val = 1 if new_guest_flag else 0

# Meal encoding (drop_first=True means Meal Plan 1 is baseline)
for meal in ["Meal Plan 2", "Meal Plan 3", "Not Selected"]:
    encoded_vector.append(1 if dining_choice == meal else 0)

# Room type encoding (drop_first=True means Room_Type 1 is baseline)
for room in ["Room_Type 2", "Room_Type 3", "Room_Type 4", "Room_Type 5", "Room_Type 6", "Room_Type 7"]:
    encoded_vector.append(1 if room_choice == room else 0)

# Market segment encoding (drop_first=True means Aviation is baseline)
for segment in ["Complementary", "Corporate", "Offline", "Online"]:
    encoded_vector.append(1 if channel_choice == segment else 0)

# Guest count encoding (drop_first=True means "1" is baseline)
for guest_cat in ["2", "3", "4", "5", "Group"]:
    encoded_vector.append(1 if guest_choice == guest_cat else 0)

# Build full feature row
feature_row = [
    parking_val, booking_advance, room_rate, guest_requests,
    stay_duration, arrival_day, cancel_history, new_guest_val
] + encoded_vector

input_dataframe = pd.DataFrame([feature_row], columns=feature_names)

# Scale room rate
input_dataframe["average_price"] = loaded_scaler.transform(input_dataframe[["average_price"]])

# ============================================================
# PREDICTION
# ============================================================
st.markdown("</div>", unsafe_allow_html=True)

st.markdown("<div class='crossing-lines-bg'>", unsafe_allow_html=True)
st.markdown("### Model Predictions")
st.markdown("---")

if st.button("🔮 Predict Booking Status", use_container_width=True, type="primary"):

    results_data = []

    # Model display configuration
    model_display = {
        "logistic_regression": {"icon": "📈", "name": "Logistic Regression", "has_proba": True},
        "random_forest": {"icon": "🌲", "name": "Random Forest", "has_proba": True},
        "knn": {"icon": "📍", "name": "K-Nearest Neighbors", "has_proba": True},
        "xgboost": {"icon": "⚡", "name": "XGBoost", "has_proba": True},
        "svm": {"icon": "🔲", "name": "Support Vector Machine", "has_proba": True},
        "decision_tree": {"icon": "🌳", "name": "Decision Tree", "has_proba": True},
        "ann": {"icon": "🧠", "name": "Artificial Neural Network", "has_proba": False},
    }

    # Run sklearn models
    for model_key, display_info in model_display.items():
        predictor = loaded_models[model_key]
        forecast = predictor.predict(input_dataframe)[0]

        if display_info["has_proba"] and hasattr(predictor, "predict_proba"):
            probability = predictor.predict_proba(input_dataframe)[0][1]
        else:
            probability = 0.5

        outcome = "Not Canceled" if forecast == 1 else "Canceled"
        results_data.append({
            "Model": display_info["name"],
            "Prediction": outcome,
            "Confidence": f"{probability:.2%}"
        })

    # K-Means segmentation
    kmeans_model = loaded_models["kmeans"]
    cluster_id = kmeans_model.predict(input_dataframe)[0]
    cluster_map = model_config.get("cluster_label_map", {0: 1, 1: 0})
    km_prediction = cluster_map.get(cluster_id, 1)
    km_outcome = "Not Canceled" if km_prediction == 1 else "Canceled"
    results_data.append({"Model": "K-Means Segmentation", "Prediction": km_outcome, "Confidence": "N/A"})

    # Display model cards in grid
    card_cols = st.columns(4)

    for idx, result in enumerate(results_data):
        col = card_cols[idx % 4]
        pred_icon = "✅" if result["Prediction"] == "Not Canceled" else "❌"
        with col:
            model_icon = model_display.get(result["Model"].lower().replace(" ", "_").replace("-", ""), {}).get("icon", "🔮")
            if "K-Means" in result["Model"]:
                model_icon = "🎯"
            if "Neural" in result["Model"]:
                model_icon = "🧠"

            st.markdown(f"""
            <div class="model-card">
                <div class="card-content">
                    <div class="card-icon">{model_icon}</div>
                    <div class="card-title">{result['Model']}</div>
                    <div class="card-metric">
                        {pred_icon} <span class="card-accent">{result['Prediction']}</span>
                    </div>
                    <div class="card-metric">Confidence: {result['Confidence']}</div>
                </div>
            </div>
            """, unsafe_allow_html=True)

    # Majority vote result
    not_canceled_count = sum(1 for r in results_data if r["Prediction"] == "Not Canceled")
    canceled_count = sum(1 for r in results_data if r["Prediction"] == "Canceled")
    final_outcome = "Not Canceled" if not_canceled_count >= canceled_count else "Canceled"
    final_icon = "✅" if final_outcome == "Not Canceled" else "❌"
    final_ratio = max(not_canceled_count, canceled_count) / len(results_data) * 100

    st.markdown(f"""
    <div class="result-box">
        <div class="result-icon">{final_icon}</div>
        <div class="result-text">{final_outcome}</div>
        <div class="result-prob">Ensemble Vote: {max(not_canceled_count, canceled_count)}/{len(results_data)} models ({final_ratio:.0f}%)</div>
    </div>
    """, unsafe_allow_html=True)

    # Results table
    st.markdown("### Detailed Results")
    results_df = pd.DataFrame(results_data)
    st.dataframe(results_df, use_container_width=True, hide_index=True)

else:
    st.info("👆 Enter booking details above and click **Predict** to see results from all 10 models.")

st.markdown("</div>", unsafe_allow_html=True)

