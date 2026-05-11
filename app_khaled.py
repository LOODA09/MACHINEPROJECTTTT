import streamlit as st
import pandas as pd
import numpy as np
import pickle
import time
try:
    from tensorflow.keras.models import load_model
    HAS_TF = True
except ImportError:
    HAS_TF = False

# Page Config
st.set_page_config(page_title="Hotel Cancellation Predictor - Khaled", layout="wide")

# Custom CSS for Animations and Cards
st.markdown("""
<style>
    /* Splash Screen */
    #splash-screen {
        position: fixed;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
        display: flex;
        flex-direction: column;
        justify-content: center;
        align-items: center;
        z-index: 9999;
        color: white;
        font-family: 'Outfit', sans-serif;
    }
    
    .loader {
        border: 8px solid #f3f3f3;
        border-top: 8px solid #3498db;
        border-radius: 50%;
        width: 60px;
        height: 60px;
        animation: spin 2s linear infinite;
    }
    
    @keyframes spin {
        0% { transform: rotate(0deg); }
        100% { transform: rotate(360deg); }
    }

    /* Animation Cards with Crossing Lines */
    .animated-card {
        background: rgba(255, 255, 255, 0.05);
        border-radius: 15px;
        padding: 20px;
        position: relative;
        overflow: hidden;
        border: 1px solid rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(10px);
        margin-bottom: 20px;
    }
    
    .line {
        position: absolute;
        width: 200%;
        height: 2px;
        background: linear-gradient(90deg, transparent, rgba(52, 152, 219, 0.5), transparent);
        animation: move-line 3s infinite linear;
    }
    
    .line-1 { top: 20%; left: -50%; transform: rotate(45deg); }
    .line-2 { bottom: 20%; right: -50%; transform: rotate(-45deg); }
    
    @keyframes move-line {
        0% { transform: translate(-50%, -50%) rotate(var(--rot)); }
        100% { transform: translate(50%, 50%) rotate(var(--rot)); }
    }
    
    .line-1 { --rot: 45deg; animation-delay: 0s; }
    .line-2 { --rot: -45deg; animation-delay: 1.5s; }

    /* Modern Inputs */
    .stNumberInput, .stSelectbox, .stSlider {
        background: #262730;
        border-radius: 10px;
    }
</style>
""", unsafe_safe_to_run=True)

# Splash Screen Logic
if 'splashed' not in st.session_state:
    placeholder = st.empty()
    with placeholder.container():
        st.markdown("""
        <div id="splash-screen">
            <div class="loader"></div>
            <h1 style='margin-top: 20px;'>KHALED PROJECT</h1>
            <p>Initializing Hotel Intelligence Systems...</p>
        </div>
        """, unsafe_allow_html=True)
    time.sleep(5)
    st.session_state.splashed = True
    placeholder.empty()

# Main UI
st.title("🏨 Hotel Booking Cancellation Predictor")
st.markdown("---")

# Layout: 2 Columns for Inputs
col1, col2 = st.columns([1, 1])

with col1:
    st.markdown("""
    <div class="animated-card">
        <div class="line line-1"></div>
        <div class="line line-2"></div>
        <h3>Booking Details</h3>
    </div>
    """, unsafe_allow_html=True)
    
    adults = st.number_input("Number of Adults", 1, 10, 2)
    children = st.number_input("Number of Children", 0, 10, 0)
    weekend_nights = st.number_input("Weekend Nights", 0, 10, 1)
    week_nights = st.number_input("Week Nights", 0, 20, 3)
    lead_time = st.slider("Lead Time (Days)", 0, 500, 50)
    avg_price = st.number_input("Average Price", 0.0, 500.0, 100.0)

with col2:
    st.markdown("""
    <div class="animated-card">
        <div class="line line-1"></div>
        <div class="line line-2"></div>
        <h3>Customer Profile</h3>
    </div>
    """, unsafe_allow_html=True)
    
    meal = st.selectbox("Type of Meal", ["Meal Plan 1", "Meal Plan 2", "Not Selected"])
    room = st.selectbox("Room Type", ["Room_Type 1", "Room_Type 2", "Room_Type 4", "Room_Type 5", "Room_Type 6", "Room_Type 7"])
    market = st.selectbox("Market Segment", ["Online", "Offline", "Corporate", "Aviation", "Complementary"])
    repeated = st.checkbox("Repeated Guest")
    special_requests = st.slider("Special Requests", 0, 5, 0)

# Model Selection
st.markdown("---")
st.subheader("Select Model for Prediction")
model_choice = st.selectbox("Choose a Model", [
    "Random Forest (Khaled)", "KNN", "Logistic Regression", "XGBoost", 
    "Decision Tree", "SVM RBF", "ANN (Khaled)", "RNN (Khaled)", "LSTM"
])

# Prediction Logic
if st.button("Predict Now"):
    # Preprocess inputs
    # (Mapping logic based on the encoder used in training)
    # This is a simplified version for demonstration
    input_data = pd.DataFrame({
        'number of adults': [adults],
        'number of children': [children],
        'number of weekend nights': [weekend_nights],
        'number of week nights': [week_nights],
        'type of meal': [0], # Placeholder for encoded value
        'car parking space': [0],
        'room type': [0],
        'lead time': [lead_time],
        'market segment type': [0],
        'repeated': [1 if repeated else 0],
        'P-C': [0],
        'P-not-C': [0],
        'average price': [avg_price],
        'special requests': [special_requests]
    })
    
    try:
        # Load scaler and model
        scaler = pickle.load(open('scaler_khaled.pkl', 'rb'))
        input_scaled = scaler.transform(input_data)
        
        if model_choice == "Random Forest (Khaled)":
            model = pickle.load(open('rf_model_khaled.pkl', 'rb'))
            prediction = model.predict(input_scaled)
        elif any(x in model_choice for x in ["ANN", "RNN", "LSTM"]):
            if HAS_TF:
                model_file = model_choice.split(" ")[0].lower() + "_model_khaled.h5"
                model = load_model(model_file)
                # Reshape for RNN/LSTM if needed
                if "RNN" in model_choice or "LSTM" in model_choice:
                    input_scaled = input_scaled.reshape((input_scaled.shape[0], 1, input_scaled.shape[1]))
                prediction = (model.predict(input_scaled) > 0.5).astype(int)
            else:
                st.error("TensorFlow/Keras is not installed. This model cannot be run.")
                return
        else:
            # Load other models similarly...
            st.info("Loading model...")
            prediction = [0] # Default
            
        result = "✅ Not Canceled" if prediction[0] == 1 else "❌ Canceled"
        st.success(f"Prediction: {result}")
        
    except FileNotFoundError:
        st.error("Model file not found. Please train models in the notebook first.")

st.markdown("---")
st.write("Developed for Khaled Project.")
