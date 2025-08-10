import streamlit as st
import numpy as np
import joblib
from keras.models import load_model

# 1. Set Streamlit config at the top
st.set_page_config(page_title="Credit Card Fraud Detection", layout="centered")

# ===============================
# 2. Load Artifacts
# ===============================
@st.cache_resource
def load_artifacts():
    model = load_model('autoencoder_model.h5')             # Trained model
    mse_min = joblib.load('mse_min.pkl')                   # Min MSE for normalization
    mse_max = joblib.load('mse_max.pkl')                   # Max MSE for normalization
    threshold = joblib.load('best_threshold.pkl')          # Decision threshold
    return model, mse_min, mse_max, threshold

autoencoder, mse_min, mse_max, threshold = load_artifacts()

# ===============================
# 3. Streamlit UI
# ===============================
st.title("🔍 Credit Card Fraud Detection")
st.write("Paste *30 comma-separated values* (Time, V1–V28, Amount):")

user_input = st.text_area("Input Features", placeholder="0.0, 1.1918, ..., 2.69", height=150)

if st.button("Predict"):
    try:
        # Step 1: Convert input string to float list
        values = [float(x.strip()) for x in user_input.split(",")]

        # Step 2: Validate number of inputs
        if len(values) != 30:
            st.error("❌ Please enter exactly 30 numeric values (Time, V1–V28, Amount).")
        else:
            # Step 3: Prepare input for model
            new_txn = np.array(values).reshape(1, -1)

            # Step 4: Run prediction
            reconstructed = autoencoder.predict(new_txn)
            mse = np.mean(np.power(new_txn - reconstructed, 2), axis=1)[0]

            # Step 5: Normalize MSE to anomaly score
            anomaly_score = (mse - mse_min) / (mse_max - mse_min)

            # Step 6: Classify as Fraud or Non-Fraud
            prediction = "⚠ Fraud" if mse > threshold else "✅ Non-Fraud"

            # Step 7: Show Results
            st.success(f"🔍 MSE: {mse:.6f}")
            st.info(f"📈 Anomaly Score: {anomaly_score:.6f}")
            st.markdown(f"### Final Prediction: {prediction}")

    except Exception as e:
        st.error(f"❌ Error: {str(e)}")
