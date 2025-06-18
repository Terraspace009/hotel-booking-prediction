import streamlit as st
import pandas as pd
import joblib
import os

st.set_page_config(page_title="Hotel Booking Predictor", layout="centered")
st.title("🏨 Hotel Booking Cancellation Predictor")
st.markdown("This app predicts whether a hotel booking will be cancelled based on customer data.")

# Load or upload model
model_path = "models/hotel_model.pkl"
os.makedirs("models", exist_ok=True)

if not os.path.exists(model_path):
    st.error("🚫 Model not found. Please upload 'hotel_model.pkl'")
    uploaded_file = st.file_uploader("Upload model file (.pkl)", type="pkl")
    if uploaded_file:
        with open(model_path, "wb") as f:
            f.write(uploaded_file.read())
        st.success("✅ Model uploaded! Please rerun the app.")
        st.stop()
    else:
        st.stop()
else:
    model = joblib.load(model_path)

# User inputs
with st.form("input_form"):
    lead_time = st.slider("Lead Time", 0, 500, 100)
    adults = st.number_input("Adults", 1, 5, 2)
    children = st.number_input("Children", 0, 5, 0)
    previous_cancellations = st.selectbox("Previous Cancellations", [0, 1])
    deposit_type = st.selectbox("Deposit Type", ["No Deposit", "Non Refund", "Refundable"])
    submit = st.form_submit_button("Predict Cancellation")

# Predict
if submit:
    deposit_map = {"No Deposit": 0, "Non Refund": 1, "Refundable": 2}
    features = pd.DataFrame([[
        lead_time,
        adults,
        children,
        previous_cancellations,
        deposit_map[deposit_type]
    ]], columns=["lead_time", "adults", "children", "previous_cancellations", "deposit_type"])

    prediction = model.predict(features)[0]
    if prediction == 1:
        st.error("❌ This booking is likely to be cancelled.")
    else:
        st.success("✅ This booking is likely to be honored.")
