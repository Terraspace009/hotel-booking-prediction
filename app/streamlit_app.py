import streamlit as st
import pandas as pd
import joblib
import os

# Page setup
st.set_page_config(page_title="Hotel Booking Predictor", layout="centered")
st.title("🏨 Hotel Booking Cancellation Predictor")

st.markdown("This app predicts whether a hotel booking will be cancelled based on customer data.")

# 🔍 Try to load the model
model_path = "models/hotel_model.pkl"
if os.path.exists(model_path):
    model = joblib.load(model_path)
else:
    st.error("🚫 Model not found. Please upload 'hotel_model.pkl' to the 'models/' folder.")
    uploaded_file = st.file_uploader("Upload your trained model (.pkl)", type=["pkl"])
    if uploaded_file:
        with open(model_path, "wb") as f:
            f.write(uploaded_file.read())
        st.success("✅ Model uploaded. Please reload the app.")
    st.stop()

# 🚀 Input form
with st.form("prediction_form"):
    lead_time = st.slider("Lead Time (days)", 0, 500, 100)
    adults = st.number_input("Number of Adults", min_value=1, max_value=5, value=2)
    children = st.number_input("Number of Children", min_value=0, max_value=5, value=0)
    previous_cancellations = st.selectbox("Previous Cancellations", [0, 1])
    deposit_type = st.selectbox("Deposit Type", ["No Deposit", "Non Refund", "Refundable"])
    submitted = st.form_submit_button("Predict Cancellation")

# 🧠 Prediction
if submitted:
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


