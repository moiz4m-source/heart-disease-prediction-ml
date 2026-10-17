
import streamlit as st
import pandas as pd
import joblib

# Load the trained model, scaler, and feature columns
model = joblib.load("LogReg.pkl")
scaler = joblib.load("scaler.pkl")
columns = joblib.load("cols.pkl")

# Page setup
st.title("Heart Disease Prediction")
st.markdown("Provide the following details to get a prediction.")

# User inputs
age = st.slider("Age", 18, 100, 40)

sex = st.selectbox("Sex", ["M", "F"])

chest_pain = st.selectbox(
    "Chest Pain Type", ["ATA", "NAP", "TA", "ASY"]
)

resting_bp = st.number_input(
    "Resting Blood Pressure (mm Hg)", 80, 200, 120
)

cholesterol = st.number_input(
    "Cholesterol (mg/dL)", 100, 600, 200
)

fasting_bs = st.selectbox(
    "Fasting Blood Sugar > 120 mg/dL", [0, 1]
)

resting_ecg = st.selectbox(
    "Resting ECG", ["Normal", "ST", "LVH"]
)

max_hr = st.slider("Max Heart Rate", 60, 220, 150)

exercise_angina = st.selectbox(
    "Exercise-Induced Angina", ["Y", "N"]
)

oldpeak = st.slider(
    "Oldpeak (ST Depression)", 0.0, 6.0, 1.0
)

st_slope = st.selectbox(
    "ST Slope", ["Up", "Flat", "Down"]
)

# Prediction
if st.button("Predict"):

    # Create the raw input
    raw_input = {
        "Age": age,
        "RestingBP": resting_bp,
        "Cholesterol": cholesterol,
        "FastingBS": fasting_bs,
        "MaxHR": max_hr,
        "Oldpeak": oldpeak,
        "Sex_" + sex: 1,
        "ChestPainType_" + chest_pain: 1,
        "RestingECG_" + resting_ecg: 1,
        "ExerciseAngina_" + exercise_angina: 1,
        "ST_Slope_" + st_slope: 1,
    }

    # Create DataFrame
    input_df = pd.DataFrame([raw_input])

    # Ensure exact training columns and order
    input_df = input_df.reindex(
        columns=columns,
        fill_value=0
    )

    # Scale only the four numeric features
    num_cols = [
        "RestingBP",
        "Cholesterol",
        "MaxHR",
        "Oldpeak"
    ]

    try:
        scaled_input = input_df.copy()

        scaled_input[num_cols] = scaler.transform(
            input_df[num_cols]
        )

        # Predict
        prediction = model.predict(scaled_input)[0]

        # Display result
        if prediction == 1:
            st.error("⚠️ High Risk of Heart Disease")
        else:
            st.success("✅ Low Risk of Heart Disease")

    except Exception as e:
        st.error(f"Prediction error: {e}")

