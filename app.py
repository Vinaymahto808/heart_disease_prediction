%%writefile app.py

import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load the saved components
scaler = joblib.load('scaler.joblib')
model = joblib.load('logistic_regression_model.joblib')
feature_names = joblib.load('feature_names.joblib')

# --- Streamlit Application ---
st.set_page_config(page_title="Heart Disease Prediction", layout="centered")
st.title("Heart Disease Prediction Application")
st.write("Enter patient details to predict the likelihood of heart disease.")

# Input fields for numerical features
st.sidebar.header("Patient Data Input")
age = st.sidebar.slider("Age", 18, 100, 50)
resting_bp = st.sidebar.slider("Resting Blood Pressure (mm/Hg)", 80, 200, 120)
cholesterol = st.sidebar.slider("Cholesterol (mg/dl)", 0, 600, 200)
max_hr = st.sidebar.slider("Maximum Heart Rate Achieved", 60, 220, 150)
oldpeak = st.sidebar.slider("Oldpeak (ST depression induced by exercise relative to rest)", 0.0, 6.2, 1.0, 0.1)
fasting_bs = st.sidebar.selectbox("Fasting Blood Sugar > 120 mg/dl?", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")

# Input fields for categorical features
sex = st.sidebar.radio("Sex", ['Male', 'Female'])
chest_pain_type = st.sidebar.selectbox("Chest Pain Type", ['ATA', 'NAP', 'ASY', 'TA'])
resting_ecg = st.sidebar.selectbox("Resting ECG", ['Normal', 'ST', 'LVH'])
exercise_angina = st.sidebar.radio("Exercise Induced Angina", ['Yes', 'No'])
st_slope = st.sidebar.selectbox("ST Slope", ['Up', 'Flat', 'Down'])

# Map categorical inputs to numerical values/dummy variable logic
sex_male = 1 if sex == 'Male' else 0
exercise_angina_y = 1 if exercise_angina == 'Yes' else 0

# Create a dictionary for user input
user_input = {
    'Age': age,
    'RestingBP': resting_bp,
    'Cholesterol': cholesterol,
    'FastingBS': fasting_bs,
    'MaxHR': max_hr,
    'Oldpeak': oldpeak,
    'Sex_M': sex_male,
    'ChestPainType_ATA': 1 if chest_pain_type == 'ATA' else 0,
    'ChestPainType_NAP': 1 if chest_pain_type == 'NAP' else 0,
    'ChestPainType_TA': 1 if chest_pain_type == 'TA' else 0, # ASY is the reference category
    'RestingECG_Normal': 1 if resting_ecg == 'Normal' else 0,
    'RestingECG_ST': 1 if resting_ecg == 'ST' else 0, # LVH is the reference category
    'ExerciseAngina_Y': exercise_angina_y,
    'ST_Slope_Flat': 1 if st_slope == 'Flat' else 0,
    'ST_Slope_Up': 1 if st_slope == 'Up' else 0  # Down is the reference category
}

# Convert user input to DataFrame
user_df = pd.DataFrame([user_input])

# Ensure all feature_names are present in user_df, add missing ones with 0
for col in feature_names:
    if col not in user_df.columns:
        user_df[col] = 0

# Drop any extra columns that might not be in feature_names
extra_cols = [col for col in user_df.columns if col not in feature_names]
if extra_cols:
    user_df = user_df.drop(columns=extra_cols)

# Reorder columns to match the training data
user_df = user_df[feature_names]

# Scale ALL features using the loaded scaler, as the scaler was fitted on 15 features.
# The result of scaler.transform is a numpy array, convert it back to DataFrame
user_df_scaled = pd.DataFrame(scaler.transform(user_df), columns=feature_names)


# Prediction button
if st.sidebar.button('Predict'):
    prediction = model.predict(user_df_scaled)
    prediction_proba = model.predict_proba(user_df_scaled)[:, 1]

    st.subheader("Prediction Results")
    if prediction[0] == 1:
        st.error(f"The model predicts a HIGH likelihood of Heart Disease with a probability of {prediction_proba[0]:.2f}")
    else:
        st.success(f"The model predicts a LOW likelihood of Heart Disease with a probability of {prediction_proba[0]:.2f}")

    st.write("### Input Features (Preprocessed and Scaled):")
    st.write(user_df_scaled)
