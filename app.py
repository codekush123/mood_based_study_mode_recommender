import streamlit as st
import pandas as pd
import joblib

# Load model and encoders
clf = joblib.load("mood_based_study_mode_model.pkl")
encoders = joblib.load("feature_encoders.pkl")
target_le = joblib.load("target_encoder.pkl")

st.title("Mood-Based Study Mode Recommender")

user_energy_state = st.selectbox("Select your energy state:", encoders["user_energy_state"].classes_)
user_emotional_state = st.selectbox("Select your emotional state:", encoders["user_emotional_state"].classes_)
available_time_minutes = st.number_input("Enter available time in minutes:", min_value=5, max_value=180, step=5)
day_type = st.selectbox("Select the type of day:", encoders["day_type"].classes_)
last_session_outcome = st.selectbox("Last Session Outcome:", encoders["last_session_outcome"].classes_)
recent_study_duration_avg = st.number_input("Recent Study Duration Avg (minutes):", min_value=5, max_value=180, step=5)

input_dict = {
    "user_energy_state": [encoders["user_energy_state"].transform([user_energy_state])[0]],
    "user_emotional_state": [encoders["user_emotional_state"].transform([user_emotional_state])[0]],
    "available_time_minutes": [available_time_minutes],
    "day_type": [encoders["day_type"].transform([day_type])[0]],
    "last_session_outcome": [encoders["last_session_outcome"].transform([last_session_outcome])[0]],
    "recent_study_duration_avg": [recent_study_duration_avg]
}

input_df = pd.DataFrame(input_dict)

prediction_encoded = clf.predict(input_df)[0]
prediction = target_le.inverse_transform([prediction_encoded])[0]

st.success(f"Recommended Study Mode: {prediction}")
