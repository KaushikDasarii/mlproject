# app.py
import streamlit as st
import pandas as pd
import numpy as np

from src.pipeline.predict_pipeline import CustomData, PredictPipeline

st.set_page_config(page_title="Student Exam Score Predictor", layout="centered")

st.title(" Student Exam Score Predictor")
st.write("Enter the student's details below:")

# Input form
with st.form(key='prediction_form'):
    gender = st.selectbox("Gender", ["male", "female"])
    ethnicity = st.selectbox("Race/Ethnicity", ["group A", "group B", "group C", "group D", "group E"])
    parental_level_of_education = st.selectbox("Parental Level of Education", [
        "some high school", "high school", "some college", "associate's degree", "bachelor's degree", "master's degree"
    ])
    lunch = st.selectbox("Lunch", ["standard", "free/reduced"])
    test_preparation_course = st.selectbox("Test Preparation Course", ["none", "completed"])
    reading_score = st.number_input("Reading Score", min_value=0, max_value=100, value=50)
    writing_score = st.number_input("Writing Score", min_value=0, max_value=100, value=50)

    submit_button = st.form_submit_button(label='Predict Math Score')

if submit_button:
    # Create data
    input_data = CustomData(
        gender=gender,
        race_ethnicity=ethnicity,
        parental_level_of_education=parental_level_of_education,
        lunch=lunch,
        test_preparation_course=test_preparation_course,
        reading_score=reading_score,
        writing_score=writing_score
    )

    df = input_data.get_data_as_data_frame()

    # Predict
    pipeline = PredictPipeline()
    result = pipeline.predict(df)

    st.success(f" Predicted Math Score: {result[0]:.2f}")

