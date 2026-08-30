import streamlit as st
import pandas as pd
import requests

# Base URL of the Flask backend (resolved via Docker network container name)
BACKEND_URL = "http://backend:7860"

# Set the title of the Streamlit app
st.title("ExtraaLearn - Lead Conversion Prediction")
st.write(
    "Predict whether a lead is likely to convert to a paid customer. "
    "Fill in the lead details below and click **Predict**."
)

# Section for online prediction
st.subheader("Online Prediction")

# Collect user input for lead features
age = st.number_input("Age", min_value=18, max_value=65, value=35, step=1)
current_occupation = st.selectbox("Current occupation", ["Professional", "Unemployed", "Student"])
first_interaction = st.selectbox("First interaction", ["Website", "Mobile App"])
profile_completed = st.selectbox("Profile completed", ["Low", "Medium", "High"])
website_visits = st.number_input("Website visits", min_value=0, max_value=100, value=3, step=1)
time_spent_on_website = st.number_input("Time spent on website (seconds)", min_value=0, max_value=10000, value=300, step=10)
page_views_per_visit = st.number_input("Page views per visit", min_value=0.0, max_value=50.0, value=3.0, step=0.1)
last_activity = st.selectbox("Last activity", ["Email Activity", "Phone Activity", "Website Activity"])
print_media_type1 = st.selectbox("Saw Newspaper ad (print_media_type1)", ["Yes", "No"])
print_media_type2 = st.selectbox("Saw Magazine ad (print_media_type2)", ["Yes", "No"])
digital_media = st.selectbox("Saw digital media ad", ["Yes", "No"])
educational_channels = st.selectbox("Heard via educational channels", ["Yes", "No"])
referral = st.selectbox("Came through referral", ["Yes", "No"])

# Convert user input into a DataFrame (one row)
input_data = pd.DataFrame([{
    "age": age,
    "current_occupation": current_occupation,
    "first_interaction": first_interaction,
    "profile_completed": profile_completed,
    "website_visits": website_visits,
    "time_spent_on_website": time_spent_on_website,
    "page_views_per_visit": page_views_per_visit,
    "last_activity": last_activity,
    "print_media_type1": print_media_type1,
    "print_media_type2": print_media_type2,
    "digital_media": digital_media,
    "educational_channels": educational_channels,
    "referral": referral,
}])

# Make prediction when the "Predict" button is clicked
if st.button("Predict", type="primary"):
    response = requests.post(
        f"{BACKEND_URL}/v1/lead", json=input_data.to_dict(orient="records")[0]
    )  # Send data to the Flask API
    if response.status_code == 200:
        result = response.json()
        pred = result["prediction"]
        proba = result["conversion_probability"]
        if pred == 1:
            st.success(f"Lead is LIKELY to convert (probability = {proba:.2%}).")
        else:
            st.info(f"Lead is UNLIKELY to convert (probability = {proba:.2%}).")
    else:
        st.error("Unable to connect to the prediction API.")

# Section for batch prediction
st.subheader("Batch Prediction")

# Allow users to upload a CSV file for batch prediction
uploaded_file = st.file_uploader("Upload CSV file for batch prediction", type=["csv"])

# Make batch prediction when the "Predict Batch" button is clicked
if uploaded_file is not None:
    if st.button("Predict Batch", type="primary"):
        response = requests.post(
            f"{BACKEND_URL}/v1/leadbatch", files={"file": uploaded_file}
        )  # Send file to the Flask API
        if response.status_code == 200:
            predictions = response.json()
            st.success("Batch predictions completed!")
            st.write(predictions)  # Display the predictions
        else:
            st.error("Unable to connect to the prediction API.")
