import streamlit as st
import pandas as pd
import joblib


@st.cache_resource
def load_model():
    return joblib.load("extraalearn_lead_conversion_model_v1_0.joblib")


model = load_model()

st.title("ExtraaLearn - Lead Conversion Prediction")
st.write(
    "Predict whether a lead is likely to convert to a paid customer. "
    "Fill in the lead details below and click **Predict**."
)

age = st.slider("Age", 18, 65, 35, 1)
current_occupation = st.selectbox("Current occupation", ["Professional", "Unemployed", "Student"])
first_interaction = st.selectbox("First interaction", ["Website", "Mobile App"])
profile_completed = st.selectbox("Profile completed", ["Low", "Medium", "High"])
website_visits = st.slider("Website visits", 0, 30, 3, 1)
time_spent_on_website = st.slider("Time spent on website (seconds)", 0, 3000, 300, 10)
page_views_per_visit = st.slider("Page views per visit", 0.0, 20.0, 3.0, 0.1)
last_activity = st.selectbox("Last activity", ["Email Activity", "Phone Activity", "Website Activity"])
print_media_type1 = st.selectbox("Saw Newspaper ad (print_media_type1)", ["Yes", "No"])
print_media_type2 = st.selectbox("Saw Magazine ad (print_media_type2)", ["Yes", "No"])
digital_media = st.selectbox("Saw digital media ad", ["Yes", "No"])
educational_channels = st.selectbox("Heard via educational channels", ["Yes", "No"])
referral = st.selectbox("Came through referral", ["Yes", "No"])

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

if st.button("Predict"):
    pred = model.predict(input_data)[0]
    proba = model.predict_proba(input_data)[0, 1]
    if pred == 1:
        st.success(f"Lead is LIKELY to convert. (probability = {proba:.2%})")
    else:
        st.info(f"Lead is UNLIKELY to convert. (probability = {proba:.2%})")
