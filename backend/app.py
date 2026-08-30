# Import necessary libraries
import joblib  # For loading the serialized model
import pandas as pd  # For data manipulation
from flask import Flask, request, jsonify  # For creating the Flask API

# Initialize the Flask application
lead_conversion_predictor_api = Flask("ExtraaLearn Lead Conversion Predictor")

# Load the trained machine learning pipeline (preprocessing + model)
model = joblib.load("extraalearn_lead_conversion_model_v1_0.joblib")

# Features expected by the model (names/order must match those used during training)
FEATURES = [
    "age", "current_occupation", "first_interaction", "profile_completed",
    "website_visits", "time_spent_on_website", "page_views_per_visit",
    "last_activity", "print_media_type1", "print_media_type2",
    "digital_media", "educational_channels", "referral",
]


# Define a route for the home page (GET request)
@lead_conversion_predictor_api.get('/')
def home():
    """Handle GET requests to the root URL and return a welcome message."""
    return "Welcome to the ExtraaLearn Lead Conversion Prediction API!"


# Define an endpoint for single-lead prediction (POST request)
@lead_conversion_predictor_api.post('/v1/lead')
def predict_lead():
    """
    Handle POST requests to '/v1/lead'. Expects a JSON payload with a single lead's
    features and returns the predicted conversion label and probability as JSON.
    """
    # Get the JSON data from the request body
    lead_data = request.get_json()

    # Extract the relevant features from the JSON data
    sample = {feature: lead_data[feature] for feature in FEATURES}

    # Convert the extracted data into a Pandas DataFrame
    input_data = pd.DataFrame([sample])

    # Make the prediction and estimate the conversion probability
    prediction = int(model.predict(input_data)[0])
    probability = round(float(model.predict_proba(input_data)[0, 1]), 4)
    # int()/float() conversions are needed because NumPy types are not JSON serializable

    # Return the result as a JSON response
    return jsonify({'prediction': prediction, 'conversion_probability': probability})


# Define an endpoint for batch prediction (POST request)
@lead_conversion_predictor_api.post('/v1/leadbatch')
def predict_lead_batch():
    """
    Handle POST requests to '/v1/leadbatch'. Expects a CSV file containing several
    leads and returns the predicted conversion label for each as a JSON dictionary.
    """
    # Get the uploaded CSV file from the request
    file = request.files['file']

    # Read the CSV file into a Pandas DataFrame
    input_data = pd.read_csv(file)

    # Preserve an identifier column ('ID') if present, otherwise use the row index
    if 'ID' in input_data.columns:
        ids = input_data['ID'].tolist()
    else:
        ids = input_data.index.tolist()

    # Make predictions for all leads in the DataFrame
    predictions = model.predict(input_data[FEATURES]).tolist()

    # Create a dictionary of predictions keyed by lead ID
    output_dict = {str(i): int(p) for i, p in zip(ids, predictions)}

    # Return the predictions dictionary as a JSON response
    return output_dict


# Run the Flask application in debug mode if this script is executed directly
if __name__ == '__main__':
    lead_conversion_predictor_api.run(debug=True)
