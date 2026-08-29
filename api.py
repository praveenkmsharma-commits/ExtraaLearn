import joblib
import pandas as pd
from flask import Flask, request, jsonify

app = Flask(__name__)

# Load the trained pipeline (preprocessing + model) once at startup.
model = joblib.load("extraalearn_lead_conversion_model_v1_0.joblib")

FEATURES = [
    "age", "current_occupation", "first_interaction", "profile_completed",
    "website_visits", "time_spent_on_website", "page_views_per_visit",
    "last_activity", "print_media_type1", "print_media_type2",
    "digital_media", "educational_channels", "referral",
]


@app.route("/", methods=["GET"])
def home():
    return jsonify({"message": "ExtraaLearn Lead Conversion API is running."})


@app.route("/predict", methods=["POST"])
def predict():
    """Accept a JSON body with lead features and return the prediction(s)."""
    payload = request.get_json(force=True)
    records = payload if isinstance(payload, list) else [payload]
    input_df = pd.DataFrame(records)[FEATURES]
    preds = model.predict(input_df).tolist()
    proba = model.predict_proba(input_df)[:, 1].tolist()
    results = [
        {"prediction": int(p), "conversion_probability": round(pr, 4)}
        for p, pr in zip(preds, proba)
    ]
    return jsonify(results if isinstance(payload, list) else results[0])


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
