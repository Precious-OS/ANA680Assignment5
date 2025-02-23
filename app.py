from flask import Flask, request, jsonify, render_template
import joblib
import numpy as np

# Load the model
model = joblib.load("wine_quality_model.pkl")

# Initialize Flask app
app = Flask(__name__)

# Home page
@app.route("/")
def home():
    return render_template("index.html")

# Prediction API
@app.route("/predict", methods=["POST"])
def predict():
    # Get data from the request
    data = request.json

    # Convert data to numpy array
    features = np.array(data["features"]).reshape(1, -1)

    # Make prediction
    prediction = model.predict(features)

    # Return the prediction
    return jsonify({"prediction": prediction[0]})

# Run the app
if __name__ == "__main__":
    app.run(debug=True)