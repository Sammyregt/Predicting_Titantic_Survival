from flask import Flask, render_template, request
import numpy as np
import joblib as jb
import pandas as pd
from pathlib import Path

# Initialize Flask app
app = Flask(__name__)

# Load trained model (pipeline) using file-relative path
BASE_DIR = Path(__file__).resolve().parent  # Points to app/
MODEL_PATH = BASE_DIR / "model" / "logistic_regression_model.joblib"

# Ensure model exists
if not MODEL_PATH.exists():
    raise FileNotFoundError(f"Model file not found at {MODEL_PATH}")

model = jb.load(MODEL_PATH)

# Home route
@app.route("/", methods=["GET", "POST"])
def home():
    prediction = None

    if request.method == "POST":
        try:
            # Get form values
            pclass = int(request.form["pclass"])
            sex = request.form["sex"]
            age = float(request.form["age"])
            sibsp = int(request.form["sibsp"])
            parch = int(request.form["parch"])
            fare = float(request.form["fare"])
            embarked = request.form["embarked"]

            # Convert input to DataFrame (required if using preprocessing pipeline)
            input_df = pd.DataFrame({
                'pclass': [pclass],
                'sex': [sex],
                'age': [age],
                'sibsp': [sibsp],
                'parch': [parch],
                'fare': [fare],
                'embarked': [embarked]
            })

            # Make prediction
            prediction = model.predict(input_df)[0]

        except Exception as e:
            prediction = f"Error: {str(e)}"

    return render_template("index.html", prediction=prediction)


if __name__ == "__main__":
    print("Starting Flask server...")
    print("Open this link in your browser: http://127.0.0.1:5000")
    app.run(host="0.0.0.0", port=5000, debug=True)
