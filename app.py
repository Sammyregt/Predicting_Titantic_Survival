from flask import Flask, render_template, request
import numpy as np
import joblib as jb
import pandas as pd
from pathlib import Path

# Initialize Flask app
app = Flask(__name__)

# Load trained model (pipeline) using file-relative path
#BASE_DIR = Path(__file__).resolve().parent  # Points to app/
MODEL_PATH = Path("model") / "logistic_regression_model.joblib"
SCALER_PATH = Path("model") / "scaler.joblib"

# Ensure model exists
if not MODEL_PATH.exists() or not SCALER_PATH.exists():
    raise FileNotFoundError(f"Model file not found at {MODEL_PATH}")

model = jb.load(MODEL_PATH)
scaler = jb.load(SCALER_PATH)

#columns expected by the model
NUMERIC_COLS = ['Pclass', 'Age', 'SibSp', 'Parch', 'Fare']
SEX_COLUMNS = ['Sex_male']
EMBARKED_COLUMNS = ['Embarked_Q', 'Embarked_S']

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
                'Pclass': [pclass],
                'Sex': [sex],
                'Age': [age],
                'SibSp': [sibsp],
                'Parch': [parch],
                'Fare': [fare],
                'Embarked': [embarked]
            })

            # -- Onehoting categorical variables --
            sex_dummies = pd.get_dummies(input_df['Sex'], prefix='Sex', drop_first=True)
            for col in SEX_COLUMNS:
                if col not in sex_dummies:
                    sex_dummies[col] = 0

            embarked_dummies = pd.get_dummies(input_df['Embarked'], prefix='Embarked', drop_first=True)
            for col in EMBARKED_COLUMNS:
                if col not in embarked_dummies:
                    embarked_dummies[col] = 0

            input_df = input_df.drop(['Sex', 'Embarked'], axis=1)

            # Concatenate one-hot encoded columns
            input_df = pd.concat([input_df, sex_dummies, embarked_dummies], axis=1)

            # --- Scale numeric columns ---
            input_df[NUMERIC_COLS] = scaler.transform(input_df[NUMERIC_COLS])

            # Ensure all expected columns are present
            expected_columns = NUMERIC_COLS + SEX_COLUMNS + EMBARKED_COLUMNS
            input_df = input_df[expected_columns]

            # Make prediction
            prediction = model.predict(input_df)[0]

        except Exception as e:
            prediction = f"Error: {str(e)}"

    return render_template("index.html", prediction=prediction)


if __name__ == "__main__":
    print("Starting Flask server...")
    print("Open this link in your browser: http://127.0.0.1:5000")
    app.run(host="0.0.0.0", port=5000, debug=True)
