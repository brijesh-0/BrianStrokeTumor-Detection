from flask import Flask, render_template, request, jsonify
import os
import sys
from flask_cors import CORS

# Add the parent directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from text_model.text_model1 import predict

app = Flask(__name__)
CORS(app)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/submit', methods=['POST'])
def submit():
    # Extract data from the request
    gender = int(request.form.get('gender'))  # Convert gender to int
    age = int(request.form.get('age'))  # Age as int
    hypertension = int(request.form.get('hypertension'))  # Hypertension as int
    heart_disease = int(request.form.get('heart_disease'))  # Heart Disease as int
    ever_married = int(request.form.get('ever_married'))  # Ever Married as int
    work_type = int(request.form.get('work_type'))  # Work Type as int
    residence_type = int(request.form.get('Residence_type'))  # Residence Type as int
    avg_glucose_level = float(request.form.get('avg_glucose_level'))  # Glucose level as float
    bmi = float(request.form.get('bmi'))  # BMI as float
    smoking_status = int(request.form.get('smoking_status'))  # Smoking Status as int

    # Prepare the data for prediction
    data = [
        gender, age, hypertension, heart_disease, 
        ever_married, work_type, residence_type, 
        avg_glucose_level, bmi, smoking_status
    ]

    # Call the predict function
    prediction = predict(data)
    print(prediction)
    message = ''
    if prediction == 0:
        message += "You don't Have Brain Stroke"
    else:
        message += "You Have Brain Stroke"

    return jsonify({"prediction": message, "message": "Data received successfully!"}), 200

if __name__ == '__main__':
    app.run(debug=True)