from flask import Flask, request, jsonify
import os
import joblib

app = Flask(__name__)

# Load the model and scaler
model = joblib.load('model/exported_model.h5')
scaler = joblib.load('model/scaler.pkl')

@app.route('/')
def home():
    return app.send_static_file('index.html')

@app.route('/submit', methods=['POST'])
def submit():
    # Extract data from the request
    # Preprocess the input and make predictions
    # Handle file uploads and other data

    return jsonify({"message": "Data received successfully!"}), 200

if __name__ == '__main__':
    app.run(debug=True)
