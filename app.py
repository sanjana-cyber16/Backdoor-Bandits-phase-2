from flask import Flask, render_template, send_from_directory, request, jsonify
from flask_cors import CORS
import os
import pickle
import pandas as pd
from werkzeug.security import generate_password_hash, check_password_hash

app = Flask(__name__, static_folder="static", template_folder="templates")
CORS(app)

# Ensure data directory exists
if not os.path.exists('data'):
    os.makedirs('data')

EXCEL_FILE = 'data/registered_users.xlsx'

# Load trained model
with open("model.pkl", "rb") as file:
    model = pickle.load(file)

@app.route('/')
def home():
    return render_template('index.html')  # Serve frontend

@app.route('/api/signup', methods=['POST'])
def signup():
    data = request.get_json()
    email = data['email']
    phone = data['phone']
    password = generate_password_hash(data['password'])

    # Save to Excel
    user_data = {'email': email, 'phone': phone, 'password': password}
    df = pd.DataFrame([user_data])

    if not os.path.exists(EXCEL_FILE):
        df.to_excel(EXCEL_FILE, index=False)
    else:
        existing_df = pd.read_excel(EXCEL_FILE)
        final_df = pd.concat([existing_df, df], ignore_index=True)
        final_df.to_excel(EXCEL_FILE, index=False)

    return jsonify({'message': 'Signup successful'})

@app.route('/api/login', methods=['POST'])
def login():
    data = request.get_json()
    email = data['email']
    password = data['password']

    if os.path.exists(EXCEL_FILE):
        df = pd.read_excel(EXCEL_FILE)
        user = df[df['email'].str.lower() == email.lower()]

        if not user.empty and check_password_hash(user.iloc[0]['password'], password):
            return jsonify({'message': 'Login successful'})

    return jsonify({'error': 'Invalid credentials'})

@app.route('/api/predict', methods=['POST'])
def predict():
    data = request.get_json()
    features = data['features']
    prediction = model.predict([features])
    return jsonify({'prediction': prediction.tolist()})

if __name__ == '__main__':
    app.run(port=5000, debug=True)
