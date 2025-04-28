from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

# Load the saved model and scaler
with open('rf_model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

with open('scaler.pkl', 'rb') as scaler_file:
    scaler = pickle.load(scaler_file)

@app.route('/')
def home():
    return render_template('index.html')  # Ensure templates/index.html exists

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Collect inputs from the form
        inputs = [
            'patientid', 'age', 'gender', 'ethnicity', 'education_level', 'bmi',
            'smoking', 'alcohol_consumption', 'physical_activity', 'diet_quality',
            'sleep_quality', 'family_history_alzheimers', 'cardiovascular_disease',

            'diabetes', 'depression', 'head_injury', 'hypertension', 'systolic_bp',
            'diastolic_bp', 'cholesterol_total', 'cholesterol_ldl', 'cholesterol_hdl',
            'cholesterol_triglycerides', 'mmse', 'functional_assessment',
            'memory_complaints', 'behavioral_problems', 'adl', 'confusion',
            'disorientation', 'personality_changes', 'difficulty_completing_tasks'
        ]

        # Extract and convert inputs
        data = [request.form.get(field, None) for field in inputs]
        data = list(map(float, data))  # Convert all to float (adjust as needed)

        # Reshape and scale features
        features = np.array([data])
        features_scaled = scaler.transform(features)

        # Make prediction
        prediction = model.predict(features_scaled)
        result = 'Positive for DACD' if prediction == 1 else 'Negative for DACD'

        return f'<h2>Prediction Result: {result}</h2>'

    except Exception as e:
        return f'<h2>Error: {str(e)}</h2>'

if __name__ == '__main__':
    app.run()

