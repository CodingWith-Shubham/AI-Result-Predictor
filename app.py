from flask import Flask, request, jsonify, render_template
import pandas as pd
import joblib

app = Flask(__name__)
model = joblib.load('student_model.pkl')  # Load the trained model

@app.route('/')
def home():
    return render_template('index.html')  # HTML form for input

@app.route('/predict', methods=['POST'])
def predict():
    data = request.form

    # Create dataframe from form input
    input_data = pd.DataFrame([{
        'Hours_Studied': float(data['Hours_Studied']),
        'Sleep_Hours': float(data['Sleep_Hours']),
        'Previous_Scores': float(data['Previous_Scores']),
        'Tutoring_Sessions': int(data['Tutoring_Sessions']),
        'Physical_Activity': int(data['Physical_Activity']),
        'Parental_Involvement_encoded': int(data['Parental_Involvement_encoded']),
        'Access_to_Resources_encoded': int(data['Access_to_Resources_encoded']),
        'Extracurricular_Activities_encoded': int(data['Extracurricular_Activities_encoded']),
        'Motivation_Level_encoded': int(data['Motivation_Level_encoded']),
        'Internet_Access_encoded': int(data['Internet_Access_encoded']),
        'Teacher_Quality_encoded': int(data['Teacher_Quality_encoded']),
        'School_Type_encoded': int(data['School_Type_encoded']),
        'Peer_Influence_encoded': int(data['Peer_Influence_encoded']),
        'Learning_Disabilities_encoded': int(data['Learning_Disabilities_encoded']),
        'Distance_from_Home_encoded': int(data['Distance_from_Home_encoded']),
    }])

    prediction = model.predict(input_data)[0]
    result = "Pass(congrats)" if prediction == 1 else "Back(chud gye)"
    return render_template('index.html', prediction_text=f"Result:{result}")

if __name__ == '__main__':
    app.run(debug=True)
