from flask import Flask, request, render_template
import numpy as np
import joblib

app = Flask(__name__)

# Load trained model
model = joblib.load('random_forest_model.pkl')

# Min-Max vectors from training
minVec = {
    'CreditScore': 350,
    'Age': 18,
    'Tenure': 0,
    'Balance': 0.0,
    'NumOfProducts': 1,
    'EstimatedSalary': 10.0
}

maxVec = {
    'CreditScore': 850,
    'Age': 92,
    'Tenure': 10,
    'Balance': 250000.0,
    'NumOfProducts': 4,
    'EstimatedSalary': 200000.0
}

def scale_feature(val, feature):
    return (val - minVec[feature]) / (maxVec[feature] - minVec[feature])

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Raw inputs from form
        CreditScore = float(request.form['CreditScore'])
        Geography = int(request.form['Geography'])   # 0: France, 1: Germany, 2: Spain
        Gender = int(request.form['Gender'])         # 0: Female, 1: Male
        Age = float(request.form['Age'])
        Tenure = float(request.form['Tenure'])
        Balance = float(request.form['Balance'])
        NumOfProducts = float(request.form['NumOfProducts'])
        HasCrCard = int(request.form['HasCrCard'])
        IsActiveMember = int(request.form['IsActiveMember'])
        EstimatedSalary = float(request.form['EstimatedSalary'])

        # Manual one-hot encoding
        geo = [-1, -1, -1]
        geo[Geography] = 1

        gender = [-1, -1]
        gender[Gender] = 1

        # Apply Min-Max Scaling
        CreditScore_scaled = scale_feature(CreditScore, 'CreditScore')
        Age_scaled = scale_feature(Age, 'Age')
        Tenure_scaled = scale_feature(Tenure, 'Tenure')
        Balance_scaled = scale_feature(Balance, 'Balance')
        NumOfProducts_scaled = scale_feature(NumOfProducts, 'NumOfProducts')
        EstimatedSalary_scaled = scale_feature(EstimatedSalary, 'EstimatedSalary')

        # Engineered features
        BalanceSalaryRatio = Balance / EstimatedSalary if EstimatedSalary != 0 else 0
        TenureByAge = Tenure / Age if Age != 0 else 0
        CreditScoreGivenAge = CreditScore / Age if Age != 0 else 0

        # Final list of 16 features
        final_features = [
            CreditScore_scaled
        ] + geo + gender + [
            Age_scaled, Tenure_scaled, Balance_scaled, NumOfProducts_scaled,
            HasCrCard, IsActiveMember, EstimatedSalary_scaled,
            BalanceSalaryRatio, TenureByAge, CreditScoreGivenAge
        ]

        prediction = model.predict([final_features])[0]
        output = "Will Churn" if prediction == 1 else "Will Not Churn"

        return render_template('index.html', prediction_text=f'Customer {output}')

    except Exception as e:
        return f"Error: {str(e)}"

if __name__ == "__main__":
    app.run(debug=True)
