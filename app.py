from flask import Flask, request, render_template
import joblib
import numpy as np

app = Flask(__name__)
model = joblib.load('random_forest_churn_model.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    features = [float(x) for x in request.form.values()]
    prediction = model.predict([features])
    output = "Will Churn" if prediction[0] == 1 else "Will Not Churn"
    return render_template('index.html', prediction_text=f'Customer {output}')

if __name__ == "__main__":
    app.run(debug=True)
