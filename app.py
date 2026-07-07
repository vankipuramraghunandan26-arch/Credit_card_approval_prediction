import numpy as np
import pickle
from flask import Flask, request, render_template

app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))
scaler = pickle.load(open('scaler.pkl', 'rb'))

# Common professional CSS style for centered box output layout
CENTER_BOX_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>Prediction Result</title>
    <style>
        body {{ font-family: Arial, sans-serif; background: #f4f7f6; margin: 0; padding: 0; display: flex; justify-content: center; align-items: center; height: 100vh; }}
        .result-card {{ background: white; padding: 40px; border-radius: 8px; box-shadow: 0 4px 15px rgba(0,0,0,0.1); text-align: center; max-width: 450px; width: 90%; border-top: 5px solid {border_color}; }}
        .icon {{ font-size: 50px; margin-bottom: 15px; }}
        h1 {{ color: #333; font-size: 24px; margin: 10px 0; }}
        p {{ color: #666; font-size: 15px; margin-bottom: 25px; line-height: 1.4; }}
        .back-btn {{ display: inline-block; background: #1890ff; color: white; padding: 10px 25px; text-decoration: none; border-radius: 4px; font-weight: bold; font-size: 14px; transition: background 0.2s; }}
        .back-btn:hover {{ background: #40a9ff; }}
    </style>
</head>
<body>
    <div class="result-card">
        <div class="icon">{icon}</div>
        <h1>{title}</h1>
        <p>{message}</p>
        <a href="/" class="back-btn">Go Back</a>
    </div>
</body>
</html>
"""

@app.route('/')
def home(): 
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # 1. Parsing all inputs from web UI layout fields
    g = float(request.form['g'])
    car = float(request.form['car'])
    realty = float(request.form['realty'])
    inc = float(request.form['inc'])
    inc_type = float(request.form['inc_type'])
    edu = float(request.form['edu'])
    fam = float(request.form['fam'])
    house = float(request.form['house'])
    fam_mem = float(request.form['fam_mem'])
    
    user_age = float(request.form['age_years'])
    user_exp = float(request.form['exp_years'])
    emi_past = float(request.form['emi_past'])
    loans = float(request.form['loans'])
    
    # 2. STRICT FAIL-SAFE LOGIC FOR REJECTED CASES (High Risk Profile)
    if inc < 5000 or emi_past >= 3 or loans >= 8:
        return CENTER_BOX_TEMPLATE.format(
            border_color="#f5222d", 
            icon="❗", 
            title="Credit Card Rejected!", 
            message="Reason: High Risk Profile Detected based on low income or pending defaults."
        )
    
    # 3. INTERNAL MATHEMATICAL MATRIX TRANSLATION
    age_scaled = user_age * 0.9778
    if user_exp == 0:
        ye_scaled = 0.0
    else:
        ye_scaled = -(user_exp * 0.4191)
    
    # 4. COMPILING THE FULL 16 FEATURES FOR MACHINE LEARNING SCALER
    full_features = [
        g, car, realty, 0.0, inc, inc_type, edu, fam, house, 0.0, 0.0, 0.0, 0.0, fam_mem, age_scaled, ye_scaled
    ]
    
    # Running model prediction pipeline
    final_input = scaler.transform([np.array(full_features)])
    prediction = model.predict(final_input)
    
    if prediction[0] == 1:
        return CENTER_BOX_TEMPLATE.format(
            border_color="#52c41a", 
            icon="🎉", 
            title="Credit Card Approved!", 
            message="Congratulations! Your profile successfully satisfies the credit eligibility criteria."
        )
    else:
        return CENTER_BOX_TEMPLATE.format(
            border_color="#f5222d", 
            icon="❗", 
            title="Credit Card Rejected!", 
            message="Sorry, the prediction model determined a mismatch with approval verification standards."
        )

if __name__ == '__main__': 
    app.run(host='0.0.0.0', port=5000, debug=True)
