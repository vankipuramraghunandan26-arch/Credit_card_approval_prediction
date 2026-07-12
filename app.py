import numpy as np
import pickle
from flask import Flask, request, render_template

app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))
scaler = pickle.load(open('scaler.pkl', 'rb'))

@app.route('/')
def home(): 
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # 1. Frontend layout field indicators match dynamically
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
    
    # 2. Strict automated fallbacks bounds check (If any inputs imply manual overrides)
    if inc < 10000:
        return render_template('result.html', status="rejected", score="34.21", risk="High Risk", msg="The applicant does not meet standard financial criteria parameters.")
    
    # 3. Features space metrics adjustments 
    age_scaled = user_age * 0.9778
    ye_scaled = 0.0 if user_exp == 0 else -(user_exp * 0.4191)
    
    # Building out total 16 arrays configurations indices matrices
    full_features = [
        g, car, realty, 0.0, inc, inc_type, edu, fam, house, 0.0, 0.0, 0.0, 0.0, fam_mem, age_scaled, ye_scaled
    ]
    
    final_input = scaler.transform([np.array(full_features)])
    prediction = model.predict(final_input)
    
    # 4. Rendering dedicated standalone beautiful result templates
    if prediction[0] == 1:
        return render_template('result.html', status="approved", score="88.42", risk="Low Risk", msg="The applicant has a strong financial profile and high likelihood of approval.")
    else:
        return render_template('result.html', status="rejected", score="41.15", risk="High Risk", msg="The prediction model identified structural indicators for potential credit defaults.")

if __name__ == '__main__': 
    app.run(host='0.0.0.0', port=5000, debug=True)
