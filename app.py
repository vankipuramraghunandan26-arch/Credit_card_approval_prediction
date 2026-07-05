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
    # 1. Fetch normal baseline values directly from web UI elements
    g = float(request.form['g'])
    car = float(request.form['car'])
    realty = float(request.form['realty'])
    inc = float(request.form['inc'])
    inc_type = float(request.form['inc_type'])
    edu = float(request.form['edu'])
    fam = float(request.form['fam'])
    house = float(request.form['house'])
    fam_mem = float(request.form['fam_mem'])
    
    # 2. AUTOMATIC MATHEMATICAL TRANSLATION FOR USER COMPATIBILITY
    # User entering real age (e.g., 30) -> converted to dataset format reference scale
    user_age = float(request.form['age_years'])
    age_scaled = user_age * 0.9778  # Maps 30 years near ~29.334 dataset ratio reference

    # User entering work experience in years (e.g., 5) -> converted to dataset format
    user_exp = float(request.form['exp_years'])
    if user_exp == 0:
        ye_scaled = 0.0
    else:
        ye_scaled = -(user_exp * 0.4191) # Maps 5 years experience precisely near ~ -2.095 ratio scale
    
    # 3. CONSTRUCTING FINAL 16 FEATURES VECTOR MATRIX FOR SCALER MODEL PROCESSING
    full_features = [
        g,           # 1. CODE_GENDER
        car,         # 2. FLAG_OWN_CAR
        realty,      # 3. FLAG_OWN_REALTY
        0.0,         # 4. CNT_CHILDREN (Default padding)
        inc,         # 5. AMT_INCOME_TOTAL
        inc_type,    # 6. NAME_INCOME_TYPE
        edu,         # 7. NAME_EDUCATION_TYPE
        fam,         # 8. NAME_FAMILY_STATUS
        house,       # 9. NAME_HOUSING_TYPE
        0.0,         # 10. FLAG_WORK_PHONE (Default padding)
        0.0,         # 11. FLAG_PHONE (Default padding)
        0.0,         # 12. FLAG_EMAIL (Default padding)
        0.0,         # 13. OCCUPATION_TYPE (Default padding)
        fam_mem,     # 14. CNT_FAM_MEMBERS
        age_scaled,  # 15. AGE (Converted internal scale)
        ye_scaled    # 16. YEARS_EMPLOYED (Converted internal scale)
    ]
    
    # Executing machine learning calculations matrix operations
    final_input = scaler.transform([np.array(full_features)])
    prediction = model.predict(final_input)
    
    if prediction[0] == 1:
        return "<h1> Credit Card Approved!</h1><br><a href='/'>Back</a>"
    else:
        return "<h1> Credit Card Rejected!</h1><br><a href='/'>Back</a>"

if __name__ == '__main__': 
    app.run(host='0.0.0.0', port=5000, debug=True)
