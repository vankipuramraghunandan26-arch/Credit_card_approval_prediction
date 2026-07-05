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
    # Fetching exact 14 parameters sent from the frontend grid system
    g = float(request.form['g'])
    car = float(request.form['car'])
    realty = float(request.form['realty'])
    inc = float(request.form['inc'])
    inc_type = float(request.form['inc_type'])
    edu = float(request.form['edu'])
    fam = float(request.form['fam'])
    house = float(request.form['house'])
    age = float(request.form['age'])
    ye = float(request.form['ye'])
    fam_mem = float(request.form['fam_mem'])
    
    # --------------------------------------------------------------------------
    # 🧩 PERFECT FIT: Mapping exactly into the 16 features shape expected by scaler
    # Schema columns reference template:
    # 1.CODE_GENDER, 2.FLAG_OWN_CAR, 3.FLAG_OWN_REALTY, 4.CNT_CHILDREN, 5.AMT_INCOME_TOTAL, 
    # 6.NAME_INCOME_TYPE, 7.NAME_EDUCATION_TYPE, 8.NAME_FAMILY_STATUS, 9.NAME_HOUSING_TYPE, 
    # 10.FLAG_WORK_PHONE, 11.FLAG_PHONE, 12.FLAG_EMAIL, 13.OCCUPATION_TYPE, 14.CNT_FAM_MEMBERS, 
    # 15.AGE, 16.YEARS_EMPLOYED
    # --------------------------------------------------------------------------
    full_features = [
        g,          # 1. CODE_GENDER
        car,        # 2. FLAG_OWN_CAR
        realty,     # 3. FLAG_OWN_REALTY
        0.0,        # 4. CNT_CHILDREN (Default pad)
        inc,        # 5. AMT_INCOME_TOTAL
        inc_type,   # 6. NAME_INCOME_TYPE
        edu,        # 7. NAME_EDUCATION_TYPE
        fam,        # 8. NAME_FAMILY_STATUS
        house,      # 9. NAME_HOUSING_TYPE
        0.0,        # 10. FLAG_WORK_PHONE (Default pad)
        0.0,        # 11. FLAG_PHONE (Default pad)
        0.0,        # 12. FLAG_EMAIL (Default pad)
        0.0,        # 13. OCCUPATION_TYPE (Default pad)
        fam_mem,    # 14. CNT_FAM_MEMBERS
        age,        # 15. AGE
        ye          # 16. YEARS_EMPLOYED
    ]
    
    # Transforming data matrix
    final_input = scaler.transform([np.array(full_features)])
    prediction = model.predict(final_input)
    
    if prediction[0] == 1:
        return "<h1> Credit Card Approved!</h1><br><a href='/'>Back</a>"
    else:
        return "<h1>❌ Credit Card Rejected!</h1><br><a href='/'>Back</a>"

if __name__ == '__main__': 
    app.run(host='0.0.0.0', port=5000, debug=True)
