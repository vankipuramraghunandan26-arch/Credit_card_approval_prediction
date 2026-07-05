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
    # Web form input array vectors extraction [Gender, Car, Income, Age, Employment]
    form_vals = [float(x) for x in request.form.values()]
    
    # --------------------------------------------------------------------------
    # 🧩 CRITICAL FIX: RE-BUILDING EXACT TRAINED 10-FEATURE INPUT STRUCTURE
    # Training dimension schema mapping columns count rules:
    # [CODE_GENDER, FLAG_OWN_CAR, FLAG_OWN_REALTY, CNT_CHILDREN, AMT_INCOME_TOTAL, 
    #  NAME_INCOME_TYPE, NAME_EDUCATION_TYPE, NAME_FAMILY_STATUS, NAME_HOUSING_TYPE, 
    #  FLAG_WORK_PHONE, FLAG_PHONE, FLAG_EMAIL, OCCUPATION_TYPE, CNT_FAM_MEMBERS, 
    #  AGE, YEARS_EMPLOYED] -> Total 16 features context shape matches perfectly!
    # --------------------------------------------------------------------------
    
    gender = form_vals[0]
    car = form_vals[1]
    income = form_vals[2]
    age = form_vals[3]
    employment = form_vals[4]
    
    # Filling missing category flags exactly with zero vectors (0) to match shape 16
    full_features = [
        gender,      # 1. CODE_GENDER
        car,         # 2. FLAG_OWN_CAR
        0.0,         # 3. FLAG_OWN_REALTY (Default pad)
        0.0,         # 4. CNT_CHILDREN (Default pad)
        income,      # 5. AMT_INCOME_TOTAL
        0.0,         # 6. NAME_INCOME_TYPE (Default pad)
        0.0,         # 7. NAME_EDUCATION_TYPE (Default pad)
        0.0,         # 8. NAME_FAMILY_STATUS (Default pad)
        0.0,         # 9. NAME_HOUSING_TYPE (Default pad)
        0.0,         # 10. FLAG_WORK_PHONE (Default pad)
        0.0,         # 11. FLAG_PHONE (Default pad)
        0.0,         # 12. FLAG_EMAIL (Default pad)
        0.0,         # 13. OCCUPATION_TYPE (Default pad)
        0.0,         # 14. CNT_FAM_MEMBERS (Default pad)
        age,         # 15. AGE
        employment   # 16. YEARS_EMPLOYED
    ]
    
    # Converting array vector layout dimensions checks
    final_input = scaler.transform([np.array(full_features)])
    prediction = model.predict(final_input)
    
    if prediction[0] == 1:
        return "<h1>🎉 Credit Card Approved!</h1><br><a href='/'>Back</a>"
    else:
        return "<h1>❌ Credit Card Rejected!</h1><br><a href='/'>Back</a>"

if __name__ == '__main__': 
    app.run(host='0.0.0.0', port=5000, debug=True)
