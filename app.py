import numpy as np
import pickle
from flask import Flask, request, render_template

app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))
scaler = pickle.load(open('scaler.pkl', 'rb'))

@app.route('/')
def home(): return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    form_vals = [float(x) for x in request.form.values()]
    # Building exact shape input dimension match array vector (14 zeroes pad for other encoded fields)
    full_features = form_vals[:3] + [0]*9 + form_vals[3:]
    final_input = scaler.transform([np.array(full_features)])
    prediction = model.predict(final_input)
    return "<h1>🎉 Credit Card Approved!</h1>" if prediction[0] == 1 else "<h1>❌ Credit Card Rejected!</h1>"

if __name__ == '__main__': app.run(host='0.0.0.0', port=5000, debug=True)