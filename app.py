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
    features = [float(x) for x in request.form.values()]
    final_input = scaler.transform([np.array(features)])
    prediction = model.predict(final_input)
    
    if prediction[0] == 1:
        return "<h1>🎉 Output Result: Credit Card Approved!</h1><br><a href='/'>Back</a>"
    else:
        return "<h1>❌ Output Result: Credit Card Rejected!</h1><br><a href='/'>Back</a>"

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)