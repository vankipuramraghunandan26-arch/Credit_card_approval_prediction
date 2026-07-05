@app.route('/predict', methods=['POST'])
def predict():
    # 5 fields form nunchi vastunnayi
    form_vals = [float(x) for x in request.form.values()]
    
    # 16 features ki match ayye vector ni create chesthunnam (missing values ni 0 pettochu)
    # Form values: [g, car, inc, age, ye] (5 values)
    # Mana model ki 16 kavali, so balance 11 values ni 0 ga pad chesthunnam
    full_features = form_vals + [0] * 11 
    
    # Scaler expect chese shape ki transform chesthunnam
    final_input = scaler.transform([np.array(full_features)])
    prediction = model.predict(final_input)
    
    return "<h1>🎉 Credit Card Approved!</h1>" if prediction[0] == 1 else "<h1>❌ Credit Card Rejected!</h1>"
