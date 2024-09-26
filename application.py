import pickle
from flask import Flask, request, render_template
import numpy as np

application = Flask(__name__)
app = application

# Load the models
random_forest_model = pickle.load(open('models/rf.pkl', 'rb'))
scaler_model = pickle.load(open('models/scaler.pkl', 'rb'))

@app.route("/")
def index():
    return render_template('home.html')

@app.route("/predictquality", methods=['GET', 'POST'])
def predict_quality():
    if request.method == 'POST':
        try:
            fixed_acidity = float(request.form.get('fixed_acidity'))
            volatile_acidity = float(request.form.get('volatile_acidity'))
            citric_acid = float(request.form.get('citric_acid'))
            residual_sugar = float(request.form.get('residual_sugar'))
            chlorides = float(request.form.get('chlorides'))
            free_sulfur_dioxide = float(request.form.get('free_sulfur_dioxide'))
            total_sulfur_dioxide = float(request.form.get('total_sulfur_dioxide'))
            density = float(request.form.get('density'))
            pH = float(request.form.get('pH'))
            sulphates = float(request.form.get('sulphates'))
            alcohol = float(request.form.get('alcohol'))
        except ValueError as e:
            print(f"ValueError: {e}")
            return "Invalid input, please enter numeric values.", 400
        
        # Scaling the input data
        new_data_scaled = scaler_model.transform([[fixed_acidity, volatile_acidity, citric_acid, residual_sugar,
                                                    chlorides, free_sulfur_dioxide, total_sulfur_dioxide, 
                                                    density, pH, sulphates, alcohol]])
        result = random_forest_model.predict(new_data_scaled)
        result = result.astype(int)

        return render_template('home.html', results=result[0])
    else:
        return render_template("home.html")




if __name__ == "__main__":
    application.run(host="0.0.0.0")