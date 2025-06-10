from flask import Flask, render_template, request
import pickle
import numpy as np

# Load the pre-trained model
loaded_model = pickle.load(open('random_forest_regression_model.pkl', 'rb'))

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get user input from the form
        T = float(request.form['T'])
        TM = float(request.form['TM'])
        Tm = float(request.form['Tm'])
        SLP = float(request.form['SLP'])
        H = float(request.form['H'])
        VV = float(request.form['VV'])
        V = float(request.form['V'])
        VM = float(request.form['VM'])
        
        # Prepare the data in the right format for prediction
        input_data = np.array([[T, TM, Tm, SLP, H, VV, V, VM]])
        
        # Make prediction
        prediction = loaded_model.predict(input_data)
        
        # Return the result page with the prediction
        return render_template('result.html', prediction=prediction[0])
    
    except Exception as e:
        return f"Error: {e}"

if __name__ == '__main__':
    app.run(debug=True)
