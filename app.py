from flask import Flask, request, jsonify
import numpy as np
import pickle
import os
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

model_path = 'backward_gaussian_nb_model.pkl'
if os.path.exists(model_path):
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
else:
    raise FileNotFoundError("Model not found. Please train the model first.")

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        
        age = float(data.get('age'))
        duration = float(data.get('duration'))
        campaign = float(data.get('campaign'))
        previous = float(data.get('previous'))
        
        input_data = np.array([[age, duration, campaign, previous]])
        
        prediction = model.predict(input_data)
        
        return jsonify({
            'success': True,
            'prediction': int(prediction[0]),
            'prediction_label': 'yes' if prediction[0] == 1 else 'no'
        })
    
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 400

if __name__ == '__main__':
    app.run(debug=True)
