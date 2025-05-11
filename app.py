from flask import Flask, request, jsonify
import numpy as np
import os
from flask_cors import CORS

app = Flask(__name__)
CORS(app)


try:
    import joblib
    model_path = 'backward_gaussian_nb_model.joblib'
    if os.path.exists(model_path):
        model = joblib.load(model_path)
    else:
        raise FileNotFoundError("Model not found")
except Exception as e:
    print(f"Error loading model: {str(e)}")
    # Create a simple model that always returns 0
    from sklearn.naive_bayes import GaussianNB
    model = GaussianNB()
    model.fit(np.array([[0, 0, 0, 0]]), np.array([0]))

@app.route('/', methods=['GET'])
def home():
    return jsonify({
        'status': 'online',
        'message': 'API is running. Send POST request to /predict endpoint.'
    })

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({
                'success': False,
                'error': 'No data provided'
            }), 400
            
        required_fields = ['age', 'duration', 'campaign', 'previous']
        for field in required_fields:
            if field not in data:
                return jsonify({
                    'success': False,
                    'error': f'Missing required field: {field}'
                }), 400
        
        age = float(data.get('age'))
        duration = float(data.get('duration'))
        campaign = float(data.get('campaign'))
        previous = float(data.get('previous'))
        
        input_data = np.array([[age, duration, campaign, previous]])
        
        prediction = model.predict(input_data)
        prediction_proba = model.predict_proba(input_data)[0].tolist() if hasattr(model, 'predict_proba') else None
        
        response = {
            'success': True,
            'prediction': int(prediction[0]),
            'prediction_label': 'yes' if prediction[0] == 1 else 'no'
        }
        
        if prediction_proba:
            response['probability'] = {
                'no': prediction_proba[0],
                'yes': prediction_proba[1]
            }
            
        return jsonify(response)
    
    except ValueError as e:
        return jsonify({
            'success': False,
            'error': f'Invalid input format: {str(e)}'
        }), 400
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

if __name__ == '__main__':
    app.run(debug=False)
else:
    application = app



