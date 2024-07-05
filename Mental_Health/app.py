from flask import Flask, request, jsonify
from joblib import load

app = Flask(__name__)

# Load the trained model
model = load('mental_health_model.joblib')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get form data from the request
        data = request.get_json()

        # Prepare data for prediction
        input_data = np.array([
            float(data['age']),
            float(data['cgpa']),
            int(data['anxiety']),
            int(data['panicAttack'])
            # Add more fields as needed
        ]).reshape(1, -1)

        # Make prediction
        prediction = model.predict(input_data)[0]

        # Return the prediction as JSON
        return jsonify({'prediction': prediction})

    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)
