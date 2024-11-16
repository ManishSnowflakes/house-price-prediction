from flask import Flask, request, jsonify
import joblib
import numpy as np

# Load the trained model and scaler
model = joblib.load('./src/house_price_predictor_model.pkl')
scaler = joblib.load('./src/scaler.pkl')

# Initialize Flask app
app = Flask(__name__)

# Define prediction endpoint
@app.route('/')
def home():
    return "Welcome to the House Price Prediction API! Use the '/predict' endpoint to make predictions."

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get data from POST request
        data = request.get_json()

        # Extract and scale features
        total_area = data['Total_Area']
        price_per_sqft = data['Price_per_SQFT']
        baths = data['Baths']
        input_data = np.array([[total_area, price_per_sqft, baths]])
        scaled_data = scaler.transform(input_data)

        # Make prediction
        prediction = model.predict(scaled_data)

        # Return prediction result as JSON
        return jsonify({'predicted_price': prediction[0]})

    except Exception as e:
        return jsonify({'error': str(e)})

# Run the app
if __name__ == '__main__':
    app.run(port=5005)
