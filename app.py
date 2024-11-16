from flask import Flask, request, jsonify
import os
import joblib
import numpy as np
import logging

# Initialize Flask app
app = Flask(__name__)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Load environment configurations (Optional: SECRET_KEY, DEBUG, etc.)
app.config['SECRET_KEY'] = os.getenv('SECRET_KEY', 'default-secret-key')

# Load the trained model and scaler
try:
    logger.info("Loading model and scaler...")
    model = joblib.load('./src/house_price_predictor_model.pkl')
    scaler = joblib.load('./src/scaler.pkl')
    logger.info("Model and scaler loaded successfully.")
except Exception as e:
    logger.error(f"Error loading model/scaler: {e}")
    raise e

# Define routes
@app.route('/')
def home():
    logger.info("Home route accessed.")
    return "Welcome to the House Price Prediction API! Use the '/predict' endpoint to make predictions."

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get data from POST request
        data = request.get_json()

        # Validate input data
        required_fields = ['Total_Area', 'Price_per_SQFT', 'Baths']
        for field in required_fields:
            if field not in data:
                raise ValueError(f"Missing field: {field}")

        # Extract and scale features
        total_area = data['Total_Area']
        price_per_sqft = data['Price_per_SQFT']
        baths = data['Baths']
        input_data = np.array([[total_area, price_per_sqft, baths]])
        scaled_data = scaler.transform(input_data)

        # Make prediction
        prediction = model.predict(scaled_data)

        # Return prediction result as JSON
        return jsonify({'predicted_price': float(prediction[0])})

    except ValueError as ve:
        logger.error(f"Validation Error: {ve}")
        return jsonify({'error': str(ve)}), 400

    except Exception as e:
        logger.error(f"Error during prediction: {e}")
        return jsonify({'error': 'An error occurred during prediction. Please check your input and try again.'}), 500

# Run the app
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5005)
