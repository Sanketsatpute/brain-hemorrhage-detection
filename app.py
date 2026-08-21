"""
Brain Hemorrhage Detection Web Application
Flask-based web application for detecting intracranial hemorrhage in CT scan images
"""

import os
import numpy as np
import cv2
from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
from werkzeug.utils import secure_filename
import tensorflow as tf
from tensorflow import keras
from PIL import Image
import io
from datetime import datetime
import json

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Configuration
UPLOAD_FOLDER = os.path.join(os.path.dirname(__file__), 'static', 'uploads')
ALLOWED_EXTENSIONS = {'jpg', 'jpeg', 'png', 'dcm', 'tiff'}
MAX_FILE_SIZE = 50 * 1024 * 1024  # 50 MB

MODEL_PATH = os.path.join(
    os.path.dirname(__file__),
    'model',
    'hemorrhage_model_v2.h5'
)

THRESHOLD = 0.41

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = MAX_FILE_SIZE

# Create upload folder if it doesn't exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Global variable to store the model
model = None


def load_model():
    """Load the trained V2 model"""

    global model

    model_path = MODEL_PATH

    if os.path.exists(model_path):

        try:

            model = keras.models.load_model(
                model_path
            )

            print(
                f"Model loaded successfully from {model_path}"
            )

            return True

        except Exception as e:

            print(
                f"Error loading model: {e}"
            )

            return False

    else:

        print(
            f"Model file not found at {model_path}"
        )

        return False


def allowed_file(filename):
    """Check if file extension is allowed"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def preprocess_image(image_path, target_size=(224, 224)):
    """
    Preprocess CT scan image for MobileNetV2 model prediction.

    Pipeline:
        Grayscale CT image
            ↓
        Resize to 224x224
            ↓
        Convert grayscale to RGB
            ↓
        Convert to float32
            ↓
        Add batch dimension

    MobileNetV2 preprocessing is performed inside the model.
    """

    try:

        # Read image as grayscale
        image = cv2.imread(
            image_path,
            cv2.IMREAD_GRAYSCALE
        )

        if image is None:
            # Fallback to PIL
            pil_image = Image.open(
                image_path
            ).convert('L')

            image = np.array(
                pil_image
            )

        # Resize
        image = cv2.resize(
            image,
            target_size
        )

        # Convert grayscale → RGB
        image = cv2.cvtColor(
            image,
            cv2.COLOR_GRAY2RGB
        )

        # Convert to float32
        image = image.astype(
            np.float32
        )

        # Add batch dimension
        image = np.expand_dims(
            image,
            axis=0
        )

        return image

    except Exception as e:

        print(
            f"Error preprocessing image: {e}"
        )

        return None

def make_prediction(image_path):
    """
    Make prediction on the provided image
    
    Args:
        image_path: Path to the image file
    
    Returns:
        Dictionary with prediction results
    """
    try:
        # Preprocess image
        processed_image = preprocess_image(image_path)
        
        if processed_image is None:
            return {
                'success': False,
                'error': 'Failed to preprocess image',
                'confidence': 0
            }
        
        # Make prediction
        prediction = model.predict(processed_image, verbose=0)
        confidence = float(prediction[0][0])

        print(
            "PREDICTION:",
            os.path.basename(image_path),
            confidence
        )
        
        # Classify result
        has_hemorrhage = confidence >= THRESHOLD
        
        return {
            'success': True,
            'has_hemorrhage': has_hemorrhage,
            'confidence': round(confidence * 100, 2),
            'result': 'Hemorrhage Detected' if has_hemorrhage else 'No Hemorrhage Detected'
        }
    
    except Exception as e:
        print(f"Error making prediction: {e}")
        return {
            'success': False,
            'error': str(e),
            'confidence': 0
        }


def classify_risk(confidence):
    """Classify risk level based on model threshold"""

    if confidence >= THRESHOLD:
        return 'High Risk'
    else:
        return 'Low Risk'


# Routes

@app.route('/')
def index():
    """Render home page"""
    return render_template('index.html')


@app.route('/api/upload', methods=['POST'])
def upload_file():
    """Handle file upload and prediction"""
    try:
        # Check if file is in request
        if 'file' not in request.files:
            return jsonify({
                'success': False,
                'error': 'No file provided'
            }), 400
        
        file = request.files['file']
        
        if file.filename == '':
            return jsonify({
                'success': False,
                'error': 'No file selected'
            }), 400
        
        if not allowed_file(file.filename):
            return jsonify({
                'success': False,
                'error': f'File type not allowed. Allowed types: {", ".join(ALLOWED_EXTENSIONS)}'
            }), 400
        
        # Save file
        filename = secure_filename(f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{file.filename}")
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # Make prediction
        prediction = make_prediction(filepath)
        
        if prediction['success']:
            prediction['filename'] = filename
            prediction['upload_time'] = datetime.now().isoformat()
        
        return jsonify(prediction)
    
    except Exception as e:
        return jsonify({
            'success': False,
            'error': f'Server error: {str(e)}'
        }), 500


@app.route('/api/predict', methods=['POST'])
def predict():
    """Make prediction on uploaded image"""
    try:
        data = request.get_json()
        
        if 'filename' not in data:
            return jsonify({
                'success': False,
                'error': 'No filename provided'
            }), 400
        
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(data['filename']))
        
        if not os.path.exists(filepath):
            return jsonify({
                'success': False,
                'error': 'File not found'
            }), 404
        
        prediction = make_prediction(filepath)
        return jsonify(prediction)
    
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/api/health', methods=['GET'])
def health():
    """Health check endpoint"""
    model_loaded = model is not None
    return jsonify({
        'status': 'healthy',
        'model_loaded': model_loaded,
        'timestamp': datetime.now().isoformat()
    })


@app.route('/api/info', methods=['GET'])
def info():
    """Get application info"""
    return jsonify({
        'app_name': 'Brain Hemorrhage Detection System',
        'version': '1.0.0',
        'description': 'AI-powered system for detecting intracranial hemorrhage in CT scan images',
        'supported_formats': list(ALLOWED_EXTENSIONS),
        'max_file_size_mb': MAX_FILE_SIZE / (1024 * 1024),
        'model_loaded': model is not None
    })


@app.errorhandler(413)
def request_entity_too_large(error):
    """Handle file too large error"""
    return jsonify({
        'success': False,
        'error': f'File too large. Maximum size: {MAX_FILE_SIZE / (1024 * 1024)}MB'
    }), 413


@app.errorhandler(404)
def not_found(error):
    """Handle 404 errors"""
    return jsonify({
        'success': False,
        'error': 'Endpoint not found'
    }), 404


@app.errorhandler(500)
def internal_error(error):
    """Handle 500 errors"""
    return jsonify({
        'success': False,
        'error': 'Internal server error'
    }), 500


if __name__ == '__main__':
    # Load model on startup
    if load_model():
        print("✓ Model loaded successfully")
    else:
        print("✗ Failed to load model. Please run train_model.py first")
    
    # Start Flask app
    app.run(debug=True, host='0.0.0.0', port=5000)
