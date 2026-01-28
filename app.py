"""
Flask Backend for IMDB Movie Review Sentiment Classification
Provides REST API endpoints for sentiment prediction using a pretrained Keras model.
"""

from flask import Flask, request, jsonify
from tensorflow import keras
from flask_cors import CORS
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.datasets import imdb
import numpy as np
import os


# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Global variables for model and word mappings
model = None
word_index = None
reverse_word_index = None

# Constants
VOCAB_SIZE = 10000
UNKNOWN_TOKEN_ID = 2
MAX_SEQUENCE_LENGTH = 500
MODEL_PATH = '/home/vaibhav-mishra/ML/DL/ANN_classification/Imdb_review/simple_rnn_imdb.h5'  # Update this path to your model location


def load_model_and_vocab():
    """
    Load the pretrained model and vocabulary mappings at application startup.
    """
    global model, word_index, reverse_word_index
    
    try:
        # Load the pretrained Keras model
        print(f"Loading model from {MODEL_PATH}...")
        model = keras.models.load_model(MODEL_PATH)
        print("Model loaded successfully!")
        
        # Load IMDB word index
        print("Loading IMDB word index...")
        word_index = imdb.get_word_index()
        
        # Create reverse word index for decoding (if needed for debugging)
        reverse_word_index = {value: key for key, value in word_index.items()}
        print("Vocabulary loaded successfully!")
        
        return True
    except FileNotFoundError:
        print(f"Error: Model file not found at {MODEL_PATH}")
        print("Please ensure the model file exists at the specified path.")
        return False
    except Exception as e:
        print(f"Error loading model or vocabulary: {str(e)}")
        return False


def decode_review(encoded_review):
    """
    Decode an encoded review back to text (for debugging purposes).
    
    Args:
        encoded_review: List of integer token IDs
        
    Returns:
        Decoded text string
    """
    return ' '.join([reverse_word_index.get(i - 3, '?') for i in encoded_review])


def preprocess_text(text):
    """
    Preprocess raw text for model prediction.
    Tokenizes, encodes, and pads the input text.
    
    Args:
        text: Raw review text string
        
    Returns:
        Padded sequence ready for model input
    """
    words = text.lower().split()
    encoded_review = []
    
    for word in words:
        original_idx = word_index.get(word, UNKNOWN_TOKEN_ID)
        shifted_idx = original_idx + 3
        
        if shifted_idx >= VOCAB_SIZE:
            encoded_review.append(UNKNOWN_TOKEN_ID)
        else:
            encoded_review.append(shifted_idx)
    
    padded_review = sequence.pad_sequences([encoded_review], maxlen=MAX_SEQUENCE_LENGTH)
    return padded_review


def predict_sentiment(review):
    """
    Predict sentiment for a given review.
    
    Args:
        review: Raw review text string
        
    Returns:
        Tuple of (sentiment_label, confidence_score)
    """
    preprocessed_input = preprocess_text(review)
    prediction = model.predict(preprocessed_input)
    sentiment = 'Positive' if prediction[0][0] > 0.5 else 'Negative'
    return sentiment, float(prediction[0][0])


# API Routes

@app.route('/', methods=['GET'])
def home():
    """
    Health check endpoint.
    """
    return jsonify({
        'status': 'running',
        'service': 'IMDB Sentiment Classification API',
        'model_loaded': model is not None
    }), 200


@app.route('/predict', methods=['POST'])
def predict():
    """
    Predict sentiment for a movie review.
    
    Expected JSON payload:
    {
        "review": "This movie was fantastic! I loved every moment."
    }
    
    Returns:
    {
        "review": "This movie was fantastic! I loved every moment.",
        "sentiment": "Positive",
        "confidence": 0.95,
        "status": "success"
    }
    """
    try:
        # Check if model is loaded
        if model is None:
            return jsonify({
                'status': 'error',
                'message': 'Model not loaded. Please restart the server.'
            }), 503
        
        # Get review text from request
        data = request.get_json()
        
        if not data:
            return jsonify({
                'status': 'error',
                'message': 'Invalid request. Expected JSON payload.'
            }), 400
        
        review_text = data.get('review', '').strip()
        
        # Validate input
        if not review_text:
            return jsonify({
                'status': 'error',
                'message': 'Review text cannot be empty.'
            }), 400
        
        # Predict sentiment
        sentiment, confidence = predict_sentiment(review_text)
        
        # Prepare response
        response = {
            'status': 'success',
            'review': review_text,
            'sentiment': sentiment,
            'confidence': round(confidence, 4)
        }
        
        return jsonify(response), 200
        
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': f'An error occurred during prediction: {str(e)}'
        }), 500


@app.route('/health', methods=['GET'])
def health():
    """
    Detailed health check endpoint.
    """
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None,
        'vocab_loaded': word_index is not None,
        'vocab_size': VOCAB_SIZE,
        'max_sequence_length': MAX_SEQUENCE_LENGTH
    }), 200


# Application entry point

if __name__ == '__main__':
    print("=" * 60)
    print("IMDB Sentiment Classification API")
    print("=" * 60)
    
    # Load model and vocabulary before starting the server
    if load_model_and_vocab():
        print("\nStarting Flask server...")
        print("API Endpoints:")
        print("  GET  /          - Service status")
        print("  GET  /health    - Detailed health check")
        print("  POST /predict   - Predict sentiment")
        print("=" * 60)
        
        # Run the Flask app
        app.run(debug=True, host='0.0.0.0', port=5000)
    else:
        print("\nFailed to load model or vocabulary.")
        print("Please check the model path and try again.")
        print("=" * 60)