from flask import Flask, request, jsonify
import numpy as np
from PIL import Image
import tensorflow as tf
from flask_cors import CORS
import gdown
import os
import zipfile
import io
from dotenv import load_dotenv
load_dotenv()

Model_ID = os.getenv('Model_ID')

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Function to download the model from Google Drive
def download_model():
    # URL of the Google Drive file
    url = "https://drive.google.com/uc?export=download&id="+Model_ID
    # Download the file
    output_path = "./saved_model.zip"
    gdown.download(url, output_path, quiet=False)
    # Extract the downloaded ZIP file
    with zipfile.ZipFile(output_path, 'r') as zip_ref:
        zip_ref.extractall("./")
    # Delete the ZIP file after extraction
    os.remove(output_path)

# Load the pre-trained model from SavedModel format
def load_model():
    if not hasattr(load_model, "model"):
        # Check if the saved_model folder exists
        if not os.path.exists("./saved_model"):
            # If not, download the model
            download_model()
        # Load the model
        loaded_model = tf.saved_model.load('./saved_model')
        # Extract the specific model for inference
        load_model.model = loaded_model.signatures["serving_default"]

# Define function to preprocess the image
def preprocess_image(image):
    # Resize image to the required input size of the model
    image = image.resize((128, 128))
    # Convert image to numpy array
    image_array = np.array(image) / 255.0  # Normalize pixel values
    # Add batch dimension
    image_array = np.expand_dims(image_array, axis=0)
    image_tensor = tf.convert_to_tensor(image_array, dtype=tf.float32)
    return image_tensor

@app.route('/')
def home():
    return 'Welcome to the Diabetic Retinopathy Detection API!'

# Define route for API
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Load the model
        load_model()
        # Check if request contains file
        if 'file' not in request.files:
            return jsonify({'error': 'No file part'})

        # Get file from request
        file = request.files['file']

        # Check if file is empty
        if file.filename == '':
            return jsonify({'error': 'No selected file'})

        # Read image file
        image = Image.open(io.BytesIO(file.read()))
        # Preprocess image
        processed_image = preprocess_image(image)
        # Make prediction using the model
        output = load_model.model(vgg16_input=processed_image)  # Use 'vgg16_input' as the input name

        # Get predicted probabilities
        probabilities = output['dense_1'][0].numpy()
        # Get predicted labels
        predicted_labels = ['Mild DR', 'Moderate DR', 'No DR', 'Proliferative DR', 'Severe DR']

        # Format response with predicted labels and probabilities
        response = {'predictions': {label: float(probability) for label, probability in zip(predicted_labels, probabilities)}}

        return jsonify(response)
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)
