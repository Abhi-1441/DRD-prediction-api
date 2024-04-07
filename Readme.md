# Diabetic Retinopathy Detection API

## Overview
This project implements an API for detecting diabetic retinopathy in retinal images using a pre-trained machine learning model. The model is capable of classifying retinal images into different categories of diabetic retinopathy severity.

## Model Description
The machine learning model used in this project is based on a pre-trained convolutional neural network (CNN) architecture, specifically VGG16. The model was trained on a dataset of retinal images labeled with diabetic retinopathy severity levels. During training, the model learns to classify retinal images into one of the following categories:
- Mild DR
- Moderate DR
- No DR
- Proliferative DR
- Severe DR

## Techniques Used
- Data preprocessing: The input images are resized to match the required input size of the model and normalized.
- Model architecture: VGG16 convolutional neural network architecture is used.
- Model deployment: The model is deployed as a RESTful API using Flask.
- Image processing: Images are processed using the Python Imaging Library (PIL) before being fed into the model for inference.

## Process
1. Data collection: A dataset of retinal images labeled with diabetic retinopathy severity levels is collected.
2. Data preprocessing: Images are resized and normalized before being used for training the model.
3. Model training: The VGG16 model is trained on the preprocessed retinal image dataset.
4. Model evaluation: The trained model's performance is evaluated using various evaluation metrics.
5. Deployment: The trained model is deployed as a RESTful API using Flask, allowing users to make predictions on new retinal images.

## Libraries Used
- Flask: Web framework for building APIs
- TensorFlow: Machine learning framework
- NumPy: Numerical computing library
- Pillow: Image processing library
- Flask-Cors: Flask extension for handling Cross-Origin Resource Sharing (CORS)
- gdown: Utility for downloading files from Google Drive
- python-dotenv: Python module for reading `.env` files

## Usage
To run the project locally, follow these steps:
1. Install the required libraries using `pip install -r requirements.txt`.
2. Run the Flask application using `gunicorn -w 4 -b 127.0.0.1:5000 index:app`.

Access the API at `https://house-value-predict.onrender.com/predict` to interact with the deployed model.

## Credits
- [Abhishek Pakhmode](https://github.com/Abhi-1441)

## Contact
For questions or feedback, please contact [pakhmodeabhishek1441@gmail.com].
