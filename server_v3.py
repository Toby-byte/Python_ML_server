from flask import Flask, request, jsonify
from flask_cors import CORS
from keras.models import load_model
from keras.preprocessing import image
import tensorflow as tf
import numpy as np
from PIL import Image
import io

# Initialize Flask app and enable CORS
app = Flask(__name__)
CORS(app)

# Load the trained model
model = load_model('model.h5')

# Route for making predictions
@app.route('/predict', methods=['POST'])
def predict():
    file = request.files['file']  # The image file from the request

    img = Image.open(file.stream)  # Open the image
    img = img.resize((64, 64))  # Resize the image

    img_array = tf.keras.preprocessing.image.img_to_array(img)  # Convert image to numpy array
    img_batch = np.expand_dims(img_array, axis=0)  # Reshape for model input

    prediction = model.predict(img_batch / 255.0)  # Make prediction with the model

    # Convert prediction to readable result
    if prediction[0][0] > 0.5:
        prediction_text = 'dog'
    else:
        prediction_text = 'cat'

    return jsonify({"message": prediction_text})  # Send the prediction result to client

# Start the server
if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000)
