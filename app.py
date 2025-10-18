from flask import Flask, request, jsonify
import tensorflow as tf
from PIL import Image
import numpy as np
import io

app = Flask(__name__)

# Load model and labels
model = tf.keras.models.load_model("keras_model.h5")
with open("labels.txt", "r") as f:
    labels = [line.strip() for line in f.readlines()]

def preprocess(image):
    image = image.resize((224, 224))
    arr = np.array(image).astype(np.float32) / 255.0
    arr = np.expand_dims(arr, axis=0)
    return arr

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'no file uploaded'}), 400

    file = request.files['file']
    image = Image.open(io.BytesIO(file.read())).convert("RGB")
    arr = preprocess(image)
    preds = model.predict(arr)
    idx = int(np.argmax(preds))
    confidence = float(np.max(preds))
    result = labels[idx]

    return jsonify({'prediction': result, 'confidence': confidence})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
