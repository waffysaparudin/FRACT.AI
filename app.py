import os
import tensorflow as tf
from flask import Flask, request, jsonify
from PIL import Image
import numpy as np

app = Flask(__name__)

# --- Safe path setup ---
MODEL_PATH = os.path.join(os.path.dirname(__file__), "model.tflite")
LABELS_PATH = os.path.join(os.path.dirname(__file__), "labels.txt")

# --- Load model & labels ---
interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()

with open(LABELS_PATH, "r") as f:
    labels = [line.strip() for line in f.readlines()]






from flask import Flask, request, jsonify
from PIL import Image
import numpy as np
import tensorflow as tf
import io

app = Flask(__name__)

# Load TFLite model and allocate tensors
interpreter = tf.lite.Interpreter(model_path="model.tflite")
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Load class labels
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
    input_data = preprocess(image)

    # Set input tensor
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()

    # Get prediction result
    output_data = interpreter.get_tensor(output_details[0]['index'])
    idx = int(np.argmax(output_data))
    confidence = float(np.max(output_data))
    result = labels[idx]

    return jsonify({'prediction': result, 'confidence': confidence})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
