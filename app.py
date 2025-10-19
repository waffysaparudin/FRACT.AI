import os
from flask import Flask, request, jsonify
from PIL import Image
import numpy as np

# Lightweight TFLite runtime
try:
    import tflite_runtime.interpreter as tflite
except ImportError:
    import tensorflow.lite as tflite

app = Flask(__name__)

# Safe paths
MODEL_PATH = os.path.join(os.path.dirname(__file__), "model.tflite")
LABELS_PATH = os.path.join(os.path.dirname(__file__), "labels.txt")

# Load model
interpreter = tflite.Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Load labels
with open(LABELS_PATH, "r") as f:
    labels = [line.strip().split(' ', 1)[-1] for line in f.readlines()]

@app.route('/')
def home():
    return "✅ HKL Fract.AI Quantized API is running"

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'Empty filename'}), 400

    try:
        img = Image.open(file.stream).convert('RGB')
        img = img.resize((224, 224))
        img = np.expand_dims(np.array(img, dtype=np.uint8), axis=0)

        interpreter.set_tensor(input_details[0]['index'], img)
        interpreter.invoke()
        output_data = interpreter.get_tensor(output_details[0]['index'])

        predicted_index = int(np.argmax(output_data))
        confidence = float(np.max(output_data))
        label = labels[predicted_index]

        return jsonify({'prediction': label, 'confidence': confidence})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=10000)
