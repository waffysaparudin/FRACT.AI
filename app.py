from flask import Flask, request, jsonify
from PIL import Image
import numpy as np
import tflite_runtime.interpreter as tflite
import io, base64

app = Flask(__name__)

# ===== Load your quantized TFLite model =====
MODEL_PATH = "model.tflite"
interpreter = tflite.Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# ===== Helper function =====
def preprocess_image(img):
    img = img.convert('RGB')
    img = img.resize((224, 224))
    img = np.expand_dims(np.array(img, dtype=np.uint8), axis=0)
    return img

# ===== Routes =====
@app.route('/')
def home():
    return "✅ HKL Fract.AI Quantized API is running"

@app.route('/predict', methods=['POST'])
def predict():
    print("🚀 Received a request!")
    print("Files received:", request.files)
    print("Form keys:", request.form)

    image = None

    # 1️⃣ Handle file upload (from Thunkable using multipart/form-data)
    if 'file' in request.files:
        file = request.files['file']
        if file.filename == '':
            print("❌ Empty filename")
            return jsonify({'error': 'Empty filename'}), 400
        image = Image.open(file.stream)

    # 2️⃣ Handle base64 uploads (if Thunkable sends data:image/... format)
    elif 'file' in request.form:
        print("⚠️ File not in request.files, trying base64 decode")
        data_url = request.form['file']
        try:
            header, encoded = data_url.split(",", 1)
            image_data = base64.b64decode(encoded)
            image = Image.open(io.BytesIO(image_data))
        except Exception as e:
            print("❌ Base64 decode failed:", str(e))
            return jsonify({'error': 'Invalid base64 data'}), 400
    else:
        print("❌ No file or base64 data found")
        return jsonify({'error': 'No file uploaded'}), 400

    # 3️⃣ Preprocess and predict
    try:
        img = preprocess_image(image)
        interpreter.set_tensor(input_details[0]['index'], img)
        interpreter.invoke()

        output_data = interpreter.get_tensor(output_details[0]['index'])
        prediction_index = np.argmax(output_data)
        confidence = float(np.max(output_data))

        # Update your labels.txt reading here
        labels = ["Fractured", "Normal"]
        label = labels[prediction_index] if prediction_index < len(labels) else "Unknown"

        print(f"✅ Prediction: {label}, Confidence: {confidence}")
        return jsonify({'prediction': label, 'confidence': confidence})
    except Exception as e:
        print("💥 Error:", str(e))
        return jsonify({'error': str(e)}), 500

# ===== Run =====
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=10000)
