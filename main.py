from flask import Flask, request, jsonify
import numpy as np
import tensorflow as tf
from PIL import Image
from io import BytesIO

app = Flask(__name__)
model = tf.keras.models.load_model('sign_model.h5')
class_names = [l for l in "ABCDEFGHIKLMNOPQRSTUVWX" ]  # sin J ni Z

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'}), 400

    file = request.files['image']
    img = Image.open(BytesIO(file.read())).convert('L')  # Escala de grises
    img = img.resize((28, 28))
    img_array = np.array(img).reshape(1, 28, 28, 1) / 255.0

    pred = model.predict(img_array)
    letter = class_names[np.argmax(pred)]
    return jsonify({'letter': letter})

if __name__ == '__main__':
    app.run()