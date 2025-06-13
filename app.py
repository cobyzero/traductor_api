from flask import Flask, render_template, Response
import cv2
import numpy as np
import tensorflow as tf
from flask import jsonify

app = Flask(__name__)
model = tf.keras.models.load_model('model_mnist_asl.h5')
labels = [chr(i) for i in range(65, 91) if chr(i) not in ['J', 'Z']]

cap = cv2.VideoCapture(0)
history = []

def preprocess(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    height, width = gray.shape
    min_dim = min(height, width)
    x = (width - min_dim) // 2
    y = (height - min_dim) // 2
    cropped = gray[y:y+min_dim, x:x+min_dim]
    resized = cv2.resize(cropped, (28, 28))
    normalized = resized / 255.0
    return normalized.reshape(1, 28, 28, 1)

def generate_frames():
    global history
    while True:
        success, frame = cap.read()
        if not success:
            break

        input_img = preprocess(frame)
        pred = model.predict(input_img, verbose=0)
        letter = labels[np.argmax(pred)]
        history.append(letter)

        # Mostrar predicción
        cv2.putText(frame, f'Letra: {letter}', (10, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)

        _, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html', history=' '.join(history[-30:]))

@app.route('/video')
def video():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/history')
def get_history():
    return jsonify({'history': ''.join(history[-30:])})

if __name__ == '__main__':
    app.run(debug=True)