from flask import Flask, render_template, Response, jsonify
import cv2
import numpy as np
import tensorflow as tf
from collections import deque

app = Flask(__name__)
model = tf.keras.models.load_model('model_mnist_asl.h5')
labels = [chr(i) for i in range(65, 91) if chr(i) not in ['J', 'Z']]

# Configuración de la cámara
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

# Historial de letras detectadas
history = []
last_letter = None
letter_count = 0
CONFIDENCE_THRESHOLD = 0.8  # Umbral de confianza mínimo
MIN_FRAMES = 5  # Mínimo de frames para confirmar una letra

def preprocess(frame):
    # Convertir a escala de grises
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Aplicar desenfoque gaussiano para reducir ruido
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Aplicar umbral adaptativo
    thresh = cv2.adaptiveThreshold(blurred, 255, 
                                 cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                 cv2.THRESH_BINARY_INV, 11, 2)
    
    # Encontrar contornos
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, 
                                 cv2.CHAIN_APPROX_SIMPLE)
    
    # Si no hay contornos, devolver una imagen en blanco
    if not contours:
        return np.zeros((1, 28, 28, 1))
    
    # Encontrar el contorno más grande
    largest_contour = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(largest_contour)
    
    # Asegurarse de que el ROI sea cuadrado
    size = max(w, h)
    x = max(0, x + (w - size) // 2)
    y = max(0, y + (h - size) // 2)
    
    # Obtener ROI y redimensionar a 28x28
    roi = gray[y:y+size, x:x+size]
    if roi.size == 0:
        return np.zeros((1, 28, 28, 1))
    
    resized = cv2.resize(roi, (28, 28), interpolation=cv2.INTER_AREA)
    
    # Normalizar
    normalized = resized / 255.0
    return normalized.reshape(1, 28, 28, 1)

def generate_frames():
    global history, last_letter, letter_count
    
    while True:
        success, frame = cap.read()
        if not success:
            break
            
        # Voltear la imagen horizontalmente para una experiencia tipo espejo
        frame = cv2.flip(frame, 1)
        
        # Preprocesar la imagen
        input_img = preprocess(frame)
        
        # Realizar la predicción
        pred = model.predict(input_img, verbose=0)[0]
        predicted_idx = np.argmax(pred)
        confidence = pred[predicted_idx]
        letter = labels[predicted_idx]
        
        # Dibujar el ROI en el frame
        height, width = frame.shape[:2]
        size = min(height, width) // 2
        x = (width - size) // 2
        y = (height - size) // 2
        cv2.rectangle(frame, (x, y), (x + size, y + size), (0, 255, 0), 2)
        
        # Mostrar la letra predicha y la confianza
        text = f'Letra: {letter} ({confidence*100:.1f}%)'
        cv2.putText(frame, text, (10, 40),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
        
        # Lógica para confirmar la letra
        if confidence > CONFIDENCE_THRESHOLD:
            if letter == last_letter:
                letter_count += 1
                if letter_count == MIN_FRAMES and (not history or history[-1] != letter):
                    history.append(letter)
                    if len(history) > 50:  # Limitar el tamaño del historial
                        history.pop(0)
            else:
                last_letter = letter
                letter_count = 1
        
        # Codificar el frame para la transmisión
        _, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video')
def video():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/history')
def get_history():
    # Devolver las últimas 30 letras como una cadena
    return jsonify({
        'history': ' '.join(history[-30:]),
        'last_letter': history[-1] if history else ''
    })

if __name__ == '__main__':
    app.run(debug=True)