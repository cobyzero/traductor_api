from flask import Flask, render_template, Response, jsonify
import cv2
import numpy as np
import tensorflow as tf
from collections import deque
import time
import threading
import logging

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class Camera:
    def __init__(self, src=0, width=640, height=480):
        self.src = src
        self.width = width
        self.height = height
        self.frame = None
        self.stopped = False
        self.lock = threading.Lock()
        self.last_frame_time = time.time()
        self.frame_timeout = 1.0  # segundos
        
    def start(self):
        self.stopped = False
        self.thread = threading.Thread(target=self.update, daemon=True)
        self.thread.start()
        return self
    
    def update(self):
        cap = None
        while not self.stopped:
            try:
                if cap is None or not cap.isOpened():
                    if cap is not None:
                        cap.release()
                    cap = cv2.VideoCapture(self.src)
                    cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
                    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
                    time.sleep(0.5)  # Dar tiempo a la cámara para inicializarse
                    continue
                
                ret, frame = cap.read()
                if not ret:
                    logger.warning("No se pudo leer el frame de la cámara")
                    cap.release()
                    cap = None
                    time.sleep(0.5)
                    continue
                
                with self.lock:
                    self.frame = frame
                    self.last_frame_time = time.time()
                
                # Pequeña pausa para no saturar la CPU
                time.sleep(0.03)  # ~30 FPS
                
            except Exception as e:
                logger.error(f"Error en el hilo de la cámara: {str(e)}")
                if cap is not None:
                    cap.release()
                    cap = None
                time.sleep(1)
        
        if cap is not None:
            cap.release()
    
    def read(self):
        with self.lock:
            if self.frame is None or (time.time() - self.last_frame_time) > self.frame_timeout:
                return None, None
            return True, self.frame.copy()
    
    def stop(self):
        self.stopped = True
        if hasattr(self, 'thread'):
            self.thread.join()

# Inicializar la cámara
camera = Camera().start()

def get_camera():
    global camera
    if camera is None or camera.stopped:
        camera = Camera().start()
    return camera

app = Flask(__name__)
model = tf.keras.models.load_model('model_mnist_asl.h5')
labels = [chr(i) for i in range(65, 91) if chr(i) not in ['J', 'Z']]

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
        try:
            # Obtener el frame de la cámara
            cam = get_camera()
            success, frame = cam.read()
            
            if not success or frame is None:
                # Generar un frame negro con mensaje de error
                frame = np.zeros((480, 640, 3), dtype=np.uint8)
                cv2.putText(frame, "Esperando cámara...", (50, 240), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                ret, buffer = cv2.imencode('.jpg', frame)
                yield (b'--frame\r\n'
                      b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
                time.sleep(1)
                continue
            
            # Voltear la imagen horizontalmente para una experiencia tipo espejo
            frame = cv2.flip(frame, 1)
            
            # Hacer una copia del frame para mostrar
            output_frame = frame.copy()
            
            # Dibujar el área de detección
            height, width = frame.shape[:2]
            size = min(height, width) // 2
            x = (width - size) // 2
            y = (height - size) // 2
            
            # Dibujar el área de detección con un borde más visible
            cv2.rectangle(output_frame, (x, y), (x + size, y + size), (0, 255, 0), 2)
            
            try:
                # Preprocesar solo el área de interés
                roi = frame[y:y+size, x:x+size]
                if roi.size > 0:
                    input_img = preprocess(roi)
                    
                    # Realizar la predicción (con un timeout para evitar bloqueos)
                    pred = model.predict(input_img, verbose=0)[0]
                    predicted_idx = np.argmax(pred)
                    confidence = pred[predicted_idx]
                    letter = labels[predicted_idx]
                    
                    # Mostrar la predicción
                    text = f'Letra: {letter} ({confidence*100:.1f}%)'
                    cv2.putText(output_frame, text, (10, 40),
                              cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
                    
                    # Lógica para confirmar la letra
                    if confidence > CONFIDENCE_THRESHOLD:
                        if letter == last_letter:
                            letter_count += 1
                            if letter_count == MIN_FRAMES and (not history or history[-1] != letter):
                                history.append(letter)
                                if len(history) > 50:
                                    history.pop(0)
                        else:
                            last_letter = letter
                            letter_count = 1
            except Exception as e:
                logger.error(f"Error en el procesamiento: {str(e)}")
            
            # Codificar el frame para la transmisión
            ret, buffer = cv2.imencode('.jpg', output_frame, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
            if not ret:
                logger.warning("Error al codificar el frame")
                continue
                
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                  b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
            
            # Pequeña pausa para no saturar el cliente
            time.sleep(0.03)
            
        except Exception as e:
            logger.error(f"Error en generate_frames: {str(e)}")
            time.sleep(1)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video')
def video():
    return Response(generate_frames(), 
                   mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(),
                   mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/history')
def get_history():
    # Devolver las últimas 30 letras como una cadena
    return jsonify({
        'history': ' '.join(history[-30:]),
        'last_letter': history[-1] if history else ''
    })

def run_app():
    try:
        app.run(host='0.0.0.0', port=5000, debug=True, use_reloader=False, threaded=True)
    except Exception as e:
        logger.error(f"Error en la aplicación: {str(e)}")
    finally:
        if 'camera' in globals() and camera is not None:
            camera.stop()

if __name__ == '__main__':
    try:
        run_app()
    except KeyboardInterrupt:
        logger.info("Deteniendo la aplicación...")
    finally:
        if 'camera' in globals() and camera is not None:
            camera.stop()