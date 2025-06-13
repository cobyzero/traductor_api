from flask import Flask, render_template, Response, jsonify, request
import cv2
import numpy as np
import tensorflow as tf
from collections import deque
import time
import threading
import logging
import os
import sys
import base64
import json

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('app.log')
    ]
)
logger = logging.getLogger(__name__)

# Intentar importar MediaPipe para detección de manos
try:
    import mediapipe as mp
    from mediapipe.tasks import python
    from mediapipe.tasks.python import vision
    from mediapipe import solutions
    from mediapipe.framework.formats import landmark_pb2
    MEDIAPIPE_AVAILABLE = True
except ImportError:
    logger.warning("MediaPipe no está instalado. La detección de manos no estará disponible.")
    MEDIAPIPE_AVAILABLE = False

class HandDetector:
    def __init__(self):
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5
        )
        self.mp_drawing = mp.solutions.drawing_utils
        
    def process(self, image):
        if not MEDIAPIPE_AVAILABLE:
            return image
            
        # Convertir la imagen de BGR a RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Procesar la imagen y detectar manos
        results = self.hands.process(image_rgb)
        
        # Dibujar los landmarks de las manos
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                self.mp_drawing.draw_landmarks(
                    image,
                    hand_landmarks,
                    self.mp_hands.HAND_CONNECTIONS,
                    self.mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                    self.mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2)
                )
        
        return image

class Camera:
    def __init__(self, src=0, width=640, height=480, is_virtual=False):
        self.src = src
        self.width = width
        self.height = height
        self.frame = None
        self.stopped = False
        self.lock = threading.Lock()
        self.last_frame_time = time.time()
        self.frame_timeout = 2.0  # segundos
        self.is_virtual = is_virtual
        self.hand_detector = HandDetector() if MEDIAPIPE_AVAILABLE else None
        self.test_frame = self._create_test_frame() if is_virtual else None
        
    def start(self):
        self.stopped = False
        self.thread = threading.Thread(target=self.update, daemon=True)
        self.thread.start()
        return self
    
    def _create_test_frame(self):
        """Crea un frame de prueba con un círculo que se mueve"""
        frame = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        t = time.time()
        center_x = int(self.width // 2 + self.width // 3 * np.sin(t * 2))
        center_y = int(self.height // 2 + self.height // 3 * np.cos(t * 3))
        cv2.circle(frame, (center_x, center_y), 30, (0, 255, 0), -1)
        cv2.putText(frame, "MODO DE PRUEBA", (50, 50), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        return frame
    
    def update(self):
        cap = None
        
        # Si es una cámara virtual, solo usamos el frame de prueba
        if self.is_virtual:
            while not self.stopped:
                with self.lock:
                    self.test_frame = self._create_test_frame()
                    self.frame = self.test_frame
                    self.last_frame_time = time.time()
                time.sleep(0.03)  # ~30 FPS
            return
            
        # Para cámaras reales
        while not self.stopped:
            try:
                if cap is None or not cap.isOpened():
                    if cap is not None:
                        cap.release()
                    
                    # Intentar diferentes fuentes de video
                    if isinstance(self.src, str) and self.src.startswith('http'):
                        # Para streams RTSP o HTTP
                        cap = cv2.VideoCapture(self.src, cv2.CAP_FFMPEG)
                    else:
                        # Para cámaras USB
                        cap = cv2.VideoCapture(self.src, cv2.CAP_ANY)
                    
                    if not cap.isOpened():
                        logger.error(f"No se pudo abrir la fuente de video: {self.src}")
                        time.sleep(2)
                        continue
                        
                    cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
                    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
                    cap.set(cv2.CAP_PROP_FPS, 30)
                    time.sleep(1)  # Dar tiempo a la cámara para inicializarse
                    logger.info(f"Cámara {self.src} inicializada correctamente")
                    continue
                
                ret, frame = cap.read()
                if not ret:
                    logger.warning("No se pudo leer el frame de la cámara")
                    cap.release()
                    cap = None
                    time.sleep(1)
                    continue
                
                # Procesar detección de manos si está disponible
                if self.hand_detector:
                    frame = self.hand_detector.process(frame)
                
                with self.lock:
                    self.frame = frame
                    self.last_frame_time = time.time()
                
                time.sleep(0.03)  # ~30 FPS
                
            except Exception as e:
                logger.error(f"Error en el hilo de la cámara: {str(e)}", exc_info=True)
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
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max

# Crear carpeta de subidas si no existe
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Cargar el modelo de reconocimiento de señas
try:
    model = tf.keras.models.load_model('model_mnist_asl.h5')
    labels = [chr(i) for i in range(65, 91) if chr(i) not in ['J', 'Z']]
except Exception as e:
    logger.error(f"Error al cargar el modelo: {str(e)}")
    logger.warning("El sistema funcionará en modo solo detección de manos")
    model = None
    labels = []

# Configuración de la cámara
CAMERA_SOURCE = os.environ.get('CAMERA_SOURCE', '0')  # Por defecto cámara 0, puede ser una URL RTSP
IS_VIRTUAL = os.environ.get('VIRTUAL_CAMERA', 'true').lower() == 'true'  # Por defecto modo virtual para evitar errores

# Inicializar la cámara
camera = Camera(
    src=CAMERA_SOURCE if not IS_VIRTUAL else 0,
    width=800,
    height=600,
    is_virtual=IS_VIRTUAL
).start()

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

# Ruta para la página de inicio
@app.route('/')
def home():
    return render_template('home.html')

# Ruta para la cámara en tiempo real
@app.route('/camera')
def camera():
    return render_template('camera.html', 
                         hand_detection=MEDIAPIPE_AVAILABLE,
                         model_loaded=model is not None,
                         camera_source=CAMERA_SOURCE if not IS_VIRTUAL else 'Virtual',
                         is_virtual=IS_VIRTUAL)

# Ruta para subir imagen
@app.route('/upload', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        if 'file' not in request.files:
            return jsonify({'error': 'No se ha seleccionado ningún archivo'}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No se ha seleccionado ningún archivo'}), 400
        
        if file:
            # Guardar el archivo
            filename = os.path.join(app.config['UPLOAD_FOLDER'], 'temp.jpg')
            file.save(filename)
            
            # Procesar la imagen
            result = process_image(filename)
            return jsonify(result)
    
    return render_template('upload.html')

@app.route('/video')
def video():
    return Response(generate_frames(), 
                   mimetype='multipart/x-mixed-replace; boundary=frame')

# Ruta para el feed de video
@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

# Función para procesar una imagen estática
def process_image(image_path):
    try:
        # Leer la imagen
        frame = cv2.imread(image_path)
        if frame is None:
            return {'error': 'No se pudo leer la imagen'}
        
        # Hacer una copia para el procesamiento del modelo
        processed_frame = frame.copy()
        
        # Redimensionar manteniendo la relación de aspecto para visualización
        height, width = frame.shape[:2]
        max_size = 800
        if height > width:
            new_height = max_size
            new_width = int(width * (max_size / height))
        else:
            new_width = max_size
            new_height = int(height * (max_size / width))
            
        display_frame = cv2.resize(frame.copy(), (new_width, new_height))
        
        # Procesar con MediaPipe si está disponible
        hand_landmarks = None
        if MEDIAPIPE_AVAILABLE and hand_detector:
            # Procesar el frame original para la detección de manos
            display_frame, hand_landmarks = hand_detector.process(display_frame)
            
            # Si se detectaron manos, procesar para el modelo
            if hand_landmarks:
                # Crear una máscara para la mano detectada
                h, w = processed_frame.shape[:2]
                mask = np.zeros((h, w), dtype=np.uint8)
                
                # Dibujar los puntos de referencia de la mano en la máscara
                for hand_landmark in hand_landmarks:
                    points = []
                    for landmark in hand_landmark.landmark:
                        x = min(int(landmark.x * w), w-1)
                        y = min(int(landmark.y * h), h-1)
                        points.append([x, y])
                    
                    if points:
                        # Crear un polígono convexo alrededor de los puntos de la mano
                        hull = cv2.convexHull(np.array(points, dtype=np.int32))
                        cv2.fillConvexPoly(mask, hull, 255)
                
                # Aplicar la máscara a la imagen
                processed_frame = cv2.bitwise_and(processed_frame, processed_frame, mask=mask)
        
        # Preprocesar la imagen para el modelo
        model_input = preprocess(processed_frame)
        
        # Hacer la predicción si el modelo está cargado
        detected_letter = '?'
        confidence = 0.0
        
        if model is not None:
            # Realizar la predicción
            prediction = model.predict(np.expand_dims(model_input, axis=0))[0]
            predicted_class = np.argmax(prediction)
            confidence = float(prediction[predicted_class])
            
            # Mapear la clase predicha a la letra correspondiente
            # Asumiendo que las letras están en orden alfabético (A=0, B=1, ..., Z=25)
            # Excluyendo J y Z según el alfabeto de lenguaje de señas
            asl_letters = [chr(i) for i in range(65, 91) if chr(i) not in ['J', 'Z']]
            if 0 <= predicted_class < len(asl_letters):
                detected_letter = asl_letters[predicted_class]
            
            # Guardar en el historial si la confianza es mayor al 50%
            if confidence > 0.5:
                history.append({
                    'letter': detected_letter,
                    'confidence': float(confidence),
                    'timestamp': time.time()
                })
        
        # Convertir a formato para mostrar en el navegador
        _, buffer = cv2.imencode('.jpg', display_frame)
        frame_bytes = buffer.tobytes()
        
        return {
            'image': base64.b64encode(frame_bytes).decode('utf-8'),
            'detected_letter': detected_letter,
            'confidence': confidence,
            'hand_detected': hand_landmarks is not None
        }
    except Exception as e:
        logger.error(f"Error al procesar la imagen: {str(e)}")
        return {'error': str(e)}

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