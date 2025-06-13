from flask import Flask, render_template, jsonify, request, send_from_directory
import cv2
import numpy as np
import tensorflow as tf
import os
import logging
import sys
from datetime import datetime

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
        self.stopped = True  # Inicialmente detenido
        self.lock = threading.Lock()
        self.last_frame_time = time.time()
        self.frame_timeout = 2.0  # segundos
        self.is_virtual = is_virtual
        self.hand_detector = HandDetector() if MEDIAPIPE_AVAILABLE else None
        self.test_frame = self._create_test_frame() if is_virtual else None
        self.thread = None
        self.cap = None
    
    def _create_test_frame(self):
        """Crea un frame de prueba con un círculo que se mueve"""
        frame = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        t = time.time()
        x = int((np.sin(t) * 0.4 + 0.5) * self.width)
        y = int((np.cos(t * 0.7) * 0.4 + 0.5) * self.height)
        cv2.circle(frame, (x, y), 30, (0, 255, 0), -1)
        return frame
    
    def start(self):
        """Inicia la cámara y el hilo de captura."""
        if self.thread is not None and self.thread.is_alive():
            return True  # Ya está corriendo
            
        self.stopped = False
        
        try:
            if self.is_virtual:
                self.thread = threading.Thread(target=self._update_virtual, daemon=True)
                self.thread.start()
                logger.info("Cámara virtual iniciada correctamente")
                return True
                
            # Para cámara real
            logger.info(f"Intentando abrir la cámara: {self.src}")
            self.cap = cv2.VideoCapture(self.src)
            
            if not self.cap.isOpened():
                logger.error(f"No se pudo abrir la fuente de video: {self.src}")
                # Intentar forzar la apertura con backend específico
                self.cap = cv2.VideoCapture(self.src, cv2.CAP_ANY)
                if not self.cap.isOpened():
                    logger.error("Segundo intento fallido. Usando modo virtual.")
                    self.is_virtual = True
                    self.thread = threading.Thread(target=self._update_virtual, daemon=True)
                    self.thread.start()
                    return True
        except Exception as e:
            logger.error(f"Error al iniciar la cámara: {str(e)}")
            self.is_virtual = True
            self.thread = threading.Thread(target=self._update_virtual, daemon=True)
            self.thread.start()
            return True
            
        # Configurar propiedades de la cámara
        try:
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
            
            # Iniciar el hilo de actualización
            self.thread = threading.Thread(target=self._update, daemon=True)
            self.thread.start()
            
            logger.info(f"Cámara {self.src} iniciada correctamente")
            return True
            
        except Exception as e:
            logger.error(f"Error al configurar la cámara: {str(e)}")
            self.is_virtual = True
            self.thread = threading.Thread(target=self._update_virtual, daemon=True)
            self.thread.start()
            return True

# Configuración de la aplicación
UPLOAD_FOLDER = 'static/uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}

# Asegurar que el directorio de subidas exista
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Inicializar la aplicación Flask
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max
app.config['UPLOAD_EXTENSIONS'] = ALLOWED_EXTENSIONS

# Cargar el modelo de reconocimiento de señas
try:
    model = tf.keras.models.load_model('model_mnist_asl.h5')
    labels = [chr(i) for i in range(65, 91) if chr(i) not in ['J', 'Z']]
    logger.info("Modelo cargado exitosamente")
except Exception as e:
    logger.error(f"Error al cargar el modelo: {str(e)}")
    logger.warning("El sistema funcionará en modo solo detección de manos")
    model = None
    labels = []

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/api/process-image', methods=['POST'])
def process_upload():
    if 'image' not in request.files:
        return jsonify({'error': 'No se proporcionó ninguna imagen'}), 400
    
    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'Nombre de archivo vacío'}), 400
    
    if file and allowed_file(file.filename):
        # Generar un nombre de archivo único
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"{timestamp}_{file.filename}"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        
        # Guardar la imagen
        file.save(filepath)
        
        try:
            # Procesar la imagen
            result = process_image(filepath)
            
            # Devolver el resultado
            return jsonify({
                'status': 'success',
                'filename': filename,
                'result': result
            })
            
        except Exception as e:
            logger.error(f"Error al procesar la imagen: {str(e)}")
            return jsonify({'error': str(e)}), 500
    
    return jsonify({'error': 'Tipo de archivo no permitido'}), 400

def preprocess(frame):
    # Asegurarse de que la imagen tenga 3 canales (por si es en escala de grises)
    if len(frame.shape) == 2:
        frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
    
    # Convertir a escala de grises
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Aplicar desenfoque gaussiano para reducir ruido
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Aplicar umbral adaptativo
    thresh = cv2.adaptiveThreshold(
        blurred, 255, 
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
        cv2.THRESH_BINARY_INV, 
        11, 2
    )
    
    # Encontrar contornos
    contours, _ = cv2.findContours(
        thresh, 
        cv2.RETR_EXTERNAL, 
        cv2.CHAIN_APPROX_SIMPLE
    )
    
    # Si no hay contornos, devolver una imagen en blanco
    if not contours:
        return np.zeros((28, 28, 1), dtype=np.float32)
    
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
        return np.zeros((28, 28, 1), dtype=np.float32)
    
    # Redimensionar a 28x28 y normalizar a [0, 1]
    resized = cv2.resize(roi, (28, 28), interpolation=cv2.INTER_AREA)
    
    # Asegurar que los valores estén en el rango [0, 1]
    normalized = resized.astype(np.float32) / 255.0
    
    # Añadir dimensión del canal si es necesario
    if len(normalized.shape) == 2:
        normalized = np.expand_dims(normalized, axis=-1)
    
    return normalized

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
            try:
                # Asegurarse de que la entrada tenga la forma correcta (batch_size, 28, 28, 1)
                if len(model_input.shape) == 3:
                    model_input = np.expand_dims(model_input, axis=0)
                
                # Realizar la predicción
                prediction = model.predict(model_input)[0]
                predicted_class = np.argmax(prediction)
                confidence = float(prediction[predicted_class])
            except Exception as e:
                logger.error(f"Error en la predicción del modelo: {str(e)}")
                confidence = 0.0
                predicted_class = -1
            
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

@app.route('/start_processing', methods=['POST'])
def start_processing():
    """Inicia el procesamiento de la cámara."""
    global processing_active
    processing_active = True
    camera.start()
    return jsonify({'status': 'processing_started'})

@app.route('/stop_processing', methods=['POST'])
def stop_processing():
    """Detiene el procesamiento de la cámara."""
    global processing_active
    processing_active = False
    return jsonify({'status': 'processing_stopped'})

@app.route('/history')
def get_history():
    # Devolver las últimas 30 letras como una cadena
    return jsonify({
        'history': ' '.join(history[-30:]),
        'last_letter': history[-1] if history else ''
    })

def run_app():
    try:
        app.run(debug=True)
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