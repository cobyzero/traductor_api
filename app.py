import eel
import numpy as np
from PIL import Image
import base64
import io
from tensorflow.keras.models import load_model
import os
import sys
# Cargar el modelo

if getattr(sys, 'frozen', False):
    try:
        sys.stdout = open('log.txt', 'w', encoding='utf-8')
        sys.stderr = sys.stdout
    except Exception:
        # Si no se puede redirigir, no hacer nada (evita que sys.stdout quede como None)
        pass


def resource_path(relative_path):
    """Soporta ejecución desde PyInstaller"""
    try:
        base_path = sys._MEIPASS  # Folder temporal al ejecutar .exe
    except Exception:
        base_path = os.path.abspath(".")

    return os.path.join(base_path, relative_path)

model = load_model(resource_path('model/sign_language_model.h5'), compile=False)
labels = [chr(i) for i in range(65, 91) if i not in [74, 90]]  # A-Z sin J ni Z

# Iniciar Eel
eel.init(resource_path('templates'))

def preprocess_image(image_data_url):
    header, encoded = image_data_url.split(',', 1)
    image_data = base64.b64decode(encoded)
    image = Image.open(io.BytesIO(image_data)).convert('L').resize((28, 28))
    arr = np.array(image) / 255.0
    return arr.reshape(1, 28, 28, 1)

@eel.expose
def predict_letter(image_data_url):
    try:
        image_array = preprocess_image(image_data_url)
        prediction = model.predict(image_array)[0]
        index = int(np.argmax(prediction))
        confidence = float(np.max(prediction))
        return {
            "letter": labels[index],
            "confidence": round(confidence * 100, 2)
        }
    except Exception as e:
        return {"error": str(e)}

eel.start('index.html', mode='my_portable_chromium', 
                        host='localhost', 
                        port=27000, 
                        block=True)