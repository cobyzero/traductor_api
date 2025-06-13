from app import app, camera

if __name__ == "__main__":
    # Iniciar la cámara al arrancar el servidor
    camera.start()
    app.run(host='0.0.0.0', port=5000)
