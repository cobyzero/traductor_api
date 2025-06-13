#!/bin/bash

# Activar el entorno virtual (si lo estás usando)
# source venv/bin/activate

# Variables de entorno
export PRODUCTION=true
export VIRTUAL_CAMERA=true  # Cambiar a false si tienes cámara en el servidor

# Instalar dependencias
pip install -r requirements.txt

# Iniciar Gunicorn
echo "Iniciando el servidor Gunicorn..."
gunicorn --config gunicorn_config.py wsgi:app
