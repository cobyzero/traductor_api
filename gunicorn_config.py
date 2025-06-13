# Número de workers
workers = 2

# Dirección y puerto
bind = '0.0.0.0:5000'

# Tiempo máximo de espera
worker_class = 'gevent'
worker_connections = 1000
timeout = 30
keepalive = 2

# Logging
accesslog = 'gunicorn_access.log'
errorlog = 'gunicorn_error.log'
loglevel = 'info'
