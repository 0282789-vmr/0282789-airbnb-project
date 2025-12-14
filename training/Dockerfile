FROM python:3.12-slim

WORKDIR /app

# Dependencias
COPY training/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Código
COPY training/train.py /app/train.py

# (Opcional) si tienes módulos compartidos, copia también utils/
# COPY utils /app/utils

CMD ["python", "train.py"]
