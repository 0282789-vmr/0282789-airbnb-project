FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY train.py .

# Variables que usará train.py
ENV BUCKET_NAME=0282789_bucket
ENV CONFIG_GCS_PATH=airbnb-project/artifacts/config.json

CMD ["python", "train.py"]

