FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY main.py .

ENV PORT=8080
ENV BUCKET_NAME=0282789_bucket
ENV CONFIG_GCS_PATH=airbnb-project/artifacts/config.json

EXPOSE 8080

CMD ["uvicorn", "main:app", "--host=0.0.0.0", "--port=8080"]

