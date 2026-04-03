FROM python:3.11-slim
WORKDIR /app
COPY pipeline2_retrain/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY utils/s3.py ./utils/s3.py
COPY pipeline2_retrain/retrain.py .
CMD ["python", "retrain.py"]
