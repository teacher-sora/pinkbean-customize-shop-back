FROM python:3.11-slim
WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app.py .
COPY data ./data

ENV EMBED_DIM=256
EXPOSE 8080

# cpu=1 + 워커 2개(병렬). 각 워커가 벡터행렬을 메모리에 적재(8MB f16→16MB f32, ×2).
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8080", "--workers", "2"]
