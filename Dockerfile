FROM python:3.11-slim
WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app.py .
COPY data ./data

ENV EMBED_DIM=256
ENV WEB_CONCURRENCY=4
EXPOSE 8080

# 단일 머신에 리소스 집중 + 워커 다중화(요청 병렬). 워커 수 = WEB_CONCURRENCY(fly.toml [env]).
# 각 워커가 벡터행렬을 메모리에 적재(8MB f16→16MB f32).
CMD ["sh", "-c", "uvicorn app:app --host 0.0.0.0 --port 8080 --workers ${WEB_CONCURRENCY:-4}"]
