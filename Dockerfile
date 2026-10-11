FROM python:3.11-slim
WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app.py .
COPY data ./data

ENV EMBED_DIM=256
ENV WEB_CONCURRENCY=1
EXPOSE 8080

# 워커 수 = WEB_CONCURRENCY(운영 값은 fly.toml [env] 가 정한다 — 1).
# 워커마다 벡터 두 벌(상세·짧은 캡션)과 메타를 메모리에 올린다.
CMD ["sh", "-c", "uvicorn app:app --host 0.0.0.0 --port 8080 --workers ${WEB_CONCURRENCY:-1}"]
