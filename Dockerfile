FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

RUN groupadd --gid 10001 medhub && \
    useradd --uid 10001 --gid medhub --no-create-home medhub && \
    mkdir -p /data/chroma_db && chown -R medhub:medhub /app /data

USER 10001:10001
ENV CHROMA_PERSIST_DIR=/data/chroma_db PYTHONUNBUFFERED=1

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
  CMD ["python", "-c", "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health/live', timeout=3)"]
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]
