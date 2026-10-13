FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY engine ./engine
COPY server ./server
COPY web ./web
COPY voices ./voices
ENV PYTHONUNBUFFERED=1
CMD ["uvicorn", "server.worker_app:build_app", "--factory", "--host", "127.0.0.1", "--port", "8100"]
