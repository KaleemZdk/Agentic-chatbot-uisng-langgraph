FROM python:3.12-slim

# libgomp1: faiss-cpu's wheels are built with OpenMP and fail at import
# time without it on slim/Debian bases. curl: used by the healthcheck.
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgomp1 \
        curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Sqlite checkpoint db + FAISS index are written here (see DATA_DIR in
# chatbot.py) — mount a volume on this path to persist them across
# container restarts/rebuilds.
RUN mkdir -p /app/data
VOLUME ["/app/data"]

EXPOSE 8501

HEALTHCHECK --interval=30s --timeout=5s --start-period=15s \
    CMD curl --fail http://localhost:8501/_stcore/health || exit 1

ENTRYPOINT ["streamlit", "run", "app.py", \
            "--server.port=8501", \
            "--server.address=0.0.0.0"]