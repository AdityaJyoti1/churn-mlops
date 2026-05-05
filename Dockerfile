FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PORT=8080

WORKDIR /app

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential && rm -rf /var/lib/apt/lists/*

# Python deps
COPY requirements-serve.txt .
RUN pip install --upgrade pip && pip install -r requirements-serve.txt

# App code
COPY src/app.py ./app.py

EXPOSE 8080
CMD exec uvicorn app:app --host 0.0.0.0 --port ${PORT}
