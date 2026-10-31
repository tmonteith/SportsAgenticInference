# Stage 1: Official llama-server binary & shared libraries from ggml-org
FROM ghcr.io/ggml-org/llama.cpp:server AS llama-bin

# Stage 2: Build Python dependency wheels
FROM python:3.12-slim AS python-builder

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /build

COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip wheel --no-cache-dir --wheel-dir /build/wheels -r requirements.txt

# Stage 3: Runtime Image
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=8081 \
    MODEL_ID=microsoft/Phi-4-mini-instruct \
    MODEL_ALIAS=local-model \
    MODEL_CACHE_DIR=/models \
    LLAMA_SERVER_BIN=/opt/llama/llama-server \
    LLAMA_INTERNAL_HOST=127.0.0.1 \
    LLAMA_INTERNAL_PORT=8082 \
    LD_LIBRARY_PATH=/opt/llama

WORKDIR /app

# Install OpenMP runtime required by llama-server and create non-root user + model cache dir
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    ca-certificates \
    curl \
    && rm -rf /var/lib/apt/lists/* && \
    groupadd --system appgroup && \
    useradd --system --gid appgroup --home-dir /app --no-create-home appuser && \
    mkdir -p /models /opt/llama

# Copy prebuilt llama-server binary and shared libraries from official llama.cpp image
COPY --from=llama-bin /app /opt/llama
RUN ln -s /opt/llama/llama-server /usr/local/bin/llama-server

# Install pre-built Python wheels
COPY --from=python-builder /build/wheels /wheels
RUN pip install --no-cache-dir --no-index --find-links=/wheels /wheels/* && \
    rm -rf /wheels

# Copy application source and agent skills
COPY app ./app
COPY .agents/skills ./.agents/skills

RUN chown -R appuser:appgroup /app /models /opt/llama

USER appuser

VOLUME ["/models"]

EXPOSE 8081

CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8081}"]

