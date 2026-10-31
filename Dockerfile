# Stage 1: Build llama-server from llama.cpp (multi-arch: linux/amd64 & linux/arm64)
FROM debian:bookworm-slim AS llama-builder

ARG LLAMA_CPP_REF=master

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    git \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /src
RUN git clone --depth 1 --branch ${LLAMA_CPP_REF} https://github.com/ggml-org/llama.cpp.git && \
    cmake -S llama.cpp -B build \
      -DCMAKE_BUILD_TYPE=Release \
      -DBUILD_SHARED_LIBS=OFF \
      -DLLAMA_BUILD_SERVER=ON \
      -DLLAMA_BUILD_TESTS=OFF \
      -DLLAMA_BUILD_EXAMPLES=OFF \
      -DLLAMA_CURL=OFF && \
    cmake --build build --target llama-server -j "$(nproc)" && \
    cp build/bin/llama-server /usr/local/bin/llama-server && \
    strip /usr/local/bin/llama-server

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
    LLAMA_SERVER_BIN=/usr/local/bin/llama-server \
    LLAMA_INTERNAL_HOST=127.0.0.1 \
    LLAMA_INTERNAL_PORT=8082

WORKDIR /app

# Install OpenMP runtime required by llama-server and create non-root user + model cache dir
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/* && \
    groupadd --system appgroup && \
    useradd --system --gid appgroup --home-dir /app --no-create-home appuser && \
    mkdir -p /models

# Copy compiled llama-server binary
COPY --from=llama-builder /usr/local/bin/llama-server /usr/local/bin/llama-server

# Install pre-built Python wheels
COPY --from=python-builder /build/wheels /wheels
RUN pip install --no-cache-dir --no-index --find-links=/wheels /wheels/* && \
    rm -rf /wheels

# Copy application source and agent skills
COPY app ./app
COPY .agents/skills ./.agents/skills

RUN chown -R appuser:appgroup /app /models

USER appuser

VOLUME ["/models"]

EXPOSE 8081

CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8081}"]
