---
name: sports-inference
description: Run, query, and test the SportsAgenticInference FastAPI service managing llama-server and microsoft/Phi-4-mini-instruct. Use when starting the local inference server, running OpenAI-compatible chat or MCP tool-calling requests on port 8081, checking model/process health, or running its test suite.
---

# SportsAgenticInference Skill

This repository contains the Python FastAPI OpenAI-compatible inference gateway (`app/main.py`) that manages:
1. **`microsoft/Phi-4-mini-instruct` (GGUF)** downloaded and cached automatically via `huggingface_hub`.
2. **`llama-server`** (`llama.cpp` HTTP server) spawned on internal port `127.0.0.1:8082` with `--jinja` enabled.
3. **OpenAI-Compatible Gateway on `:8081`** (`GET /health`, `GET /v1/models`, `POST /v1/chat/completions`) designed for seamless integration with `SportsAgenticAgent` (`http://localhost:8081/v1` locally, `http://llama-server:8081/v1` in Kubernetes).

## Running the Inference Service

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8081 --reload
```

### Docker / Docker Compose
```bash
docker compose up --build
```

### Helm (Kubernetes)
```bash
helm lint charts/sportsagentic-inference
helm upgrade --install llama-server charts/sportsagentic-inference \
  --namespace sportsagentic --create-namespace
```

## Querying the Inference Server

### 1. Check Health (`GET /health`)
```bash
curl -s http://localhost:8081/health
```

### 2. List Models (`GET /v1/models`)
```bash
curl -s http://localhost:8081/v1/models
```

### 3. Chat Completion (`POST /v1/chat/completions`)
```bash
curl -s http://localhost:8081/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "local-model",
    "messages": [
      {"role": "system", "content": "You are a college football assistant."},
      {"role": "user", "content": "Who is the head coach of Ole Miss?"}
    ]
  }'
```

### 4. Tool-Calling Completion with SportsAgentic MCP Tools
```bash
curl -s http://localhost:8081/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "local-model",
    "messages": [
      {"role": "user", "content": "What was the score of the Ole Miss game?"}
    ],
    "tools": [
      {
        "type": "function",
        "function": {
          "name": "get_cfb_scores",
          "description": "Get college football scores",
          "parameters": {
            "type": "object",
            "properties": {
              "team": {"type": "string"}
            }
          }
        }
      }
    ]
  }'
```

## Running the Test Suite

```bash
pytest -v
```
