
# Fast LLM Guard

## Overview
Fast LLM Guard is a scalable, production-ready NLP inference platform with distributed training, optimization, and real-time monitoring.

---

## Quickstart Guide

### 1. Clone the Repository

```bash
git clone https://github.com/iyed-guizeni/fast_llm_guard.git
cd fast_llm_guard
```

---

### 2. Download the ONNX Model from Hugging Face

- Go to [Hugging Face Model Repo](https://huggingface.co/iyed-guizeni/fast_llm_guard)
- Download `model.onnx` from the repository.

Or use the CLI:

```bash
pip install huggingface_hub
huggingface-cli login  # if private
huggingface-cli download iyed-guizeni/fast_llm_guard model.onnx
```

---

### 3. Place the Model File

Move the downloaded `model.onnx` to:

```
model_repository/fast_llm_guard/1/model.onnx
```

Create folders if they don’t exist.

---

### 4. Run the Project with Docker Compose

```bash
docker-compose up --build
```

This will start:
- Triton Inference Server (serving your ONNX model)
- FastAPI (NLP API)
- Prometheus (metrics)
- Grafana (monitoring dashboards)

---

### 5. Test the API

Send a request to the FastAPI endpoint:

```bash
curl -X POST "http://localhost:9000/predict" -H "Content-Type: application/json" -d '{"texts": ["Your input text here"]}'
```

---

### 6. Monitor Metrics

- **Grafana:** [http://localhost:3000](http://localhost:3000)
- **Prometheus:** [http://localhost:9090](http://localhost:9090)

---

## Notes

- Make sure Docker is installed and running.
- You can customize model paths and endpoints in the config files.

---

## License

MIT
