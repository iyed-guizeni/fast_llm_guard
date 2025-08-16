FROM python:3.11-slim

WORKDIR /app

COPY fast_llm_guard /app/fast_llm_guard
COPY requirements.txt /app/requirements.txt

RUN pip install --no-cache-dir -r requirements.txt

CMD ["uvicorn", "fast_llm_guard.inference.api:app", "--host", "0.0.0.0", "--port", "9000"]