from typing import Union, List
from fastapi import APIRouter, HTTPException
import httpx
import numpy as np
from pydantic import BaseModel, field_validator
from transformers import AutoTokenizer
from prometheus_client import Summary, Counter, Gauge
from .logging_config import logger
#prometheus custom metrics
INFERENCE_TIME = Summary('inference_time_seconds', 'Time spent on model inference')
BATCH_SIZE = Gauge('batch_size', 'Number of texts per request')
SAFE_COUNTER = Counter('safe_predictions_total', 'Number of SAFE predictions')
UNSAFE_COUNTER = Counter('unsafe_predictions_total', 'Number of UNSAFE predictions')



inference = APIRouter()

TRITON_URL = "http://localhost:8000/v2/models/fast_llm_guard/infer"
base_model_name = "distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(base_model_name)


class InferenceRequest(BaseModel):
    texts: Union[str, List[str]]
    @field_validator("texts")
    def not_empty(cls, v):
        if isinstance(v, str) and not v.strip():
            raise ValueError("Input text cannot be empty.")
        if isinstance(v, list):
            if not v:
                raise ValueError("Input list cannot be empty.")
            if any((not isinstance(t, str) or not t.strip()) for t in v):
                raise ValueError("All texts must be non-empty strings.")
        return v

@inference.post("/predict")
async def predict(request: InferenceRequest):
    with INFERENCE_TIME.time():
        texts = [request.texts] if isinstance(request.texts, str) else request.texts
        BATCH_SIZE.set(len(texts))
        
        try:
            inputs = tokenizer(texts, return_tensors='np', truncation=True, padding=True, max_length=128)
            input_ids = inputs["input_ids"].astype(np.int64)
            attention_mask = inputs["attention_mask"].astype(np.int64)
        except Exception as e:
            logger.error(f"Tokenization error: {str(e)}")
            raise HTTPException(status_code=400, detail=f"Tokenization error: {str(e)}")

        payload = {
            "inputs": [
                {
                    "name": "input_ids",
                    "shape": input_ids.shape,
                    "datatype": "INT64",
                    "data": input_ids.tolist()
                },
                {
                    "name": "attention_mask",
                    "shape": attention_mask.shape,
                    "datatype": "INT64",
                    "data": attention_mask.tolist()
                }
            ],
            "outputs": [{"name": "logits"}]
        }

        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(TRITON_URL, json=payload, timeout=10)
                response.raise_for_status()
                result = response.json()
        except httpx.HTTPStatusError as e:
            logger.error(f"Failed Request to Triton server: {str(e)}")
            raise HTTPException(status_code=502, detail=f"Triton server error: {e.response.text}")
        except httpx.RequestError as e:
            logger.critical(f"Triton server is unreachable: {str(e)}")
            raise HTTPException(status_code=504, detail=f"Triton server not reachable: {str(e)}")
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Unexpected error: {str(e)}")

        try:
            logits = np.array(result["outputs"][0]["data"]).reshape(result["outputs"][0]["shape"])
            probs = np.exp(logits) / np.exp(logits).sum(axis=-1, keepdims=True)
            #preds = np.argmax(probs, axis=-1)
            preds = ["UNSAFE" if prob[1] > 0.2 else "SAFE" for prob in probs]
        
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Postprocessing error: {str(e)}")

        SAFE_COUNTER.inc(sum(1 for p in preds if p == "SAFE"))
        UNSAFE_COUNTER.inc(sum(1 for p in preds if p == "UNSAFE"))
        
        return {
            "predictions": preds,
            "probabilities": probs.tolist()
        }