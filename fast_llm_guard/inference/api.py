from fastapi import FastAPI
from fast_llm_guard.inference.routes import inference
from prometheus_fastapi_instrumentator import Instrumentator

app = FastAPI() 


app.include_router(inference.inference)
Instrumentator().instrument(app).expose(app)
#collect metrics-> middleware for measure -> expose /metrics