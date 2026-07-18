from fastapi import FastAPI
from ai_inference_optimization_platform.config.settings import settings


print(settings.app_name)
print(settings.default_model)
print(settings.redis_url)

app = FastAPI(
    title="AI Inference Optimization Platform",
    version="0.1.0",
)


@app.get("/")
async def root():
    return {
        "message": "AI Inference Optimization Platform"
    }


@app.get("/health")
async def health():
    return {
        "status": "healthy"
    }

