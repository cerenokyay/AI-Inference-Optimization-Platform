from fastapi import FastAPI

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