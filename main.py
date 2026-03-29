from fastapi import FastAPI
from endpoints import router
from dotenv import load_dotenv

load_dotenv()

app = FastAPI(
    title="Multi-Agent Research Assistant",
    description="AI-powered research system using multiple specialized agents",
    version="1.0.0"
)

app.include_router(router)

@app.get("/")
def root():
    return {
        "status": "ok",
        "docs": "/docs",
        "endpoints": ["/research/", "/upload/", "/files/", "/clear/"]
    }