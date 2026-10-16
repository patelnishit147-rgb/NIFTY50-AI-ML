
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.routers import market

app = FastAPI(
    title="MarketMind AI API",
    description="AI-powered market analytics backend",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
    "http://localhost:5173",
    "http://localhost:5174",
    "http://127.0.0.1:5173",
    "http://127.0.0.1:5174",
    "https://nifty50-ai-dashboard.onrender.com",
],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register the market router
# The /api/market prefix is already defined in market.py
app.include_router(market.router)


@app.get("/")
def home():
    return {
        "message": "Welcome to MarketMind AI",
        "status": "running",
    }


@app.get("/api/health")
def health_check():
    return {
        "status": "healthy",
        "application": "MarketMind AI",
    }
