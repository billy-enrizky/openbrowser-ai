"""Main FastAPI application."""

import logging
from contextlib import asynccontextmanager
from uuid import uuid4

from fastapi import FastAPI, WebSocket
from fastapi.middleware.cors import CORSMiddleware

from app.api import projects, tasks
from app.core.config import settings
from app.models.schemas import AvailableModelsResponse, LLMModel
from app.websocket.handler import handle_websocket

# Configure logging
logging.basicConfig(
    level=logging.DEBUG if settings.DEBUG else logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler."""
    logger.info("Starting OpenBrowser Backend API")
    yield
    logger.info("Shutting down OpenBrowser Backend API")


app = FastAPI(
    title="OpenBrowser API",
    description="Backend API for OpenBrowser AI Chat Interface",
    version="0.1.0",
    lifespan=lifespan,
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(tasks.router, prefix="/api/v1")
app.include_router(projects.router, prefix="/api/v1")


@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "name": "OpenBrowser API",
        "version": "0.1.0",
        "status": "running",
    }


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {"status": "healthy"}


@app.get("/api/v1/models", response_model=AvailableModelsResponse)
async def get_available_models():
    """Get available LLM models based on configured API keys."""
    available_models = settings.get_available_models()
    available_providers = settings.get_available_providers()
    
    models = [
        LLMModel(id=m["id"], name=m["name"], provider=m["provider"])
        for m in available_models
    ]
    
    # Determine default model
    default_model = None
    if available_models:
        # Prefer the configured default if its provider is available
        default_provider = None
        if "gemini" in settings.DEFAULT_LLM_MODEL.lower() or "google" in settings.DEFAULT_LLM_MODEL.lower():
            default_provider = "google"
        elif "gpt" in settings.DEFAULT_LLM_MODEL.lower() or "openai" in settings.DEFAULT_LLM_MODEL.lower():
            default_provider = "openai"
        elif "claude" in settings.DEFAULT_LLM_MODEL.lower() or "anthropic" in settings.DEFAULT_LLM_MODEL.lower():
            default_provider = "anthropic"
        
        if default_provider in available_providers:
            default_model = settings.DEFAULT_LLM_MODEL
        else:
            # Use first available model as default
            default_model = available_models[0]["id"]
    
    return AvailableModelsResponse(
        models=models,
        providers=available_providers,
        default_model=default_model,
    )


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for real-time agent communication."""
    client_id = str(uuid4())
    await handle_websocket(websocket, client_id)


@app.websocket("/ws/{client_id}")
async def websocket_endpoint_with_id(websocket: WebSocket, client_id: str):
    """WebSocket endpoint with client-specified ID."""
    await handle_websocket(websocket, client_id)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app.main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.DEBUG,
    )

