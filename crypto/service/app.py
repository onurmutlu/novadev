"""
FastAPI Application

NovaDev Crypto API - On-chain intelligence service

Endpoints:
- GET /healthz: Health check
- GET /wallet/{address}/report: Wallet activity report
- GET /docs: OpenAPI documentation (Swagger UI)
- GET /redoc: Alternative API documentation (ReDoc)

Usage:
    # Development (auto-reload)
    uvicorn crypto.service.app:app --reload --port 8000
    
    # Production (multi-worker)
    uvicorn crypto.service.app:app --host 0.0.0.0 --port 8000 --workers 4
    
    # With Gunicorn
    gunicorn crypto.service.app:app --workers 4 --worker-class uvicorn.workers.UvicornWorker --bind 0.0.0.0:8000
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import logging

from .config import settings
from .deps import init_db_pool, init_cache
from .routes import health, wallet

# Configure logging
logging.basicConfig(
    level=getattr(logging, settings.log_level.upper()),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(
    title="NovaDev Crypto API",
    description="On-chain intelligence API for wallet activity analysis",
    version="0.1.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS middleware (disabled by default, enable if needed)
# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"],  # Change to specific origins in production
#     allow_credentials=True,
#     allow_methods=["GET"],
#     allow_headers=["*"],
# )


@app.on_event("startup")
async def startup_event():
    """
    Initialize services on startup
    
    Steps:
    1. Initialize database connection pool
    2. Initialize cache
    3. Log startup message
    """
    logger.info("Starting NovaDev Crypto API...")
    
    # Initialize database pool
    try:
        init_db_pool(settings.db_path)
        logger.info(f"Database pool initialized: {settings.db_path}")
    except Exception as e:
        logger.error(f"Failed to initialize database pool: {e}")
        raise
    
    # Initialize cache
    try:
        init_cache(
            capacity=settings.cache_capacity,
            ttl_seconds=settings.cache_ttl
        )
        logger.info(
            f"Cache initialized: capacity={settings.cache_capacity}, "
            f"ttl={settings.cache_ttl}s"
        )
    except Exception as e:
        logger.error(f"Failed to initialize cache: {e}")
        raise
    
    logger.info("✅ NovaDev Crypto API started successfully")
    logger.info(f"   Chain ID: {settings.chain_id}")
    logger.info(f"   Docs: http://localhost:8000/docs")


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    logger.info("Shutting down NovaDev Crypto API...")
    logger.info("✅ Shutdown complete")


# Include routers
app.include_router(health.router)
app.include_router(wallet.router)


# Root endpoint
@app.get("/", include_in_schema=False)
async def root():
    """Root endpoint redirect to docs"""
    from fastapi.responses import RedirectResponse
    return RedirectResponse(url="/docs")


if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        "crypto.service.app:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
