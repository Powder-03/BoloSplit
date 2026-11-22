"""FastAPI application entrypoint with lifespan, CORS, and modular routers."""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.config import settings
from app.database import init_db
from app.routers.expenses import router as expenses_router
from app.routers.groups import router as groups_router
from app.routers.self_expenses import router as self_expenses_router
from app.routers.settlements import router as settlements_router
from app.routers.staging import router as staging_router
from app.routers.voice import router as voice_router
from app.routers.web_login import router as web_login_router


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Application lifespan managing database startup extension initialization."""
    # Initialize DB tables and pg_trgm extension
    await init_db()
    yield


app = FastAPI(
    title=settings.APP_NAME,
    description="Voice-first Splitwise-style expense tracker backend with direct voice control and settlements.",
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

# CORS configuration supporting iOS SwiftUI, React Native, and Web clients
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health", tags=["Health"])
async def health_check() -> dict[str, str]:
    """Health check endpoint for Kubernetes / Docker / load balancers."""
    return {
        "status": "healthy",
        "service": settings.APP_NAME,
        "environment": settings.ENVIRONMENT,
    }


# Include routers directly (e.g. /voice, /expenses/self, /expenses, /settlements, /groups, /balances, /friends, /login)
app.include_router(voice_router)
app.include_router(settlements_router)
app.include_router(self_expenses_router)
app.include_router(expenses_router)
app.include_router(groups_router)
app.include_router(staging_router)
app.include_router(web_login_router)

# Also mount under API v1 prefix (e.g. /api/v1/voice, /api/v1/settlements, ...)
app.include_router(voice_router, prefix=settings.API_V1_PREFIX)
app.include_router(settlements_router, prefix=settings.API_V1_PREFIX)
app.include_router(self_expenses_router, prefix=settings.API_V1_PREFIX)
app.include_router(expenses_router, prefix=settings.API_V1_PREFIX)
app.include_router(groups_router, prefix=settings.API_V1_PREFIX)
app.include_router(staging_router, prefix=settings.API_V1_PREFIX)
