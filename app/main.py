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
from app.routers.voice import router as voice_router
from app.routers.web_login import router as web_login_router


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Application lifespan managing database startup initialization."""
    await init_db()
    yield


app = FastAPI(
    title=settings.APP_NAME,
    description="Voice-first Splitwise-style expense tracker backend with single-pass voice control and settlements.",
    version="2.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

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


core_routers = [
    voice_router,
    settlements_router,
    self_expenses_router,
    expenses_router,
    groups_router,
]

for r in core_routers:
    app.include_router(r)
    app.include_router(r, prefix=settings.API_V1_PREFIX)

if settings.ENVIRONMENT.strip().lower() != "production":
    app.include_router(web_login_router)
