import logging
import time
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from starlette.middleware.base import BaseHTTPMiddleware

# Ensure uploads directory exists before mounting
Path("uploads").mkdir(exist_ok=True)

from .database import init_db
from .routers import auth, datasets, eda, jobs, workspaces, extra, sql_editor, join_builder, sources, warehouse, ai as ai_router, feedback as feedback_router

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger("autoeda")


@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    logger.info("AutoEDA backend started — DB initialised")
    yield


app = FastAPI(
    title="Jman Group AutoEDA API",
    description="Production-grade Automated EDA Platform — Backend API",
    version="2.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://localhost:3001",
        "http://localhost:3002",
        "http://127.0.0.1:3000",
        "https://autoeda-frontend-k7rt.vercel.app",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        start = time.perf_counter()
        response = await call_next(request)
        duration_ms = round((time.perf_counter() - start) * 1000)
        logger.info(
            "%s %s %s %dms",
            request.method,
            request.url.path,
            response.status_code,
            duration_ms,
        )
        return response


app.add_middleware(RequestLoggingMiddleware)

app.mount("/uploads", StaticFiles(directory="uploads"), name="uploads")


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error("Unhandled exception: %s", exc, exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"detail": str(exc), "code": "INTERNAL_ERROR"},
    )


app.include_router(auth.router, prefix="/api/v1")
app.include_router(workspaces.router, prefix="/api/v1")
app.include_router(datasets.router, prefix="/api/v1")
app.include_router(eda.router, prefix="/api/v1")
app.include_router(jobs.router, prefix="/api/v1")
app.include_router(extra.router, prefix="/api/v1")
app.include_router(sql_editor.router, prefix="/api/v1")
app.include_router(join_builder.router, prefix="/api/v1")
app.include_router(sources.router, prefix="/api/v1")
app.include_router(warehouse.router, prefix="/api/v1")
app.include_router(ai_router.router, prefix="/api/v1")
app.include_router(feedback_router.router, prefix="/api/v1")


@app.get("/api/v1/health")
def health():
    return {"status": "ok", "version": "2.0.0", "service": "Jman AutoEDA"}
