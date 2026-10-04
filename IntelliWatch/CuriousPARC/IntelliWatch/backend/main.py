from contextlib import asynccontextmanager
from pathlib import Path
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles

from configs.settings import get_settings
from configs.logging_config import setup_logging
from backend.api.routes import router
from backend.api.auth_routes import auth_router

settings = get_settings()
PROJECT_ROOT = Path(__file__).resolve().parent.parent
WORKSPACE_ROOT = PROJECT_ROOT.parents[1]
UI_DIR = WORKSPACE_ROOT / "UI"
if not UI_DIR.exists():
    UI_DIR = WORKSPACE_ROOT / "ui"

FRONTEND_DIR = UI_DIR if UI_DIR.exists() else (PROJECT_ROOT / "frontend")
LEGACY_FRONTEND_DIR = PROJECT_ROOT / "frontend"


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    FastAPI lifespan manager handling startup and shutdown events cleanly.
    """
    logger = setup_logging(settings.LOG_LEVEL, settings.LOG_FILE)
    logger.info(f"Starting {settings.PROJECT_NAME} v{settings.VERSION} [{settings.ENVIRONMENT}]")
    yield
    logger.info(f"Shutting down {settings.PROJECT_NAME}...")
    try:
        from backend.services.camera_manager import get_camera_manager
        get_camera_manager().stop_all()
    except Exception as e:
        logger.warning(f"Error during camera manager shutdown: {e}")


app = FastAPI(
    title=f"{settings.PROJECT_NAME} API",
    version=settings.VERSION,
    description="IntelliWatch: AI-Powered Scene Understanding for Industrial Safety & Monitoring",
    lifespan=lifespan,
)

import logging
from fastapi import Request
from fastapi.responses import JSONResponse

logger = logging.getLogger("intelliwatch.app")

# Configure CORS with explicit origins from settings
cors_origins = settings.CORS_ORIGINS
allow_all_origins = "*" in cors_origins or len(cors_origins) == 0

app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_origins if not allow_all_origins else ["*"],
    allow_credentials=not allow_all_origins,
    allow_methods=["GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS"],
    allow_headers=["*"],
)


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception):
    """Prevents leakage of internal stack traces and server file paths in production."""
    logger.error(f"Unhandled server error at {request.method} {request.url.path}: {exc}", exc_info=True)
    if settings.DEBUG:
        return JSONResponse(
            status_code=500,
            content={"detail": f"Internal server error: {str(exc)}", "type": type(exc).__name__},
        )
    return JSONResponse(
        status_code=500,
        content={"detail": "An internal server error occurred. Please contact system administrator."},
    )

# Register API routes
app.include_router(router)
app.include_router(auth_router)

# Mount UI ES module source tree if exists
if (FRONTEND_DIR / "src").exists():
    app.mount("/src", StaticFiles(directory=str(FRONTEND_DIR / "src")), name="ui-src")

# Mount legacy frontend assets for backwards compatibility
if LEGACY_FRONTEND_DIR.exists():
    app.mount("/frontend", StaticFiles(directory=str(LEGACY_FRONTEND_DIR)), name="legacy-frontend")


@app.get("/styles.css", include_in_schema=False)
async def serve_styles():
    f = FRONTEND_DIR / "styles.css"
    if f.exists():
        return FileResponse(str(f), media_type="text/css")
    raise HTTPException(status_code=404, detail="styles.css not found")


@app.get("/app.js", include_in_schema=False)
async def serve_app_js():
    f = FRONTEND_DIR / "app.js"
    if f.exists():
        return FileResponse(str(f), media_type="application/javascript")
    raise HTTPException(status_code=404, detail="app.js not found")


@app.get("/favicon.ico", include_in_schema=False)
async def serve_favicon():
    f = FRONTEND_DIR / "favicon.ico"
    if f.exists():
        return FileResponse(str(f))
    return Response(status_code=204)


@app.get("/eval", tags=["Dashboard"])
async def serve_eval_dashboard():
    """
    Serves the AI Performance Validation Dashboard — full-screen evaluation metrics page.
    Metrics are sourced exclusively from reports/person_ppe_evaluation.json and
    reports/model_benchmark.json via /api/v1/model/* endpoints. No values are hardcoded.
    """
    eval_file = LEGACY_FRONTEND_DIR / "eval-dashboard.html"
    if eval_file.exists():
        return FileResponse(str(eval_file), media_type="text/html")
    raise HTTPException(status_code=404, detail="eval-dashboard.html not found in frontend/")


@app.get("/", tags=["Navigation"])
@app.get("/landing", tags=["Navigation"])
@app.get("/dashboard", tags=["Dashboard"])
@app.get("/product", tags=["Navigation"])
@app.get("/demo", tags=["Navigation"])
@app.get("/app", tags=["Dashboard"])
@app.get("/app/{full_path:path}", tags=["Dashboard"])
async def serve_spa_frontend(full_path: str = ""):
    """
    Serves the authoritative IntelliWatch single-page interface and handles SPA client-side routing.
    """
    index_file = FRONTEND_DIR / "index.html"
    if index_file.exists():
        return FileResponse(str(index_file))
    return {"status": "ok", "project": settings.PROJECT_NAME, "message": "IntelliWatch frontend ready"}

