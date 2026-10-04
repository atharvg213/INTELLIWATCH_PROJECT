import uvicorn
from configs.settings import get_settings

if __name__ == "__main__":
    settings = get_settings()
    print(f"Starting {settings.PROJECT_NAME} API server...")
    print(f"Listening on http://{settings.API_HOST}:{settings.API_PORT}")
    print(f"Interactive Docs: http://{settings.API_HOST}:{settings.API_PORT}/docs")
    print(f"Health Check: http://{settings.API_HOST}:{settings.API_PORT}/health")
    print(f"Operator Dashboard: http://{settings.API_HOST}:{settings.API_PORT}/dashboard")
    uvicorn.run(
        "backend.main:app",
        host=settings.API_HOST,
        port=settings.API_PORT,
        reload=settings.DEBUG,
    )
