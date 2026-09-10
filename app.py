"""FastAPI Application entry point for Generic CAM."""

from pathlib import Path
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

from cam_engine.api import router as cam_router

app = FastAPI(
    title="Generic CAM Studio API",
    description="Geometry-driven CAM toolpath planning and G-code generation engine",
    version="0.1.0",
)

# Enable CORS for local dev / client integrations
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include CAM API Router
app.include_router(cam_router)

# Mount static frontend directory
STATIC_DIR = Path(__file__).parent / "static"
STATIC_DIR.mkdir(exist_ok=True)

app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.get("/")
async def serve_index():
    index_file = STATIC_DIR / "index.html"
    if index_file.exists():
        return FileResponse(index_file)
    return {"status": "Generic CAM API is running", "docs": "/docs"}


if __name__ == "__main__":
    import os
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("app:app", host="127.0.0.1", port=port, reload=True)
