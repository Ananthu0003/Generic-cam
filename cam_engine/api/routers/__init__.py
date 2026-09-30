"""Routers package for VexCAM API.

Each module exposes a thin FastAPI ``APIRouter`` that parses HTTP input,
delegates to a ``cam_engine.api.services`` function, and serialises the
response.  No business logic lives here.
"""
