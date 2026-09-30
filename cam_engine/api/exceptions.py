"""Reusable HTTP exception factories for VexCAM API.

Import these helpers in routers to keep error-raising DRY and consistent.
"""

from fastapi import HTTPException


def cam_error(code: str, message: str, status_code: int = 400) -> HTTPException:
    """Wrap a CamError-style code+message as an HTTPException."""
    return HTTPException(status_code=status_code, detail=f"[{code}] {message}")


def not_found(resource: str, resource_id: str) -> HTTPException:
    """Return a 404 HTTPException for a missing resource."""
    return HTTPException(
        status_code=404,
        detail=f"{resource} '{resource_id}' not found",
    )


def bad_request(detail: str) -> HTTPException:
    """Return a 400 HTTPException with an arbitrary detail string."""
    return HTTPException(status_code=400, detail=detail)


def unprocessable(detail: str) -> HTTPException:
    """Return a 422 HTTPException for file-processing failures."""
    return HTTPException(status_code=422, detail=detail)
