"""Expose the FastAPI application as backend/app.main.

This module keeps start_project.ps1 and deployment commands based on
``app.main:app`` compatible with EchoSafe's root-level backend modules.
"""

try:
    from ..main import app
except ImportError:
    # When uvicorn is started with backend/ as the working directory,
    # app.main is loaded as a top-level app package instead of backend.app.
    from main import app

__all__ = ["app"]
