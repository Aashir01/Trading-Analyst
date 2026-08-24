"""HTTP API and web frontend.

    uvicorn mfie.api:app --host 0.0.0.0 --port 8000
    python -m mfie serve

``app`` is a fully configured FastAPI application serving both the JSON API
under ``/api`` and the interface at ``/``.
"""

from mfie.api.app import API_VERSION, app, create_app
from mfie.api.service import EngineService, get_service

__all__ = ["API_VERSION", "EngineService", "app", "create_app", "get_service"]
