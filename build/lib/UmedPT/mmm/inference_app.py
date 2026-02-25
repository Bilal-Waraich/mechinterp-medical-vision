from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING
from urllib.parse import urljoin

import logfire
import redis
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

try:
    from fastapi import FastAPI
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.responses import FileResponse
except ImportError:
    if not TYPE_CHECKING:
        FastAPI, CORSMiddleware, FileResponse = None, None, None

from mmm.api.DLModel import DLModel
from mmm.api.LSModel import LSModel
from mmm.settings import mtl_settings


class APISettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="MMMAPI_")

    app_base: str = Field(
        "http://localhost:8000",
        description="The base URL of this service",
    )

    allow_cors: bool = True


def build_app() -> FastAPI:
    settings = APISettings()
    app = FastAPI(
        description=f"""
    See the [API documentation]({urljoin(settings.app_base, "/docs")}) for more information.

    For a complete example, see `medicalmultitaskmodeling/examples/model_api.ipynb`.
    """
    )
    logfire.configure(service_name="m3_api")
    logfire.instrument_fastapi(app)

    if settings.allow_cors:
        app.add_middleware(
            CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"]
        )

    app.include_router((dlmodel := DLModel(urljoin(settings.app_base, "/peft"))).router, prefix="/peft")
    app.include_router((lsmodel := LSModel(urljoin(settings.app_base, "/labelstudio"))).router, prefix="/labelstudio")

    @app.get("/")
    async def status():
        try:
            kv_available = mtl_settings.kv.ping()
        except redis.exceptions.ConnectionError:
            kv_available = False
        return {
            "kv_status": kv_available,
            "kv_url": mtl_settings.redis_url,
        }

    @app.get("/favicon.ico", include_in_schema=False)
    async def api_favicon():
        return FileResponse(Path(__file__).parent.joinpath("resources").joinpath("api_favicon.png"))

    return app


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(build_app(), host="0.0.0.0", port=8000, reload=True)
