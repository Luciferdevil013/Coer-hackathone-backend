from __future__ import annotations

import logging

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.routes import router
from app.db import init_engine

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")


def create_app() -> FastAPI:
    init_engine()
    app = FastAPI(title="DataGuardian AI", version="1.0.0",
                  description="AI-powered data lifecycle management: KEEP · CLEAN · ARCHIVE · SECURE · DELETE")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
        allow_methods=["*"], allow_headers=["*"],
    )
    app.include_router(router)
    return app


app = create_app()
