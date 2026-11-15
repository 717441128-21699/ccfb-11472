"""Flask 管理页面：来源登记、表画像、单列详情、漂移事件、阈值配置。"""
from __future__ import annotations

from flask import Flask

from .. import storage


def create_app(db_path: str | None = None) -> Flask:
    app = Flask(__name__)
    app.config["DB_PATH"] = db_path

    def get_conn():
        return storage.connect(app.config["DB_PATH"])

    from . import routes
    routes.register(app, get_conn)
    return app
