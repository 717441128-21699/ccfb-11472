"""启动管理页面：py scripts/run_web.py，默认 http://127.0.0.1:5000"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dataprobe import storage
from dataprobe.web import create_app


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=5000)
    args = ap.parse_args()

    conn = storage.connect()
    storage.init_db(conn)
    app = create_app()
    print(f"DataProbe 管理页面启动：http://{args.host}:{args.port}")
    app.run(host=args.host, port=args.port, debug=False)


if __name__ == "__main__":
    main()
