"""Loopback-only HTTP application for reading user-selected experiment folders."""

from __future__ import annotations

import argparse
import hashlib
import json
import mimetypes
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlsplit

from run_viewer.parser import parse_run, read_object

ROOT = Path(__file__).resolve().parents[1]
STATIC = Path(__file__).parent / "static"


class ViewerServer(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, address, default_path):
        self.default_path = str(default_path)
        self.runs = {}
        self.cache = {}
        self.lock = threading.Lock()
        super().__init__(address, ViewerHandler)

    def register(self, path):
        key = hashlib.sha256(str(path).encode()).hexdigest()[:24]
        with self.lock:
            self.runs[key] = path
        return key

    def report(self, path):
        signature = tuple(
            (file.name, file.stat().st_mtime_ns, file.stat().st_size)
            for file in sorted(path.iterdir())
            if file.is_file()
            and (
                file.name
                in {"configuration.json", "states.json", "run.log", "episode.mp4"}
                or file.name.endswith((".stream.jsonl", ".json"))
            )
        )
        with self.lock:
            cached = self.cache.get(path)
        if cached and cached[0] == signature:
            return cached[1]
        report = parse_run(path)
        report["run_id"] = self.register(path)
        with self.lock:
            # Bound retained parsed text when browsing many large experiments.
            if len(self.cache) >= 8:
                self.cache.pop(next(iter(self.cache)))
            self.cache[path] = (signature, report)
        return report


class ViewerHandler(BaseHTTPRequestHandler):
    server: ViewerServer

    def log_message(self, message, *args):
        # Avoid printing directory paths or the contents of user logs.
        if args and isinstance(args[0], str) and args[0].startswith('"GET /api/asset'):
            return
        super().log_message(message, *args)

    def permitted(self):
        host = self.headers.get("Host", "")
        allowed = {
            f"127.0.0.1:{self.server.server_port}",
            f"localhost:{self.server.server_port}",
        }
        origin = self.headers.get("Origin")
        return host in allowed and (
            not origin or origin in {"http://" + value for value in allowed}
        )

    def send_headers(self, code, content_type, length=None):
        self.send_response(code)
        self.send_header("Content-Type", content_type)
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Cache-Control", "no-store")
        self.send_header(
            "Content-Security-Policy",
            "default-src 'self'; script-src 'self'; style-src 'self' 'unsafe-inline'; img-src 'self' data:; media-src 'self'; connect-src 'self'; frame-ancestors 'none'",
        )
        if length is not None:
            self.send_header("Content-Length", str(length))

    def json_response(self, payload, code=200):
        body = json.dumps(payload, ensure_ascii=False, allow_nan=False).encode()
        self.send_headers(code, "application/json; charset=utf-8", len(body))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if not self.permitted():
            self.json_response({"error": "仅允许本机页面访问。"}, 403)
            return
        url = urlsplit(self.path)
        if url.path == "/api/bootstrap":
            self.json_response({"default_path": self.server.default_path})
            return
        if url.path == "/api/asset":
            self.asset(parse_qs(url.query))
            return
        name = {"/": "index.html", "/app.js": "app.js", "/style.css": "style.css"}.get(
            url.path
        )
        if name is None:
            self.json_response({"error": "页面不存在。"}, 404)
            return
        path = STATIC / name
        body = path.read_bytes()
        mime = {".html": "text/html", ".js": "text/javascript", ".css": "text/css"}[
            path.suffix
        ]
        self.send_headers(200, mime + "; charset=utf-8", len(body))
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self):
        if not self.permitted():
            self.json_response({"error": "仅允许本机页面访问。"}, 403)
            return
        if self.path != "/api/open":
            self.json_response({"error": "接口不存在。"}, 404)
            return
        if self.headers.get_content_type() != "application/json":
            self.json_response({"error": "请使用页面提交目录。"}, 415)
            return
        try:
            size = int(self.headers.get("Content-Length", 0))
            if not 0 < size <= 8192:
                raise ValueError("目录请求过长或为空。")
            payload = json.loads(self.rfile.read(size))
            if not isinstance(payload, dict):
                raise TypeError("目录请求应为 JSON 对象。")
            raw_path = payload.get("path", "")
            if not isinstance(raw_path, str) or not raw_path.strip():
                raise ValueError("请输入本机的实验目录。")
            path = Path(raw_path.strip()).expanduser()
            path = (ROOT / path).resolve() if not path.is_absolute() else path.resolve()
            if not path.is_dir():
                raise ValueError("找不到该目录，请检查本机路径是否正确。")
            if (
                (path / "configuration.json").is_file()
                or next(path.glob("*.stream.jsonl"), None)
                or next(path.glob("transcript_*.json"), None)
            ):
                self.json_response({"report": self.server.report(path)})
                return
            runs = []
            for child in sorted(path.iterdir(), reverse=True):
                if not child.is_dir() or child.name.startswith((".", "_")):
                    continue
                marker = child / "configuration.json"
                if not marker.is_file():
                    continue
                try:
                    config = read_object(marker)
                    runs.append(
                        {
                            "name": child.name,
                            "path": str(child),
                            "model": config.get("model", config.get("planner")),
                            "success": config.get("task_success"),
                            "modified": marker.stat().st_mtime,
                        }
                    )
                except (ValueError, TypeError, OSError):
                    continue
                if len(runs) == 300:
                    break
            if not runs:
                raise ValueError(
                    "该目录里没有实验记录。请选择包含 configuration.json 的实验目录，或它的上级 outputs 目录。"
                )
            self.json_response(
                {
                    "runs": sorted(runs, key=lambda row: row["modified"], reverse=True),
                    "path": str(path),
                }
            )
        except (ValueError, TypeError, KeyError) as exc:
            self.json_response({"error": str(exc)}, 400)
        except OSError:
            self.json_response({"error": "无法读取该目录，请检查文件权限。"}, 403)

    def asset(self, query):
        try:
            key = query.get("run", [""])[0]
            with self.server.lock:
                root = self.server.runs.get(key)
            if root is None:
                raise FileNotFoundError
            relative = unquote(query.get("path", [""])[0])
            path = (root / relative).resolve()
            if (
                not path.is_relative_to(root)
                or not path.is_file()
                or path.suffix.lower() not in {".png", ".jpg", ".jpeg", ".webp", ".mp4"}
            ):
                raise FileNotFoundError
            size = path.stat().st_size
            start, end, status = 0, size - 1, 200
            range_header = self.headers.get("Range")
            if range_header:
                import re

                match = re.fullmatch(r"bytes=(\d*)-(\d*)", range_header)
                if not match or not any(match.groups()):
                    raise ValueError
                if match[1]:
                    start = int(match[1])
                    end = min(int(match[2]), end) if match[2] else end
                else:
                    start = max(0, size - int(match[2]))
                if start > end or start >= size:
                    raise ValueError
                status = 206
            mime = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
            self.send_headers(status, mime, end - start + 1)
            self.send_header("Accept-Ranges", "bytes")
            if status == 206:
                self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
            self.end_headers()
            with path.open("rb") as stream:
                stream.seek(start)
                remaining = end - start + 1
                while remaining:
                    chunk = stream.read(min(remaining, 1024 * 1024))
                    if not chunk:
                        break
                    self.wfile.write(chunk)
                    remaining -= len(chunk)
        except FileNotFoundError:
            self.json_response({"error": "本轮没有保存该图片或视频。"}, 404)
        except ValueError:
            self.json_response({"error": "视频范围无效。"}, 416)
        except (BrokenPipeError, ConnectionResetError):
            pass


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--default-path", type=Path, default=ROOT / "outputs")
    args = parser.parse_args()
    server = ViewerServer(("127.0.0.1", args.port), args.default_path)
    print(f"服务已启动，请在浏览器打开：http://127.0.0.1:{server.server_port}", flush=True)
    print("保持此终端运行；按 Ctrl+C 停止服务。无需提前激活 .venv。", flush=True)
    print("仅在本机读取文件，不调用模型，也不上传数据。", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
