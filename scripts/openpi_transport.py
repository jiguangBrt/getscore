"""Use the existing RTC-capable OpenPI client for both request protocols."""

from __future__ import annotations

import json
import math
import os
import threading
import time
from pathlib import Path

from openpi_client.websocket_client_policy import WebsocketClientPolicy


class OpenPiTransport:
    """Serialize requests; apply the same timeouts to sync and rtc_v1 messages.

    A timeout closes the connection in the upstream client. Reconnect only for
    the next explicit request; never replay a timed-out action request.
    """

    def __init__(
        self, host, port, *, connect_timeout_s, request_timeout_s, timing_path=None
    ):
        for value in (connect_timeout_s, request_timeout_s):
            if not math.isfinite(value) or value <= 0:
                raise ValueError("OpenPI timeouts must be finite and positive")
        for key in ("NO_PROXY", "no_proxy"):
            os.environ[key] = ",".join(
                filter(None, [os.environ.get(key), host, "127.0.0.1", "localhost"])
            )
        self._kwargs = {
            "host": host,
            "port": port,
            "connect_timeout_s": connect_timeout_s,
            "request_timeout_s": request_timeout_s,
        }
        self._client = None
        self._lock = threading.Lock()
        self._timing_path = Path(timing_path) if timing_path else None

    def _connect(self):
        if self._client is None:
            self._client = WebsocketClientPolicy(**self._kwargs)
        return self._client

    def metadata(self):
        with self._lock:
            return self._connect().get_server_metadata()

    def infer(self, request):
        """Forward either an ordinary observation or an intact RTC envelope."""
        with self._lock:
            started = time.monotonic()
            timing = {
                "time_unix": time.time(),
                "mode": "rtc" if request.get("request_type") == "rtc_v1" else "sync",
                "request_timeout_s": self._kwargs["request_timeout_s"],
            }
            try:
                result = self._connect().infer(request)
                timing.update(
                    ok=result.get("ok", True),
                    client_timing=result.get("client_timing"),
                    server_timing=result.get(
                        "server_timing", result.get("data", {}).get("server_timing")
                    ),
                )
                return result
            except Exception as exc:
                timing.update(ok=False, error_type=type(exc).__name__, error=str(exc))
                if self._client is not None:
                    self._client.close()
                    self._client = None
                raise
            finally:
                timing["wall_ms"] = (time.monotonic() - started) * 1000
                if self._timing_path:
                    self._timing_path.parent.mkdir(parents=True, exist_ok=True)
                    with self._timing_path.open("a") as output:
                        output.write(json.dumps(timing) + "\n")

    def close(self):
        with self._lock:
            if self._client is not None:
                self._client.close()
                self._client = None
