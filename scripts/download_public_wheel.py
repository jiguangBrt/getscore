"""Fetch public PyPI wheels in resumable ranges and verify the published SHA256."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import threading
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path


def download(package, version, destination, workers=16):
    with urllib.request.urlopen(
        f"https://pypi.org/pypi/{package}/{version}/json", timeout=45
    ) as response:
        metadata = json.load(response)
    candidates = [
        item
        for item in metadata["urls"]
        if item["filename"].endswith(".whl")
        and (
            item["filename"].endswith("none-any.whl")
            or "manylinux" in item["filename"]
            and "x86_64" in item["filename"]
            and ("py3-none-" in item["filename"] or "cp311-cp311-" in item["filename"])
        )
    ]
    if len(candidates) != 1:
        raise RuntimeError(
            f"Expected one compatible wheel for {package}: {[x['filename'] for x in candidates]}"
        )
    item = candidates[0]
    destination.mkdir(parents=True, exist_ok=True)
    target = destination / item["filename"]

    def valid(path):
        if not path.is_file() or path.stat().st_size != item["size"]:
            return False
        with path.open("rb") as source:
            return (
                hashlib.file_digest(source, "sha256").hexdigest()
                == item["digests"]["sha256"]
            )

    if valid(target):
        print("verified", target, flush=True)
        return
    part = target.with_suffix(".part")
    state_path = target.with_suffix(".ranges.json")
    chunk = 512 * 1024
    if state_path.exists():
        state = json.loads(state_path.read_text())
        if state["sha256"] != item["digests"]["sha256"]:
            raise RuntimeError("PyPI artifact changed")
    else:
        state = {"sha256": item["digests"]["sha256"], "done": []}
        with part.open("wb") as output:
            output.truncate(item["size"])
    lock = threading.Lock()

    def fetch(start):
        end = min(start + chunk, item["size"]) - 1
        for attempt in range(8):
            try:
                request = urllib.request.Request(
                    item["url"], headers={"Range": f"bytes={start}-{end}"}
                )
                with urllib.request.urlopen(request, timeout=45) as response:
                    if (
                        response.status != 206
                        or response.headers.get("Content-Range")
                        != f"bytes {start}-{end}/{item['size']}"
                    ):
                        raise RuntimeError("PyPI CDN did not honor Range")
                    data = response.read(end - start + 2)
                if len(data) != end - start + 1:
                    raise RuntimeError("Truncated wheel range")
                fd = os.open(part, os.O_WRONLY)
                try:
                    offset = 0
                    while offset < len(data):
                        offset += os.pwrite(fd, data[offset:], start + offset)
                finally:
                    os.close(fd)
                with lock:
                    state["done"].append(start)
                    temporary = state_path.with_suffix(".tmp")
                    temporary.write_text(json.dumps(state))
                    temporary.replace(state_path)
                    if len(state["done"]) % 32 == 0:
                        print(
                            package,
                            "MiB",
                            round(len(state["done"]) * chunk / 1024**2, 1),
                            flush=True,
                        )
                return
            except Exception as exc:
                print(package, "retry", start, attempt + 1, str(exc), flush=True)
                if attempt == 7:
                    raise
                time.sleep(min(2**attempt, 20))

    print("downloading", package, version, item["size"], flush=True)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        list(
            pool.map(
                fetch,
                (
                    start
                    for start in range(0, item["size"], chunk)
                    if start not in state["done"]
                ),
            )
        )
    if not valid(part):
        raise RuntimeError("PyPI SHA256 verification failed")
    part.replace(target)
    state_path.unlink()
    target.with_suffix(".provenance.json").write_text(json.dumps(item, indent=2) + "\n")
    print("downloaded and verified", target, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("package")
    parser.add_argument("version")
    parser.add_argument("--destination", type=Path, default=Path("data/wheels-pypi"))
    parser.add_argument("--workers", type=int, default=16)
    args = parser.parse_args()
    download(args.package, args.version, args.destination, args.workers)
