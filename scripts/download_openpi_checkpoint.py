"""Download the public OpenPI LIBERO checkpoint and verify GCS checksums."""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
import threading
import time
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import google_crc32c


def download(
    destination: Path, workers: int = 64, mirror_prefix: str | None = None,
    chunk_bytes: int = 512 * 1024,
) -> None:
    if workers < 1 or chunk_bytes < 1:
        raise ValueError('workers and chunk_bytes must be positive')
    prefix = "checkpoints/pi05_libero/"
    api = "https://storage.googleapis.com/storage/v1/b/openpi-assets/o?"
    objects = []
    token = None
    while True:
        query = {"prefix": prefix}
        if token:
            query["pageToken"] = token
        for attempt in range(8):
            try:
                with urllib.request.urlopen(
                    api + urllib.parse.urlencode(query), timeout=45
                ) as response:
                    page = json.load(response)
                break
            except Exception as exc:
                print("metadata retry", attempt + 1, str(exc), flush=True)
                if attempt == 7:
                    raise
                time.sleep(2)
        objects.extend(
            item for item in page.get("items", []) if not item["name"].endswith("/")
        )
        token = page.get("nextPageToken")
        if not token:
            break
    if not objects:
        raise RuntimeError("GCS returned no checkpoint objects")
    destination.mkdir(parents=True, exist_ok=True)
    (destination / "source-objects.json").write_text(
        json.dumps(objects, indent=2) + "\n"
    )

    def valid(path, item):
        if not path.is_file() or path.stat().st_size != int(item["size"]):
            return False
        # GCS composite objects publish CRC32C but do not have an MD5 hash.
        crc = google_crc32c.Checksum()
        md5 = hashlib.md5()
        with path.open("rb") as source:
            while chunk := source.read(8 * 1024 * 1024):
                crc.update(chunk)
                md5.update(chunk)
        return base64.b64encode(crc.digest()).decode() == item["crc32c"] and (
            "md5Hash" not in item
            or base64.b64encode(md5.digest()).decode() == item["md5Hash"]
        )

    def fetch(item):
        relative = Path(item["name"][len(prefix) :])
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("Invalid GCS object path")
        target = destination / relative
        if valid(target, item):
            print("verified", relative, flush=True)
            return
        target.parent.mkdir(parents=True, exist_ok=True)
        partial = target.with_name(target.name + ".part")
        url = "https://storage.googleapis.com/openpi-assets/" + urllib.parse.quote(
            item["name"], safe="/"
        )
        url += "?generation=" + item["generation"]
        for attempt in range(6):
            try:
                if valid(partial, item):
                    partial.replace(target)
                    print("downloaded", relative, item["size"], flush=True)
                    return
                offset = partial.stat().st_size if partial.exists() else 0
                if offset >= int(item["size"]):
                    partial.unlink()
                    offset = 0
                request = urllib.request.Request(
                    url, headers={"Range": f"bytes={offset}-"} if offset else {}
                )
                with urllib.request.urlopen(request, timeout=180) as response:
                    mode = "ab" if offset and response.status == 206 else "wb"
                    with partial.open(mode) as output:
                        while chunk := response.read(4 * 1024 * 1024):
                            output.write(chunk)
                if not valid(partial, item):
                    raise RuntimeError(f"Size/checksum mismatch for {relative}")
                partial.replace(target)
                print("downloaded", relative, item["size"], flush=True)
                return
            except Exception as exc:
                print("retry", attempt + 1, relative, str(exc), flush=True)
                if attempt == 5:
                    raise
                time.sleep(min(2**attempt, 20))
        raise RuntimeError(f"Could not download {relative}")

    print(
        "objects",
        len(objects),
        "bytes",
        sum(int(item["size"]) for item in objects),
        flush=True,
    )
    jobs = []
    large = []
    for item in objects:
        target = destination / item["name"][len(prefix) :]
        if int(item["size"]) < chunk_bytes or valid(target, item):
            fetch(item)
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        part = target.with_name(target.name + ".range.part")
        progress_path = target.with_name(target.name + ".ranges.json")
        if progress_path.exists():
            progress = json.loads(progress_path.read_text())
            if progress["generation"] != item["generation"]:
                raise RuntimeError("Existing chunk download has a different generation")
            previous_chunk = progress["chunk_bytes"]
            if previous_chunk != chunk_bytes:
                if previous_chunk % chunk_bytes == 0:
                    progress["done"] = [
                        start + offset
                        for start in progress["done"]
                        for offset in range(
                            0, min(previous_chunk, int(item["size"]) - start), chunk_bytes
                        )
                    ]
                elif chunk_bytes % previous_chunk == 0:
                    completed = set(progress["done"])
                    progress["done"] = [
                        start for start in range(0, int(item["size"]), chunk_bytes)
                        if all(offset in completed for offset in range(
                            start, min(start + chunk_bytes, int(item["size"])), previous_chunk
                        ))
                    ]
                else:
                    raise RuntimeError("Cannot resume incompatible chunk sizes")
                progress["chunk_bytes"] = chunk_bytes
        else:
            progress = {
                "generation": item["generation"],
                "chunk_bytes": chunk_bytes,
                "done": [],
            }
            with part.open("wb") as output:
                output.truncate(int(item["size"]))
            legacy = target.with_name(target.name + ".part")
            if legacy.exists():
                reusable = min(legacy.stat().st_size, int(item["size"])) // chunk_bytes
                with legacy.open("rb") as source, part.open("r+b") as output:
                    for index in range(reusable):
                        output.write(source.read(chunk_bytes))
                        progress["done"].append(index * chunk_bytes)
        entry = (item, target, part, progress_path, progress, threading.Lock())
        large.append(entry)
        for start in range(0, int(item["size"]), chunk_bytes):
            if start not in progress["done"]:
                jobs.append(
                    (entry, start, min(start + chunk_bytes, int(item["size"])) - 1)
                )

    resolved_urls = {}

    def fetch_range(job):
        (item, target, part, progress_path, progress, lock), start, end = job
        original_url = (
            mirror_prefix.rstrip("/")
            + "/"
            + urllib.parse.quote(item["name"][len(prefix) :], safe="/")
            if mirror_prefix
            else "https://storage.googleapis.com/openpi-assets/"
            + urllib.parse.quote(item["name"], safe="/")
        )
        for attempt in range(8):
            try:
                request = urllib.request.Request(
                    resolved_urls.get(item["name"], original_url),
                    headers={"Range": f"bytes={start}-{end}"},
                )
                with urllib.request.urlopen(request, timeout=45) as response:
                    expected_range = f"bytes {start}-{end}/{item['size']}"
                    if (
                        response.status != 206
                        or response.headers.get("Content-Range") != expected_range
                    ):
                        raise RuntimeError("GCS did not honor the requested byte range")
                    resolved_urls[item["name"]] = response.geturl()
                    data = response.read(end - start + 2)
                if len(data) != end - start + 1:
                    raise RuntimeError("Truncated GCS range")
                fd = os.open(part, os.O_WRONLY)
                try:
                    offset = 0
                    while offset < len(data):
                        offset += os.pwrite(fd, data[offset:], start + offset)
                finally:
                    os.close(fd)
                with lock:
                    progress["done"].append(start)
                    temporary = progress_path.with_suffix(".tmp")
                    temporary.write_text(json.dumps(progress))
                    temporary.replace(progress_path)
                    if len(progress["done"]) % 128 == 0:
                        print(
                            "range progress",
                            target.name,
                            len(progress["done"]),
                            flush=True,
                        )
                return
            except Exception as exc:
                # CDN signed URLs can expire during a long download. Resolve a
                # fresh one from the pinned mirror revision on the next attempt.
                resolved_urls.pop(item["name"], None)
                print(
                    "range retry", target.name, start, attempt + 1, str(exc), flush=True
                )
                if attempt == 7:
                    raise
                time.sleep(min(2**attempt, 20))

    # Spread active requests across objects instead of exhausting one object
    # before requesting the next checkpoint shard.
    jobs.sort(key=lambda job: job[1])
    with ThreadPoolExecutor(max_workers=workers) as pool:
        list(pool.map(fetch_range, jobs))
    for item, target, part, progress_path, _, _ in large:
        if not valid(part, item):
            raise RuntimeError(
                f"Whole-object GCS checksum verification failed: {target}"
            )
        part.replace(target)
        progress_path.unlink()
        print("downloaded and verified", target, flush=True)
    manifest = {
        "source": "gs://openpi-assets/checkpoints/pi05_libero",
        "transfer_mirror": mirror_prefix,
        "objects": objects,
    }
    (destination / "download-manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n"
    )
    print("All checkpoint objects verified", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("destination", type=Path)
    parser.add_argument("--workers", type=int, default=64)
    parser.add_argument("--mirror-prefix", default=None)
    parser.add_argument("--chunk-bytes", type=int, default=512 * 1024)
    args = parser.parse_args()
    download(args.destination, args.workers, args.mirror_prefix, args.chunk_bytes)
