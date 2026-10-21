"""Complete the reused LIBERO assets, checking each new file against Git's blob hash."""

import hashlib
import json
import shutil
import time
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

root = Path(__file__).resolve().parents[1]
evidence = root / "outputs/_archive/records/liberopro-setup-2026-10-19"
base = root / "data/libero-pro/assets"
entries = json.loads((evidence / "pro-assets-missing.json").read_text())
revision = (evidence / "pro-assets-hf-revision.txt").read_text().strip()
groups = {}
for entry in entries:
    groups.setdefault(entry["sha"], []).append(entry)


def fetch(group):
    sha, files = group
    entry = files[0]
    target = base / entry["path"]
    url = (
        "https://huggingface.co/datasets/RLinf/LIBERO-PRO-assets/resolve/"
        + revision
        + "/"
        + urllib.parse.quote(entry["path"], safe="/")
    )
    for attempt in range(8):
        try:
            with urllib.request.urlopen(url, timeout=45) as response:
                data = response.read()
            if hashlib.sha1(f"blob {len(data)}\0".encode() + data).hexdigest() != sha:
                raise RuntimeError(
                    "Asset content differs from the pinned LIBERO-PRO Git blob"
                )
            target.parent.mkdir(parents=True, exist_ok=True)
            temporary = target.with_name(target.name + ".getscore-partial")
            temporary.write_bytes(data)
            temporary.replace(target)
            for alias in files[1:]:
                other = base / alias["path"]
                other.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(target, other)
            print("verified", entry["path"], len(data), flush=True)
            return
        except Exception as exc:
            print("retry", entry["path"], attempt + 1, str(exc), flush=True)
            if attempt == 7:
                raise
            time.sleep(min(2**attempt, 20))


with ThreadPoolExecutor(max_workers=8) as pool:
    list(pool.map(fetch, groups.items()))
print("All additional LIBERO-PRO assets verified", flush=True)
