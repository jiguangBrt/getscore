"""Read-only export of selected installed Python 3.10 GAWM dependencies."""

import json
import sys
import tarfile
from importlib import metadata
from pathlib import Path

from packaging.requirements import Requirement

selected = set()
skip = {"torch", "torchvision", "av", "triton", "scikit-learn"}
queue = [
    "transformers",
    "accelerate",
    "albumentations",
    "numpydantic",
    "omegaconf",
    "pydantic",
    "einops",
    "timm",
    "rich",
    "websockets",
    "msgpack",
    "pandas",
]
files = {}
versions = {}
while queue:
    name = queue.pop().lower().replace("_", "-")
    if name in selected or name in skip or name.startswith("nvidia-"):
        continue
    selected.add(name)
    dist = metadata.distribution(name)
    versions[name] = dist.version
    for requirement in dist.requires or []:
        req = Requirement(requirement)
        if req.marker is None or req.marker.evaluate({"extra": ""}):
            queue.append(req.name)
    for entry in dist.files or []:
        if (
            ".." in entry.parts
            or "__pycache__" in entry.parts
            or entry.suffix in {".pyc", ".pyo"}
        ):
            continue
        if (
            name == "pytorch3d"
            and entry.parts[0] == "pytorch3d"
            and len(entry.parts) > 1
            and entry.parts[1] not in {"__init__.py", "transforms", "common"}
        ):
            continue
        path = Path(dist.locate_file(entry))
        if path.is_file():
            files[str(entry)] = path
print(json.dumps({"versions": versions, "file_count": len(files)}), file=sys.stderr)
with tarfile.open(fileobj=sys.stdout.buffer, mode="w|gz") as archive:
    for entry, path in sorted(files.items()):
        archive.add(path, arcname=entry, recursive=False)
