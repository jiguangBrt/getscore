"""Repack existing uv wheel archives for installation over the local network."""

import sys
import zipfile
from concurrent.futures import ThreadPoolExecutor
from importlib.metadata import distribution
from pathlib import Path

cache = (
    Path(sys.argv[2])
    if len(sys.argv) > 2
    else Path("/mnt/reacher-fast/openpi_ur_pp_202607/uv-cache/archive-v0")
)
destination = Path(sys.argv[1])
destination.mkdir(parents=True, exist_ok=True)
names = ["torch", "torchvision", "triton"]
names.extend(
    requirement.split("==")[0]
    for requirement in distribution("torch").requires
    if requirement.startswith("nvidia-") and "==" in requirement
)


def pack(name):
    version = distribution(name).version
    normalized = name.replace("-", "_")
    candidates = list(cache.glob(f"*/{normalized}-{version}.dist-info"))
    metadata = distribution(name).read_text("METADATA")
    candidates = [p for p in candidates if (p / "METADATA").read_text() == metadata]
    if not candidates:
        raise RuntimeError(f"No matching uv archive for {name} {version}")
    info = candidates[0]
    wheel = (info / "WHEEL").read_text()
    tag = next(
        line.split(": ", 1)[1]
        for line in wheel.splitlines()
        if line.startswith("Tag: ")
    )
    output = destination / f"{normalized}-{version}-{tag}.whl"
    if output.exists():
        return
    temporary = output.with_suffix(".partial")
    with zipfile.ZipFile(
        temporary, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=1
    ) as archive:
        for path in info.parent.rglob("*"):
            if path.is_file():
                archive.write(path, path.relative_to(info.parent))
    temporary.replace(output)
    print(output.name, output.stat().st_size, flush=True)


with ThreadPoolExecutor(max_workers=4) as pool:
    list(pool.map(pack, names))
