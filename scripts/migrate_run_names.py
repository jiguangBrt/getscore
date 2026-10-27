"""Rename saved experiments without modifying recorded media or event logs."""

from __future__ import annotations

import argparse
from datetime import datetime
import json
import os
from pathlib import Path
import shutil
import sys
from zoneinfo import ZoneInfo

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from run_viewer.naming import canonical_name, experiment_parts

LEGACY_RUNS = (
    "liberopro-gawm-gpt6astra-20261021-162733",
    "liberopro-pi05-gpt6astra-20261020-104215",
)


def rewrite_metadata(value, moves):
    if isinstance(value, dict):
        return {key: rewrite_metadata(part, moves) for key, part in value.items()}
    if isinstance(value, list):
        return [rewrite_metadata(part, moves) for part in value]
    if isinstance(value, str):
        for old, new in moves:
            if value == str(old) or value.startswith(str(old) + "/"):
                return str(new) + value[len(str(old)):]
            if value == old.name:
                return new.name
    return value


def migrate(root, *, apply=False):
    root = Path(root).resolve()
    moves = []
    for source in sorted(root.iterdir()):
        if source.is_symlink() or not (source / "configuration.json").is_file():
            continue
        if source.name in LEGACY_RUNS:
            destination = root / "_archive" / "runs" / source.name
        elif experiment_parts(source.name):
            destination = root / canonical_name(source.name)
        else:
            continue
        if destination != source:
            if destination.exists() or destination.is_symlink():
                raise FileExistsError(f"目标目录已存在，未执行迁移：{destination}")
            config = json.loads((source / "configuration.json").read_text())
            if config.get("run_status") in {"starting", "running"}:
                raise RuntimeError(f"实验仍在运行，未执行迁移：{source}")
            moves.append((source, destination))
    replacements = dict(moves)
    links = []
    batches = [path for path in root.iterdir() if (path / "cases").is_dir()]
    for batch in batches:
        for link in (batch / "cases").iterdir():
            if link.is_symlink() and link.resolve() in replacements:
                target = replacements[link.resolve()]
                alias = link.with_name(target.name)
                if alias != link and (alias.exists() or alias.is_symlink()):
                    raise FileExistsError(f"批次目标链接已存在，未执行迁移：{alias}")
                links.append((link, alias, target))
    manifest = dict(moves=[dict(old=str(old), new=str(new)) for old, new in moves],
                    renamed=sum(new.parent == root for _, new in moves),
                    archived=sum(new.parent != root for _, new in moves),
                    compatibility_links=len(links), applied=apply)
    if not apply or not moves:
        return manifest

    stamp = datetime.now(ZoneInfo("Asia/Shanghai")).strftime("%Y%m%d-%H%M%S-%f")
    backup = root / "_archive" / "records" / f"run-name-migration-{stamp}"
    backup.mkdir(parents=True)
    manifest["metadata_backup"] = str(backup)
    (backup / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    for old, new in moves:
        new.parent.mkdir(parents=True, exist_ok=True)
        old.rename(new)
    for link, alias, target in links:
        temporary = link.with_name("." + link.name + ".renaming")
        temporary.symlink_to(os.path.relpath(target, link.parent), target_is_directory=True)
        temporary.replace(link)
        if alias != link:
            alias.symlink_to(os.path.relpath(target, alias.parent), target_is_directory=True)
    metadata = []
    for path in root.iterdir():
        if (path / "configuration.json").is_file():
            metadata.extend(path / name for name in ("benchmark_result.json", "artifact-completeness.json"))
    metadata.extend(batch / "output-layout-audit.json" for batch in batches)
    for path in metadata:
        if not path.is_file():
            continue
        original = json.loads(path.read_text())
        updated = rewrite_metadata(original, moves)
        if original == updated:
            continue
        relative = path.relative_to(root)
        snapshot = backup / "metadata" / relative
        snapshot.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, snapshot)
        temporary = path.with_name(path.name + ".renaming")
        temporary.write_text(json.dumps(updated, ensure_ascii=False, indent=2) + "\n")
        temporary.replace(path)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--apply", action="store_true", help="执行迁移；默认只显示计划")
    args = parser.parse_args()
    print(json.dumps(migrate(args.root, apply=args.apply), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
