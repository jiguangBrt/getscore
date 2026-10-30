"""Stable experiment names, reversible migration, and unique batch statistics."""

import json
from pathlib import Path

import pytest

from run_viewer.naming import canonical_name, case_directories, experiment_name, experiment_parts
from scripts.migrate_run_names import LEGACY_RUNS, migrate


def write_json(path, value):
    path.write_text(json.dumps(value), encoding="utf-8")


@pytest.mark.parametrize("name", ["long-original-9-gpt6-gawm", "long-original-gpt6-gawm-9"])
def test_current_and_previous_order_identify_the_same_experiment(name):
    parts = experiment_parts(name)
    assert parts == dict(simulation="long-original", experiment="9",
                         language_model="gpt6", action_model="gawm")
    assert experiment_name(**parts) == "long-original-9-gpt6-gawm"
    assert canonical_name(name) == "long-original-9-gpt6-gawm"


@pytest.mark.parametrize("name", [*LEGACY_RUNS, "run", "long-original-swap-benchmark"])
def test_unstructured_and_timestamp_names_are_not_filter_combinations(name):
    assert experiment_parts(name) is None


@pytest.mark.parametrize("parts", [
    ("../long", 9, "gpt6", "gawm"), ("long", -1, "gpt6", "gawm"),
    ("long", 0, "", "pi05"), ("long", 0, "gpt6", "a/b"),
])
def test_name_components_cannot_escape_the_output_directory(parts):
    with pytest.raises(ValueError):
        experiment_name(*parts)


def test_migration_preserves_recordings_and_aliases_without_double_counting(tmp_path):
    root = tmp_path / "outputs"
    root.mkdir()
    old = root / "long-original-gpt6-gawm-9"
    old.mkdir()
    config = {"run_status": "finished", "command": ["--output-dir", str(old)]}
    write_json(old / "configuration.json", config)
    (old / "episode.mp4").write_bytes(b"original recording")
    result = dict(name=old.name, directory=str(old), video=str(old / "episode.mp4"),
                  variant="original", planner="gpt6", policy="gawm", success=True, error=None)
    write_json(old / "benchmark_result.json", result)
    for name in LEGACY_RUNS:
        (root / name).mkdir()
        write_json(root / name / "configuration.json", {"run_status": "finished"})
    batch = root / "batch"
    (batch / "cases").mkdir(parents=True)
    link = batch / "cases" / old.name
    link.symlink_to(Path("../..") / old.name, target_is_directory=True)
    write_json(batch / "output-layout-audit.json", {"cases": [{"directory": str(old)}]})

    plan = migrate(root)
    assert (plan["renamed"], plan["archived"]) == (1, 2)
    assert old.is_dir()
    applied = migrate(root, apply=True)
    new = root / "long-original-9-gpt6-gawm"
    assert not old.exists() and new.is_dir()
    assert (new / "episode.mp4").read_bytes() == b"original recording"
    assert json.loads((new / "configuration.json").read_text()) == config
    assert json.loads((new / "benchmark_result.json").read_text())["directory"] == str(new)
    assert json.loads((new / "benchmark_result.json").read_text())["name"] == new.name
    assert link.resolve() == new
    assert (batch / "cases" / new.name).resolve() == new
    assert [p.name for p in case_directories(batch)] == [new.name]
    for name in LEGACY_RUNS:
        assert not (root / name).exists()
        assert (root / "_archive" / "runs" / name).is_dir()
    snapshot = Path(applied["metadata_backup"]) / "metadata" / new.name / "benchmark_result.json"
    assert json.loads(snapshot.read_text()) == result
    assert migrate(root, apply=True)["moves"] == []

    from scripts.run_long_benchmark import refresh
    assert refresh(batch)["completed"] == 1


def test_migration_detects_collisions_before_moving_any_directory(tmp_path):
    for name in ["long-original-gpt6-gawm-9", "long-original-9-gpt6-gawm", LEGACY_RUNS[0]]:
        (tmp_path / name).mkdir()
        write_json(tmp_path / name / "configuration.json", {"run_status": "finished"})
    with pytest.raises(FileExistsError):
        migrate(tmp_path, apply=True)
    assert (tmp_path / LEGACY_RUNS[0]).is_dir()


def test_migration_refuses_to_rename_a_running_experiment(tmp_path):
    run = tmp_path / "long-original-gpt6-gawm-9"
    run.mkdir()
    write_json(run / "configuration.json", {"run_status": "running"})
    with pytest.raises(RuntimeError):
        migrate(tmp_path, apply=True)
    assert run.is_dir()


def test_summary_reads_flat_catalog_without_cases_or_deferred_results(tmp_path):
    from scripts.run_long_benchmark import refresh

    run = tmp_path / 'long-swap-6-kimi-pi05'
    run.mkdir()
    write_json(run / 'configuration.json', {'run_status': 'finished'})
    write_json(run / 'benchmark_result.json', dict(name=run.name, variant='swap',
        planner='kimi', policy='pi05', success=True, error=None))
    # Deferred experiments need not have a directory or result yet.
    summary = refresh(tmp_path)
    assert summary['completed'] == 1
    assert summary['groups']['swap-kimi-pi05']['successes'] == 1
    assert [r['name'] for r in summary['results']] == [run.name]
