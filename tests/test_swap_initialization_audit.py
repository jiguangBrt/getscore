"""Retain nominal reruns without hiding unexpected initial-condition changes."""
import json
from pathlib import Path

import numpy as np
import pytest

from scripts.run_swap_kimi_2000 import verify_initialization


@pytest.mark.parametrize('image_match,baseline_varies,robot_changes,accepted', [
    (True, False, False, True),
    (False, True, False, True),
    (False, False, False, False),
    (False, True, True, False),
])
def test_initial_condition_audit(tmp_path, image_match, baseline_varies, robot_changes, accepted):
    source = tmp_path / 'original/long-swap-3-kimi-pi05'
    output = tmp_path / 'rerun/long-swap-3-kimi-pi05'
    config = dict(variant='swap', suite='libero_10_swap', task=3, seed=0, initial_state_index=0,
                  bddl_files=dict(sha256='same-bddl'), init_files=dict(sha256='same-init'))
    rows = []
    for directory, state, image_hash in [
        (source, np.zeros((1, 8)), 'old-image'),
        (output, np.full((1, 8), int(robot_changes)), 'old-image' if image_match else 'new-image'),
    ]:
        trajectory = directory / 'trajectory'
        trajectory.mkdir(parents=True)
        np.savez(trajectory / 'transitions.npz', states=state)
        (directory / 'configuration.json').write_text(json.dumps(config))
        row = dict(trajectory=str(trajectory), initial_observation_sha256=image_hash)
        (directory / 'benchmark_result.json').write_text(json.dumps(row))
        rows.append(row)
    if baseline_varies:
        other = source.parent / 'long-swap-3-kimi-gawm'
        other.mkdir()
        (other / 'benchmark_result.json').write_text(json.dumps(dict(initial_observation_sha256='other-old-image')))
    if accepted:
        audit = verify_initialization(source, output, *rows)
        assert audit['initial_robot_state_exact']
        assert audit['initial_observation_exact'] == image_match
    else:
        with pytest.raises(RuntimeError):
            verify_initialization(source, output, *rows)
