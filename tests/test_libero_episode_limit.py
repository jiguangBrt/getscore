"""Real-simulator regression for reset settling versus the episode limit."""

import os

import numpy as np
import pytest


@pytest.mark.skipif(
    os.environ.get("GETSCORE_RUN_LIBERO_ENV") != "1",
    reason="Requires the local LIBERO-PRO assets and EGL renderer",
)
def test_short_episode_truncates_after_reset_settling_without_killing_worker(
    monkeypatch,
):
    monkeypatch.syspath_prepend(os.environ["RPENT_RLINF_ROOT"])
    from robots.libero.env_server import LiberoEnvFacade, make_env

    env = make_env(2, 0, "libero_object_swap", max_episode_steps=2)
    facade = LiberoEnvFacade(env, meta={})
    try:
        observation, _ = facade.reset()
        assert np.isfinite(observation["states"]).all()
        action = np.array([0, 0, 0, 0, 0, 0, -1], dtype=np.float32)
        _, _, terminated, truncated, _ = facade.step(action)
        assert not terminated and not truncated
        _, _, terminated, truncated, _ = facade.step(action)
        assert not terminated and truncated
        # The worker must remain usable for final artifact capture.
        assert "robot0_eef_pos" in facade.raw_obs()
        assert facade.get_camera_meta("agentview", height=256, width=256)
    finally:
        facade.close()
