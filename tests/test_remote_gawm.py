"""GAWM action/state/image contracts that differ from the OpenPI adapter."""

import numpy as np
import pytest

from scripts.gawm_rpent_bridge import GawmBridge, libero_actions, libero_example


@pytest.fixture
def observation():
    image = np.zeros((1, 256, 256, 3), dtype=np.uint8)
    image[0, 0, 0] = [1, 2, 3]
    image[0, -1, -1] = [10, 20, 30]
    return {
        "main_images": image,
        "wrist_images": image.copy(),
        "states": np.arange(8, dtype=np.float32)[None],
        "task_descriptions": ["pick up the salad dressing and place it in the basket"],
    }


def test_preserves_rgb_resolution_orientation_and_raw_proprio(observation):
    result = libero_example(observation)
    for image in result["image"]:
        np.testing.assert_array_equal(image, observation["main_images"][0])
    np.testing.assert_array_equal(result["state"], np.arange(8))
    assert result["lang"] == observation["task_descriptions"][0]


def test_gripper_threshold_matches_original_starvla_eval_without_mutating_input():
    raw = np.arange(56, dtype=np.float32).reshape(1, 8, 7) / 100
    raw[0, :, 6] = [0, 0.49, 0.5, 0.51, 1, 0.2, 0.7, 0.9]
    original = raw.copy()
    result = libero_actions(raw, 8)
    np.testing.assert_array_equal(result[..., :6], raw[..., :6])
    np.testing.assert_array_equal(result[0, :, 6], [1, 1, 1, -1, -1, 1, -1, -1])
    np.testing.assert_array_equal(raw, original)


@pytest.mark.parametrize(
    "raw", [np.zeros((8, 7)), np.zeros((1, 8, 6)), np.full((1, 8, 7), np.nan)]
)
def test_rejects_bad_action_chunk(raw):
    with pytest.raises(ValueError):
        libero_actions(raw, 8)


def test_server_error_is_not_executed_or_retried(observation):
    class Transport:
        calls = 0

        def metadata(self):
            return {"source": "starvla-gawm-libero"}

        def infer(self, request):
            self.calls += 1
            assert request["examples"][0]["lang"] == observation["task_descriptions"][0]
            return {"ok": False, "error": {"message": "unknown training task"}}

    transport = Transport()
    bridge = GawmBridge(transport)
    with pytest.raises(RuntimeError, match="unknown training task"):
        bridge.predict(observation)
    assert transport.calls == 1


def test_bridge_executes_full_eight_step_gawm_chunk(observation):
    class Transport:
        def metadata(self):
            return {"source": "starvla-gawm-libero"}

        def infer(self, request):
            return {"ok": True, "data": {"actions": np.zeros((1, 8, 7))}}

    bridge = GawmBridge(Transport())
    result = bridge.predict(observation)
    assert bridge.status()["num_action_chunks"] == 8
    assert result.shape == (1, 8, 7)
    np.testing.assert_array_equal(result[..., 6], 1)
