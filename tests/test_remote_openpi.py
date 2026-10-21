"""Protocol, timeout, and observation checks without models or paid APIs."""

import json
import threading
import time

import numpy as np
import pytest
from openpi_client import msgpack_numpy
from websockets.exceptions import ConnectionClosed
from websockets.sync.server import serve

from scripts.openpi_rpent_bridge import OpenPiBridge, libero_observation
from scripts.openpi_transport import OpenPiTransport


@pytest.fixture
def observation():
    # Distinct corners detect an accidental second rotation.
    image = np.zeros((1, 224, 224, 3), dtype=np.uint8)
    image[0, 0, 0] = [1, 2, 3]
    image[0, -1, -1] = [10, 20, 30]
    return {
        "main_images": image,
        "wrist_images": image.copy(),
        "states": np.arange(8, dtype=np.float32)[None],
        "task_descriptions": ["put the soup in the basket"],
    }


def test_libero_images_state_and_prompt_are_preserved(observation):
    result = libero_observation(observation)
    np.testing.assert_array_equal(
        result["observation/image"], observation["main_images"][0]
    )
    np.testing.assert_array_equal(result["observation/state"], np.arange(8))
    assert result["prompt"] == observation["task_descriptions"][0]


def test_bridge_uses_only_first_five_actions_and_preserves_gripper(observation):
    class Transport:
        def infer(self, request):
            return {"actions": np.arange(70).reshape(10, 7)}

    bridge = OpenPiBridge(Transport())
    np.testing.assert_array_equal(
        bridge.predict(observation), np.arange(70).reshape(1, 10, 7)[:, :5]
    )


@pytest.mark.parametrize(
    "payload", [{"prompt": "test"}, {"request_type": "rtc_v1", "request_id": 42}]
)
@pytest.mark.parametrize(
    "delay,timeout,should_timeout", [(0, 1.0, False), (0.25, 0.05, True)]
)
def test_shared_websocket_timeout_for_sync_and_rtc(
    tmp_path, payload, delay, timeout, should_timeout
):
    received = []

    def handler(connection):
        packer = msgpack_numpy.Packer()
        try:
            connection.send(packer.pack({"test": True}))
            received.append(msgpack_numpy.unpackb(connection.recv()))
            time.sleep(delay)
            connection.send(packer.pack({"ok": True, "actions": np.zeros((10, 7))}))
        except ConnectionClosed:
            pass

    with serve(handler, "127.0.0.1", 0, compression=None) as server:
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        port = server.socket.getsockname()[1]
        log = tmp_path / "timing.jsonl"
        transport = OpenPiTransport(
            "127.0.0.1",
            port,
            connect_timeout_s=2,
            request_timeout_s=timeout,
            timing_path=log,
        )
        try:
            if should_timeout:
                with pytest.raises(TimeoutError, match="timed out"):
                    transport.infer(payload)
            else:
                assert transport.infer(payload)["ok"] is True
        finally:
            transport.close()
            server.shutdown()
            thread.join(timeout=2)
        assert received == [payload]  # No retries/replays of action requests.
        timing = json.loads(log.read_text())
        assert timing["mode"] == ("rtc" if "request_type" in payload else "sync")
        assert timing["request_timeout_s"] == timeout
        assert timing["ok"] is not should_timeout


@pytest.mark.parametrize("bad", [0, -1, float("nan"), float("inf")])
def test_shared_transport_rejects_invalid_timeouts(bad):
    with pytest.raises(ValueError):
        OpenPiTransport("localhost", 8008, connect_timeout_s=2, request_timeout_s=bad)
