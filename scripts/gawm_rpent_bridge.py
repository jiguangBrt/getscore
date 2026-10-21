"""Translate RPent LIBERO observations and StarVLA GAWM action chunks."""

from __future__ import annotations

import argparse
import tomllib
from pathlib import Path

import numpy as np
from rpent.utils.rpc import RpcFacade

from scripts.openpi_transport import OpenPiTransport


def libero_example(obs):
    """RLinf already rotates RGB and packs XYZ/axis-angle/gripper state."""
    state = np.asarray(obs["states"], dtype=np.float32)
    if state.shape != (1, 8) or not np.isfinite(state).all():
        raise ValueError(f"Expected finite LIBERO state [1,8], got {state.shape}")
    images = []
    for key in ("main_images", "wrist_images"):
        array = np.asarray(obs[key])
        if (
            array.ndim != 4
            or array.shape[0] != 1
            or array.shape[-1] != 3
            or array.dtype != np.uint8
        ):
            raise ValueError(
                f"Expected {key} uint8 [1,H,W,3], got {array.shape}/{array.dtype}"
            )
        # Preserve native RGB resolution; GAWM's rgb_pixels performs the
        # checkpoint's OpenCV-linear resize and ImageNet normalization.
        images.append(np.ascontiguousarray(array[0]))
    tasks = obs["task_descriptions"]
    if len(tasks) != 1 or not isinstance(tasks[0], str):
        raise ValueError("Expected one task instruction")
    return {"image": images, "state": state[0], "lang": tasks[0], "robot_tag": "franka"}


def libero_actions(raw, execute_horizon):
    actions = np.array(raw, dtype=np.float32, copy=True)
    if actions.shape != (1, 8, 7) or not np.isfinite(actions).all():
        raise ValueError(f"Expected finite GAWM actions [1,8,7], got {actions.shape}")
    if not 1 <= execute_horizon <= 8:
        raise ValueError("execute_horizon must be in [1,8]")
    # Match StarVLA's eval_libero._binarize_gripper_open exactly.
    actions[..., 6] = 1.0 - 2.0 * (actions[..., 6] > 0.5)
    return actions[:, :execute_horizon]


class GawmBridge(RpcFacade):
    def __init__(self, transport, execute_horizon=8):
        self.transport = transport
        self.execute_horizon = execute_horizon
        self.last_prediction = None
        self.metadata = transport.metadata()
        if self.metadata.get("source") != "starvla-gawm-libero":
            raise ValueError("Expected the GAWM LIBERO service")
        super().__init__()
        self._rpc.update(
            {
                "vla.predict": self.predict,
                "vla.status": self.status,
                "vla.last_prediction": lambda: self.last_prediction,
            }
        )

    def status(self):
        return {
            "num_action_chunks": self.execute_horizon,
            "action_dim": 7,
            "mode": "sync",
            "source": "starvla-gawm-libero",
            "metadata": self.metadata,
        }

    def predict(self, obs, options=None):
        if options and options.get("mode") == "rtc":
            raise ValueError("GAWM LIBERO currently uses synchronous inference")
        example = libero_example(obs)
        response = self.transport.infer(
            {"examples": [example], "unnorm_key": "franka", "do_sample": False}
        )
        if not response.get("ok"):
            raise RuntimeError(f"GAWM inference failed: {response.get('error')}")
        data = response["data"]
        actions = libero_actions(data["actions"], self.execute_horizon)
        self.last_prediction = {
            "action_shape": list(np.asarray(data["actions"]).shape),
            "executed_shape": list(actions.shape),
            "client_timing": response.get("client_timing"),
            "server_timing": data.get("server_timing"),
            "instruction": example["lang"],
        }
        return actions

    def close(self):
        self.transport.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--timing-log", type=Path, required=True)
    args = parser.parse_args()
    config = tomllib.loads(args.config.read_text())
    policy = dict(config["gawm"])
    horizon = policy.pop("execute_horizon")
    transport = OpenPiTransport(**policy, timing_path=args.timing_log)
    bridge = GawmBridge(transport, horizon)
    print(bridge.status(), flush=True)
    bridge.serve(
        transport="http", host=config["bridge"]["host"], port=config["bridge"]["port"]
    )


if __name__ == "__main__":
    main()
