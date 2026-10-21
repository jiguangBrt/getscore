"""Expose the existing OpenPI WebSocket policy as an RPent VLA RPC service."""

from __future__ import annotations

import argparse
import tomllib
from pathlib import Path

import numpy as np
from openpi_client import image_tools
from rpent.utils.rpc import RpcFacade

from scripts.openpi_transport import OpenPiTransport


def libero_observation(obs):
    """Translate RPent's batched observation, preserving image orientation.

    RLinf's LIBERO adapter already rotates the two camera images by 180 degrees
    and represents state as XYZ, axis-angle, and two gripper joint positions.
    Only resize here; a second rotation would invert the model's observations.
    """
    state = np.asarray(obs["states"], dtype=np.float32)
    if state.shape != (1, 8) or not np.isfinite(state).all():
        raise ValueError(f"Expected finite LIBERO state [1,8], got {state.shape}")
    result = {"observation/state": state[0], "prompt": str(obs["task_descriptions"][0])}
    for source, target in (
        ("main_images", "observation/image"),
        ("wrist_images", "observation/wrist_image"),
    ):
        image = np.asarray(obs[source])
        if (
            image.ndim != 4
            or image.shape[0] != 1
            or image.shape[-1] != 3
            or image.dtype != np.uint8
        ):
            raise ValueError(
                f"Expected {source} uint8 [1,H,W,3], got {image.shape}/{image.dtype}"
            )
        result[target] = image_tools.convert_to_uint8(
            image_tools.resize_with_pad(image[0], 224, 224)
        )
    return result


class OpenPiBridge(RpcFacade):
    def __init__(self, transport):
        self.transport = transport
        self.last_prediction = None
        super().__init__()
        self._rpc.update(
            {
                "vla.predict": self.predict,
                "vla.status": self.status,
                "vla.infer_raw": self.transport.infer,
                "vla.last_prediction": lambda: self.last_prediction,
            }
        )

    def status(self):
        return {
            "num_action_chunks": 5,
            "action_dim": 7,
            "mode": "sync",
            "source": "official-openpi-pi05-libero",
            "metadata": self.transport.metadata(),
        }

    def predict(self, obs, options=None):
        if options and options.get("mode") == "rtc":
            raise ValueError(
                "LIBERO RTC execution is not implemented; use sync or forward an existing RTC envelope with vla.infer_raw"
            )
        result = self.transport.infer(libero_observation(obs))
        actions = np.asarray(result["actions"], dtype=np.float32)
        if (
            actions.ndim != 2
            or actions.shape[1] != 7
            or len(actions) < 5
            or not np.isfinite(actions).all()
        ):
            raise ValueError(f"Invalid LIBERO action chunk: {actions.shape}")
        self.last_prediction = {
            "action_shape": list(actions.shape),
            "client_timing": result.get("client_timing"),
            "server_timing": result.get("server_timing"),
        }
        # Execute the first five actions before synchronously observing/replanning,
        # matching OpenPI's reference replan_steps and RPent's LIBERO runtime.
        return actions[None, :5]

    def close(self):
        self.transport.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path, default=Path("configs/liberopro-local.toml")
    )
    parser.add_argument("--timing-log", type=Path, required=True)
    args = parser.parse_args()
    config = tomllib.loads(args.config.read_text())
    transport = OpenPiTransport(**config["openpi"], timing_path=args.timing_log)
    bridge = OpenPiBridge(transport)
    print(bridge.status(), flush=True)
    bridge.serve(
        transport="http", host=config["bridge"]["host"], port=config["bridge"]["port"]
    )


if __name__ == "__main__":
    main()
