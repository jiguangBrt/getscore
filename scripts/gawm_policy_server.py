"""Serve the copied StarVLA GAWM checkpoint without changing its source tree."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--host", default="192.168.50.73")
    parser.add_argument("--port", type=int, default=8115)
    args = parser.parse_args()
    root = args.root.resolve()
    sys.path.insert(0, str(root / "repo"))

    # This is an RGB/state policy. The reused image contains optional audio
    # packages without their native libraries; do not import them for DINOv3.
    import transformers.utils.import_utils as transformer_imports

    transformer_imports._soundfile_available = False
    transformer_imports._librosa_available = False

    from deployment.model_server.policy_norm_processor import PolicyNormProcessor
    from deployment.model_server.tools.websocket_policy_server import (
        WebsocketPolicyServer,
    )
    from starVLA.model.framework.share_tools import dict_to_namespace, read_mode_config
    from starVLA.model.framework.WM4A.GAWM import GAWM

    checkpoint = root / "runtime/checkpoints/gawm_libero_80000_20261018_002850.pt"
    config, stats = read_mode_config(checkpoint)
    torch.manual_seed(0)
    framework = GAWM(dict_to_namespace(config))
    weights = torch.load(checkpoint, map_location="cpu", weights_only=True)
    weights = framework.remap_checkpoint_state_dict(weights)
    # The upstream generic loader filters incompatible keys. This deployment
    # requires every tensor to match, so no component runs with random weights.
    loaded = framework.load_state_dict(weights, strict=True)
    framework.norm_stats = stats
    framework = framework.to("cuda").eval()
    processor = PolicyNormProcessor(str(checkpoint), unnorm_key="franka")
    vocabulary = list(config["framework"]["lang_cond"]["task_names"])
    metadata = {
        "source": "starvla-gawm-libero",
        "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        "strict_load": {
            "missing_keys": loaded.missing_keys,
            "unexpected_keys": loaded.unexpected_keys,
        },
        "action_chunk_size": 8,
        "action_dim": 7,
        "image_size": list(framework.image_size),
        "image_resize_resample": framework.image_resize_resample,
        "task_vocabulary": vocabulary,
        "state_normalization": "training PolicyNormProcessor",
        "action_unnormalization": "training PolicyNormProcessor",
        "gripper_encoding": "open_probability; client converts >0.5 to -1 else +1",
        "torch": torch.__version__,
        "gpu": torch.cuda.get_device_name(),
    }
    (root / "logs/model-metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n"
    )

    class Policy:
        def predict_action(self, examples, unnorm_key="franka", **kwargs):
            if unnorm_key != "franka" or len(examples) != 1:
                raise ValueError("This service accepts one franka example")
            started = time.monotonic()
            example = dict(examples[0], robot_tag="franka")
            task_key = "franka:" + " ".join(example["lang"].strip().lower().split())
            if task_key not in vocabulary:
                raise ValueError(
                    f"GAWM requires a saved training task; unknown instruction: {example['lang']!r}"
                )
            example["state"] = processor.apply_state(example["state"])
            with torch.inference_mode():
                result = framework.predict_action(examples=[example], **kwargs)
            actions = processor.unapply_actions(
                np.asarray(result["normalized_actions"])[0]
            )
            actions = np.asarray(actions, dtype=np.float32)[None]
            if actions.shape != (1, 8, 7) or not np.isfinite(actions).all():
                raise ValueError(f"Invalid GAWM action shape/values: {actions.shape}")
            torch.cuda.synchronize()
            timing = {"infer_ms": (time.monotonic() - started) * 1000}
            with (root / "logs/inference.jsonl").open("a") as stream:
                stream.write(
                    json.dumps(
                        {
                            "time_unix": time.time(),
                            "task": example["lang"],
                            "action_shape": list(actions.shape),
                            **timing,
                        }
                    )
                    + "\n"
                )
            return {"actions": actions, "server_timing": timing}

    class TimedServer(WebsocketPolicyServer):
        def _route_message(self, message):
            result = super()._route_message(message)
            # Expose timing at the envelope level used by the existing client,
            # so its RTT estimate can subtract policy computation time.
            timing = result.get("data", {}).get("server_timing")
            if timing is not None:
                result["server_timing"] = timing
            return result

    print(
        json.dumps({k: v for k, v in metadata.items() if k != "task_vocabulary"}),
        flush=True,
    )
    TimedServer(
        policy=Policy(),
        host=args.host,
        port=args.port,
        metadata=metadata,
        idle_timeout=-1,
    ).serve_forever()


if __name__ == "__main__":
    main()
