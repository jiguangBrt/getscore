"""Record fresh pure-GAWM trials using the historical standard-LIBERO protocol."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import time
from importlib import metadata
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
import torch

EXPECTED_CHECKPOINT = "f698e76ae129ccfbe8349418acd79044f2a1ca6366d098546afb477fd79e7e13"


def proprio(observation):
    """Match the quaternion-to-axis-angle conversion in the source evaluator."""
    quaternion = np.asarray(observation["robot0_eef_quat"], dtype=np.float64).copy()
    quaternion[3] = np.clip(quaternion[3], -1, 1)
    denominator = np.sqrt(1 - quaternion[3] ** 2)
    rotation = (
        np.zeros(3)
        if math.isclose(denominator, 0)
        else quaternion[:3] * 2 * math.acos(quaternion[3]) / denominator
    )
    return np.concatenate(
        [observation["robot0_eef_pos"], rotation, observation["robot0_gripper_qpos"]]
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--host", default="192.168.50.73")
    parser.add_argument("--port", type=int, default=8115)
    parser.add_argument("--episodes", type=int, default=5)
    args = parser.parse_args()
    if not 1 <= args.episodes <= 50:
        parser.error("episodes must be between 1 and 50")
    root = args.root.resolve()
    for directory in ("videos", "episodes", "runtime/libero-config"):
        (root / directory).mkdir(parents=True, exist_ok=True)

    from examples.LIBERO.eval_files.model2libero_interface import ModelClient
    from libero.libero import benchmark, get_libero_path
    from libero.libero.envs import OffScreenRenderEnv

    np.random.seed(7)
    suite = benchmark.get_benchmark_dict()["libero_object"]()
    task = suite.get_task(2)
    assert task.language == "pick up the salad dressing and place it in the basket"
    # These are trusted benchmark init states copied read-only from the source.
    init_path = (
        Path(get_libero_path("init_states"))
        / task.problem_folder
        / f"{task.name}.pruned_init"
    )
    initial_states = torch.load(init_path, map_location="cpu", weights_only=False)
    assert len(initial_states) >= args.episodes
    client = ModelClient(
        host=args.host,
        port=args.port,
        unnorm_key="franka",
        execute_horizon=8,
        action_ensemble=False,
        adaptive_ensemble_alpha=0,
    )
    server_metadata = client.client.get_server_metadata()
    assert server_metadata["checkpoint_sha256"] == EXPECTED_CHECKPOINT
    assert server_metadata["strict_load"] == {"missing_keys": [], "unexpected_keys": []}
    env = OffScreenRenderEnv(
        bddl_file_name=str(
            Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
        ),
        camera_heights=256,
        camera_widths=256,
    )
    env.seed(7)
    manifest = {
        "status": "running",
        "historical_trajectory_export": False,
        "description": "Fresh pure-GAWM rerun of the historical task protocol; historical per-step trajectories were not saved.",
        "suite": "libero_object",
        "task_id": 2,
        "task_language": task.language,
        "seed": 7,
        "num_trials": args.episodes,
        "initial_state_indices": list(range(args.episodes)),
        "max_policy_steps": 280,
        "settling_steps": 10,
        "execute_horizon": 8,
        "temporal_action_ensemble": False,
        "fps": 10,
        "video_note": "Agentview RGB before each policy action plus the final observation; settling steps omitted.",
        "checkpoint_sha256": EXPECTED_CHECKPOINT,
        "init_states_sha256": hashlib.sha256(init_path.read_bytes()).hexdigest(),
        "simulation_versions": {
            name: metadata.version(name)
            for name in ("robosuite", "mujoco", "numpy", "torch", "imageio")
        },
        "server_metadata": server_metadata,
        "policy_endpoint": f"ws://{args.host}:{args.port}",
        "original_evaluation_directory": "/data/liuyang/gawm-exp/playground/Evaluations/abl_libero_full_s42_20261017_libero_50ep_seed7",
        "differences": [
            "Newly executed trajectories, not recovered historical videos",
            "Inference uses the already-deployed PyTorch 2.11/CUDA13 server; historical inference used PyTorch 2.8",
            "Video adds the final post-action frame",
        ],
        "results": [],
    }

    def save():
        (root / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
        with (root / "index.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(
                stream,
                fieldnames=[
                    "episode",
                    "initial_state_index",
                    "success",
                    "policy_steps",
                    "model_queries",
                    "video",
                    "trajectory",
                    "elapsed_s",
                ],
            )
            writer.writeheader()
            writer.writerows(manifest["results"])

    save()
    try:
        for episode in range(args.episodes):
            started = time.monotonic()
            client.reset(task_description=task.language)
            env.reset()
            observation = env.set_init_state(initial_states[episode])
            for _ in range(10):
                observation, _, _, _ = env.step([0] * 6 + [-1])
            states = [np.asarray(env.get_sim_state()).copy()]
            proprio_states = [proprio(observation)]
            actions, chunks, query_steps = [], [], []
            video_path = root / "videos" / f"episode_{episode + 1:02d}.mp4"
            if video_path.exists():
                raise FileExistsError(video_path)
            # Save the model XML and state/action sequence for future exact replay.
            (root / "episodes" / f"episode_{episode + 1:02d}.xml").write_text(
                env.env.sim.model.get_xml()
            )
            success = False
            with imageio.get_writer(
                video_path, fps=10, codec="libx264", macro_block_size=1
            ) as video:
                for step in range(280):
                    main_image = np.ascontiguousarray(
                        observation["agentview_image"][::-1, ::-1]
                    )
                    wrist_image = np.ascontiguousarray(
                        observation["robot0_eye_in_hand_image"][::-1, ::-1]
                    )
                    video.append_data(main_image)
                    response = client.step(
                        example={
                            "image": [main_image, wrist_image],
                            "lang": task.language,
                            "state": proprio(observation),
                        },
                        step=step,
                    )
                    if step % 8 == 0:
                        chunks.append(client.raw_actions.copy())
                        query_steps.append(step)
                    raw = response["raw_action"]
                    gripper = 1.0 - 2.0 * (
                        float(np.asarray(raw["open_gripper"]).reshape(-1)[0]) > 0.5
                    )
                    action = np.concatenate(
                        [raw["world_vector"], raw["rotation_delta"], [gripper]]
                    ).astype(np.float32)
                    if action.shape != (7,) or not np.isfinite(action).all():
                        raise ValueError("Invalid native action")
                    observation, _, done, _ = env.step(action.tolist())
                    actions.append(action)
                    states.append(np.asarray(env.get_sim_state()).copy())
                    proprio_states.append(proprio(observation))
                    if done:
                        success = True
                        break
                video.append_data(
                    np.ascontiguousarray(observation["agentview_image"][::-1, ::-1])
                )
            trajectory = root / "episodes" / f"episode_{episode + 1:02d}.npz"
            np.savez_compressed(
                trajectory,
                actions=np.asarray(actions),
                sim_states=np.asarray(states),
                proprio_states=np.asarray(proprio_states),
                predicted_chunks=np.asarray(chunks),
                query_steps=np.asarray(query_steps),
                initial_state_index=episode,
                success=success,
            )
            result = {
                "episode": episode + 1,
                "initial_state_index": episode,
                "success": success,
                "policy_steps": len(actions),
                "model_queries": len(chunks),
                "video": str(video_path.relative_to(root)),
                "trajectory": str(trajectory.relative_to(root)),
                "elapsed_s": round(time.monotonic() - started, 3),
            }
            manifest["results"].append(result)
            manifest["successes"] = sum(row["success"] for row in manifest["results"])
            save()
            print(json.dumps(result), flush=True)
        manifest["status"] = "complete"
    except Exception as exc:
        manifest.update(status="failed", error=str(exc))
        raise
    finally:
        save()
        client.client.close()
        env.close()
    print(
        json.dumps(
            {
                "status": "complete",
                "episodes": args.episodes,
                "successes": manifest["successes"],
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
