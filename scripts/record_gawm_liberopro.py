"""Record pure GAWM rollouts on five pinned LIBERO-PRO swap initial states."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "RPent"), str(ROOT / "data/rlinf-runtime")]
os.environ.update({
    "MUJOCO_GL": "egl", "PYOPENGL_PLATFORM": "egl", "LIBERO_TYPE": "pro",
    "LIBERO_PRO_ASSET_PATH": str(ROOT / "data/libero-pro/assets"),
    "LIBERO_ASSET_PATH": str(ROOT / "data/libero-pro/assets"),
    "LIBERO_CONFIG_PATH": str(ROOT / "data/libero-pro/config"),
    "LIBERO_PRO_CONFIG_PATH": str(ROOT / "data/libero-pro/config-pro"),
    "RPENT_RLINF_ROOT": str(ROOT / "data/rlinf-runtime"),
    "HF_HUB_DISABLE_IMPLICIT_TOKEN": "1",
})
for key in ("LIBERO_SUFFIX", "LIBERO_PERTURBATION"):
    os.environ.pop(key, None)

import imageio.v2 as imageio
import numpy as np

PROMPT = "pick up the salad dressing and place it in the basket"
REFERENCE = ROOT / "outputs/_archive/runs/liberopro-gawm-gpt6astra-20261021-162733"
CHECKPOINT_SHA256 = "f698e76ae129ccfbe8349418acd79044f2a1ca6366d098546afb477fd79e7e13"


def jsonable(value):
    if isinstance(value, dict):
        return {key: jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def write_json(path, value):
    path.write_text(json.dumps(jsonable(value), ensure_ascii=False, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--host", default="192.168.50.73")
    parser.add_argument("--port", type=int, default=8115)
    parser.add_argument("--max-steps", type=int, default=1000)
    args = parser.parse_args()
    if args.max_steps < 1:
        parser.error("--max-steps must be positive")
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    for folder in ("videos", "episodes", "images", "sources"):
        (output / folder).mkdir()

    from robots.libero.env_server import LiberoEnvFacade, make_env
    from scripts.gawm_rpent_bridge import libero_actions, libero_example
    from scripts.openpi_transport import OpenPiTransport
    from rlinf.envs.sim.libero.utils import benchmark
    import liberopro.liberopro as libpro

    transport = OpenPiTransport(args.host, args.port, connect_timeout_s=20,
                               request_timeout_s=120, timing_path=output / "network-timing.jsonl")
    manifest = {
        "controller": "pure GAWM, no GPT/planner/scripted approach or grasp",
        "suite": "libero_object_swap", "task_id": 2, "seeds": list(range(5)),
        "policy_prompt": PROMPT, "max_episode_steps": args.max_steps,
        "execute_horizon": 8, "fps": 20,
        "video_frames": "initial observation, then one frame per native action",
        "initialization": "LIBERO-PRO pruned_init trials 0..4; native reset settling only",
        "reference_gpt_gawm_run": str(REFERENCE), "episodes": [],
    }
    try:
        metadata = transport.metadata()
        if metadata.get("source") != "starvla-gawm-libero":
            raise RuntimeError("Unexpected policy service")
        if metadata.get("checkpoint_sha256") != CHECKPOINT_SHA256:
            raise RuntimeError(f"Unexpected checkpoint metadata: {metadata}")
        manifest["policy_metadata"] = metadata
        task = benchmark.get_benchmark(manifest["suite"])().get_task(2)
        manifest["benchmark_language"] = task.language
        for kind, filename in (("bddl_files", task.bddl_file),
                               ("init_states", task.init_states_file)):
            source = Path(libpro.get_libero_path(kind)) / task.problem_folder / filename
            (output / "sources" / source.name).write_bytes(source.read_bytes())
            manifest[kind] = {"path": str(source), "sha256": hashlib.sha256(source.read_bytes()).hexdigest()}
        (output / "sources" / "reference-configuration.json").write_bytes(
            (REFERENCE / "configuration.json").read_bytes())
        write_json(output / "manifest.json", manifest)

        for seed in range(5):
            name = f"episode_{seed + 1:02d}_seed{seed}"
            print(f"START {name}", flush=True)
            started = time.monotonic()
            env = make_env(2, seed, manifest["suite"], max_episode_steps=args.max_steps)
            facade = LiberoEnvFacade(env, meta={})
            actions, states, sim_states, raw_chunks, query_steps = [], [], [], [], []
            rewards, terminations, truncations = [], [], []
            video = output / "videos" / f"{name}.mp4"
            try:
                obs, _ = facade.reset()
                raw = facade.raw_obs()
                initial = {k: v for k, v in raw.items() if "image" not in k and "depth" not in k}
                write_json(output / "episodes" / f"{name}_initial.json", initial)
                initial_state = np.asarray(env.env.get_sim_state()[0]).copy()
                sim_states.append(initial_state)
                states.append(np.asarray(obs["states"]).copy())
                imageio.imwrite(output / "images" / f"{name}_initial.png", obs["main_images"])
                matched = None
                if seed == 0:
                    reference = json.loads((REFERENCE / "states.json").read_text())["steps"][0]["state"]
                    matched = {key: bool(np.allclose(raw[key], reference[key], atol=1e-10, rtol=0))
                               for key in ("robot0_eef_pos", "robot0_eef_quat", "robot0_gripper_qpos")}
                    if not all(matched.values()):
                        raise RuntimeError(f"Seed 0 differs from reference initial robot state: {matched}")
                with imageio.get_writer(video, fps=20, codec="libx264", quality=8,
                                        macro_block_size=None) as writer:
                    writer.append_data(np.ascontiguousarray(obs["main_images"]))
                    term = trunc = False
                    while not (term or trunc):
                        batch = {"states": np.asarray(obs["states"])[None],
                                 "main_images": np.asarray(obs["main_images"])[None],
                                 "wrist_images": np.asarray(obs["wrist_images"])[None],
                                 "task_descriptions": [PROMPT]}
                        prediction = transport.infer({"examples": [libero_example(batch)],
                                                     "unnorm_key": "franka", "do_sample": False})
                        if not prediction.get("ok"):
                            raise RuntimeError(f"Policy inference failed: {prediction}")
                        chunk = np.asarray(prediction["data"]["actions"])
                        raw_chunks.append(chunk[0].copy())
                        query_steps.append(len(actions))
                        for action in libero_actions(chunk, 8)[0]:
                            obs, reward, term, trunc, info = facade.step(action)
                            actions.append(action.copy())
                            states.append(np.asarray(obs["states"]).copy())
                            sim_states.append(np.asarray(env.env.get_sim_state()[0]).copy())
                            rewards.append(float(reward))
                            terminations.append(bool(term))
                            truncations.append(bool(trunc))
                            writer.append_data(np.ascontiguousarray(obs["main_images"]))
                            if term or trunc:
                                break
                        if len(actions) % 200 == 0:
                            print(f"PROGRESS {name} actions={len(actions)}", flush=True)
                native_success = bool(env.env.workers[0].check_success())
                if native_success != bool(term):
                    raise RuntimeError("Native success and terminated disagree")
                imageio.imwrite(output / "images" / f"{name}_final.png", obs["main_images"])
                np.savez_compressed(output / "episodes" / f"{name}.npz",
                                    actions=np.asarray(actions), proprio_states=np.asarray(states),
                                    sim_states=np.asarray(sim_states), raw_predicted_chunks=np.asarray(raw_chunks),
                                    query_steps=np.asarray(query_steps), rewards=np.asarray(rewards),
                                    terminated=np.asarray(terminations), truncated=np.asarray(truncations))
                result = {"episode": seed + 1, "seed": seed, "initial_state_index": int(env.trial_ids[0]),
                          "success": native_success, "terminated": bool(term), "truncated": bool(trunc),
                          "actions": len(actions), "queries": len(query_steps), "frames": len(actions) + 1,
                          "elapsed_s": round(time.monotonic() - started, 2),
                          "initial_sim_state_sha256": hashlib.sha256(initial_state.tobytes()).hexdigest(),
                          "reference_initial_robot_matches": matched,
                          "video": str(video.relative_to(output))}
                manifest["episodes"].append(result)
                write_json(output / "manifest.json", manifest)
                print("RESULT " + json.dumps(result), flush=True)
            finally:
                facade.close()
        with (output / "index.csv").open("w", newline="") as f:
            fields = ["episode", "seed", "initial_state_index", "success", "terminated", "truncated",
                      "actions", "queries", "frames", "elapsed_s", "video"]
            writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(manifest["episodes"])
    finally:
        transport.close()


if __name__ == "__main__":
    main()
