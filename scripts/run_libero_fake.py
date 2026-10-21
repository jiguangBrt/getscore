"""Run bounded LIBERO-PRO action calls through a loopback-only fake planner."""

from __future__ import annotations

import argparse
import json
import os
import socket
import subprocess
import sys
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "RPent"))

from rpent.utils.rpc import wait_for_ready
from rpent.utils.rpc.http_rpc import HttpRpcClient
from tests.e2e_tests.offline_planner_server import (
    OfflinePlannerServer,
    ScriptedToolCall,
)

from scripts.run_naming import libero_run_name


class ZeroTokenPlannerServer(OfflinePlannerServer):
    """Keep synthetic token accounting at zero as well as paid API usage."""

    @staticmethod
    def _send_json(handler, status, payload):
        if "usage" in payload:
            payload["usage"] = {
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0,
            }
        OfflinePlannerServer._send_json(handler, status, payload)

    @staticmethod
    def _send_stream(handler, response):
        response["usage"] = {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
        }
        OfflinePlannerServer._send_stream(handler, response)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path, default=ROOT / "configs/liberopro-local.toml"
    )
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--chunks", type=int)
    args = parser.parse_args()
    config = tomllib.loads(args.config.read_text())
    backend = config.get("policy_backend", "openpi")
    sim, bridge, policy = config["simulation"], config["bridge"], config[backend]
    count = args.chunks if args.chunks is not None else sim["action_chunks"]
    if count <= 0 or bridge["rpc_timeout_s"] <= policy["request_timeout_s"]:
        raise ValueError(
            "chunks must be positive and RPC timeout must exceed OpenPI request timeout"
        )
    with socket.socket() as probe:
        probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        probe.bind((bridge["host"], bridge["port"]))
    output = (
        args.output_dir or ROOT / "outputs" / libero_run_name(backend, "fake")
    ).resolve()
    output.mkdir(parents=True, exist_ok=False)
    env = {
        **os.environ,
        "OPENAI_API_KEY": "local-fake-no-paid-api",
        "PYDANTIC_AI_NO_BANNER": "1",
        "MUJOCO_GL": "egl",
        "PYOPENGL_PLATFORM": "egl",
        "LIBERO_TYPE": "pro",
        "NO_PROXY": "127.0.0.1,localhost,192.168.50.73",
        "no_proxy": "127.0.0.1,localhost,192.168.50.73",
        "RPENT_RLINF_ROOT": str(ROOT / "data/rlinf-runtime"),
    }
    for key in ("PYTHONPATH", "PYTHONHOME", "VIRTUAL_ENV", "CONDA_PREFIX"):
        env.pop(key, None)
    os.environ.update(
        {key: env[key] for key in ("MUJOCO_GL", "PYOPENGL_PLATFORM", "LIBERO_TYPE")}
    )
    sys.path.insert(0, str(ROOT / "data/rlinf-runtime"))
    from rlinf.envs.sim.libero.utils import benchmark

    task = benchmark.get_benchmark(sim["suite"])().get_task(sim["task"])
    prompt = task.language
    policy_prompt = sim.get("policy_prompt", prompt)
    memory = output / "fake-memory"
    (memory / "global").mkdir(parents=True)
    (memory / "MEMORY.md").write_text(
        "# Local fake planner\nNo task-specific memory.\n"
    )
    (memory / "global/GLOBAL_MEMORY.md").write_text(
        "# Global memory\nNo task-specific memory.\n"
    )
    script = tuple(
        ScriptedToolCall("pi0_doubled", {"prompt": policy_prompt, "max_chunks": 1})
        for _ in range(count)
    )
    if backend == "gawm":
        script = (
            ScriptedToolCall("segment", {"prompt": "basket", "camera": "agentview"}),
            ScriptedToolCall(
                "segment",
                {"prompt": "salad dressing", "camera": "agentview", "min_score": 0.1},
            ),
            ScriptedToolCall("set_gripper", {"gripper": -1.0, "steps": 1}),
        ) + script
    script += (
        ScriptedToolCall(
            "finish",
            {
                "status": "stuck",
                "summary": "Fake planner completed its bounded synchronous action calls; inspect native success separately.",
            },
        ),
    )
    manifest = {
        "planner": "local-fake",
        "policy_backend": backend,
        "paid_api_calls": 0,
        "mode": "sync",
        "config": config,
        "task_language": prompt,
        "policy_prompt": policy_prompt,
        "requested_chunks": count,
    }
    (output / "configuration.json").write_text(json.dumps(manifest, indent=2) + "\n")
    endpoint = f"http://{bridge['host']}:{bridge['port']}"
    rpc = HttpRpcClient(endpoint)
    with (output / "bridge.log").open("w") as bridge_log:
        process = subprocess.Popen(
            [
                sys.executable,
                "-m",
                f"scripts.{backend}_rpent_bridge",
                "--config",
                str(args.config),
                "--timing-log",
                str(output / "network-timing.jsonl"),
            ],
            cwd=ROOT,
            env=env,
            stdout=bridge_log,
            stderr=subprocess.STDOUT,
        )
        try:
            wait_for_ready(rpc, timeout_s=policy["connect_timeout_s"] + 15)
            if process.poll() is not None:
                raise RuntimeError(f"Owned bridge exited; see {output / 'bridge.log'}")
            manifest["policy_status"] = rpc.call("vla.status")
            with ZeroTokenPlannerServer(script) as fake:
                # The fake endpoint is generated by a server bound to loopback.
                # No caller-supplied planner URL or model name is accepted here.
                command = [
                    str(ROOT / ".venv/bin/rpent"),
                    "--robot",
                    "libero",
                    "--libero-type",
                    "pro",
                    "--suite",
                    sim["suite"],
                    "--task",
                    str(sim["task"]),
                    "--seed",
                    str(sim["seed"]),
                    "--max-episode-steps",
                    str(sim["max_episode_steps"]),
                    "--cuda-device",
                    str(sim["cuda_device"]),
                    "--vla-endpoint",
                    endpoint,
                    "--vla-timeout-s",
                    str(bridge["rpc_timeout_s"]),
                    "--sam3-endpoint",
                    sim["sam3_endpoint"],
                    "--planner",
                    "api",
                    "--model",
                    "openai-chat:rpent-local-fake",
                    "--base-url",
                    fake.base_url,
                    "--max-turns",
                    str(len(script)),
                    "--no-images",
                    "--memory-profile",
                    "local",
                    "--memory-dir",
                    str(memory),
                    "--output-dir",
                    str(output),
                ]
                if backend == "gawm":
                    command[:1] = [
                        sys.executable,
                        str(ROOT / "scripts/gawm_rpent_entrypoint.py"),
                    ]
                manifest["command"] = command
                with (output / "console.log").open("w") as log:
                    completed = subprocess.run(
                        command,
                        check=False,
                        cwd=ROOT / "RPent",
                        env=env,
                        stdout=log,
                        stderr=subprocess.STDOUT,
                    )
                if completed.returncode:
                    raise RuntimeError(f"RPent failed; see {output / 'console.log'}")
                fake.assert_complete()
            manifest["fake_requests"] = fake.request_count
            manifest["status"] = "completed"
            transcript_path = next(output.glob("transcript_*.json"))
            transcript = json.loads(transcript_path.read_text())
            states = json.loads((output / "states.json").read_text())["steps"]
            manifest["task_success"] = bool(states[-1]["terminated"])
            manifest["planner_finish"] = transcript.get("finish")
            manifest["planner_error"] = transcript.get("error")
            manifest["last_prediction"] = rpc.call("vla.last_prediction")
            if transcript.get("error"):
                raise RuntimeError(
                    f"Fake planner reported an error: {transcript['error']}"
                )
            (output / "configuration.json").write_text(
                json.dumps(manifest, indent=2) + "\n"
            )
            print(output)
        except Exception as exc:
            manifest.update(status="failed", error=str(exc))
            (output / "configuration.json").write_text(
                json.dumps(manifest, indent=2) + "\n"
            )
            raise
        finally:
            if process.poll() is None:
                try:
                    rpc.call("shutdown", timeout_s=3)
                except Exception:  # noqa: BLE001 - reap the owned bridge on RPC failure
                    process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()


if __name__ == "__main__":
    main()
