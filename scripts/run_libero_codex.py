"""Run RPent's native Codex planner using the existing ChatGPT login."""

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

from codex_cli_bin import bundled_codex_path
from rpent.utils.rpc import wait_for_ready
from rpent.utils.rpc.http_rpc import HttpRpcClient

from scripts.run_naming import libero_run_name


def run():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path, default=ROOT / "configs/liberopro-local.toml"
    )
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--max-turns", type=int)
    parser.add_argument("--planner-timeout-s", type=int)
    parser.add_argument("--max-episode-steps", type=int)
    args = parser.parse_args()
    config = tomllib.loads(args.config.read_text())
    backend = config.get("policy_backend", "openpi")
    sim, bridge, policy, planner = (
        config["simulation"],
        config["bridge"],
        config[backend],
        config["codex"],
    )
    limits = {
        key: getattr(args, key) if getattr(args, key) is not None else planner[key]
        for key in ("max_turns", "planner_timeout_s", "max_episode_steps")
    }
    if any(v <= 0 for v in limits.values()):
        raise ValueError("Planner and simulator budgets must be positive")
    if planner["reasoning_effort"] == "none":
        raise ValueError("This Astra profile requires reasoning; use low or higher")
    if bridge["rpc_timeout_s"] <= policy["request_timeout_s"]:
        raise ValueError("Bridge RPC timeout must exceed OpenPI request timeout")
    with socket.socket() as probe:
        probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        probe.bind((bridge["host"], bridge["port"]))

    env = {
        **os.environ,
        "CODEX_BIN": str(bundled_codex_path()),
        "PYDANTIC_AI_NO_BANNER": "1",
    }
    # This entry point intentionally uses the existing local ChatGPT login.
    for key in (
        "OPENAI_API_KEY",
        "OPENAI_BASE_URL",
        "CODEX_API_KEY",
        "CODEX_BASE_URL",
        "PYTHONPATH",
        "PYTHONHOME",
        "VIRTUAL_ENV",
        "CONDA_PREFIX",
    ):
        env.pop(key, None)
    for key in ("NO_PROXY", "no_proxy"):
        env[key] = ",".join(
            filter(None, [env.get(key), "127.0.0.1", "localhost", policy["host"]])
        )
    auth = subprocess.run(
        [env["CODEX_BIN"], "login", "status"],
        env=env,
        capture_output=True,
        text=True,
        check=False,
        timeout=15,
    )
    if auth.returncode or "Logged in using ChatGPT" not in auth.stdout + auth.stderr:
        raise RuntimeError("Codex ChatGPT login is required; run codex login locally")

    output = (
        args.output_dir
        or ROOT / "outputs" / libero_run_name(backend, "codex", model=planner["model"])
    ).resolve()
    output.mkdir(parents=True, exist_ok=False)
    memory = (ROOT / planner["memory_dir"]).resolve()
    (memory / "global").mkdir(parents=True, exist_ok=True)
    for path in (memory / "MEMORY.md", memory / "global/GLOBAL_MEMORY.md"):
        if not path.exists():
            path.write_text(
                "# Local LIBERO-PRO memory\nNo task-specific memory recorded yet.\n"
            )
    manifest = {
        "planner": "codex",
        "policy_backend": backend,
        "model": planner["model"],
        "reasoning_effort": planner["reasoning_effort"],
        "reasoning_enabled": True,
        "auth": "existing ChatGPT login",
        "api_key_used": False,
        "mode": "sync",
        "config": config,
        "limits": limits,
        "memory_dir": str(memory),
        "run_status": "starting",
        "task_success": None,
    }

    def save():
        (output / "configuration.json").write_text(
            json.dumps(manifest, indent=2) + "\n"
        )

    save()
    print(f"Output: {output}", flush=True)
    endpoint = f"http://{bridge['host']}:{bridge['port']}"
    rpc = HttpRpcClient(endpoint)
    with (output / "bridge.log").open("w") as bridge_log:
        process = subprocess.Popen(
            [
                sys.executable,
                "-m",
                f"scripts.{backend}_rpent_bridge",
                "--config",
                str(args.config.resolve()),
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
                "--cuda-device",
                str(sim["cuda_device"]),
                "--vla-endpoint",
                endpoint,
                "--vla-timeout-s",
                str(bridge["rpc_timeout_s"]),
                "--sam3-endpoint",
                sim["sam3_endpoint"],
                "--planner",
                "codex",
                "--model",
                planner["model"],
                "--reasoning-effort",
                planner["reasoning_effort"],
                "--max-turns",
                str(limits["max_turns"]),
                "--planner-timeout-s",
                str(limits["planner_timeout_s"]),
                "--max-episode-steps",
                str(limits["max_episode_steps"]),
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
            manifest.update(run_status="running", command=command)
            save()
            with (output / "console.log").open("w") as log:
                completed = subprocess.run(
                    command,
                    cwd=ROOT / "RPent",
                    env=env,
                    stdin=subprocess.DEVNULL,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    check=False,
                )
            manifests = list(output.glob("transcript_*.json"))
            if len(manifests) != 1:
                raise RuntimeError(
                    f"Expected one RPent transcript; see {output / 'console.log'}"
                )
            transcript = json.loads(manifests[0].read_text())
            states = json.loads((output / "states.json").read_text())["steps"]
            manifest.update(
                task_success=bool(states[-1]["terminated"]) if states else None,
                planner_finish=transcript.get("finish"),
                planner_error=transcript.get("error"),
                stats=transcript.get("stats"),
                exit_code=completed.returncode,
            )
            if completed.returncode or transcript.get("error"):
                raise RuntimeError(
                    f"RPent planner reported an error; see {output / 'console.log'}"
                )
            manifest["run_status"] = "finished"
        except KeyboardInterrupt:
            manifest.update(run_status="interrupted", error="Interrupted by operator")
            raise
        except Exception as exc:
            manifest.update(run_status="failed", error=str(exc))
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
            save()
    print(
        f"Run finished; native task success: {manifest['task_success']}. Output: {output}",
        flush=True,
    )


if __name__ == "__main__":
    run()
