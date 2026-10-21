"""Audit a saved fake-GPT/real-GAWM episode without starting any services."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import imageio.v2 as imageio
import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def verify(output):
    manifest = json.loads((output / "configuration.json").read_text())
    transcript = json.loads(next(output.glob("transcript_*.json")).read_text())
    steps = json.loads((output / "states.json").read_text())["steps"]
    timing = [
        json.loads(line)
        for line in (output / "network-timing.jsonl").read_text().splitlines()
    ]
    assert manifest["status"] == "completed"
    assert manifest["planner"] == "local-fake" and manifest["paid_api_calls"] == 0
    command = manifest["command"]
    assert command[command.index("--planner") + 1] == "api"
    assert command[command.index("--base-url") + 1].startswith("http://127.0.0.1:")
    assert manifest["policy_status"]["source"] == "starvla-gawm-libero"
    assert manifest["policy_status"]["metadata"]["strict_load"] == {
        "missing_keys": [],
        "unexpected_keys": [],
    }
    assert not transcript["error"]
    assert transcript["stats"]["total_input_tokens"] == 0
    assert transcript["stats"]["total_output_tokens"] == 0
    tools = [
        message for message in transcript["messages"] if message.get("role") == "tool"
    ]
    segments = []
    for message in tools:
        payload = json.loads(message["content"])
        assert not payload.get("error"), payload
        assert not payload.get("log", {}).get("result", {}).get("error"), payload
        if message["name"] == "segment":
            assert payload["found"] and not payload.get("world_error"), payload
            assert np.isfinite(payload["world_xyz"]).all()
            segments.append(payload)
    assert len(segments) == 2
    assert len(timing) == manifest["requested_chunks"]
    assert all(row["ok"] for row in timing)
    assert manifest["last_prediction"]["action_shape"] == [1, 8, 7]
    expected_frames = manifest["requested_chunks"] * 8 + 1
    frame_count = 0
    with imageio.get_reader(output / "episode.mp4") as reader:
        video_metadata = reader.get_meta_data()
        for frame in reader:
            assert frame.shape == (256, 256, 3)
            frame_count += 1
    assert frame_count == expected_frames, (frame_count, expected_frames)
    for step in steps:
        index = f"{step['step_idx']:02}"
        for camera in ("agentview", "wrist"):
            imageio.imread(output / f"{camera}.png/{index}.png")
            json.loads((output / f"{camera}_metadata.json/{index}.json").read_text())
            for suffix in ("depth", "world"):
                with np.load(output / f"{camera}_{suffix}.npz/{index}.npz") as data:
                    assert data.files
    return {
        "passed": True,
        "output_dir": str(output),
        "suite": transcript["suite"],
        "task": transcript["task"],
        "seed": transcript["seed"],
        "task_language": manifest["task_language"],
        "policy_prompt": manifest["policy_prompt"],
        "fake_requests": manifest["fake_requests"],
        "stats": transcript["stats"],
        "gawm_requests": len(timing),
        "native_actions": frame_count,
        "saved_observations": len(steps),
        "video": {
            "frames": frame_count,
            "fps": video_metadata["fps"],
            "duration_s": video_metadata["duration"],
        },
        "segments": segments,
        "round_trip_ms": [
            row["client_timing"]["transport_round_trip_ms"] for row in timing
        ],
        "server_infer_ms": [row["server_timing"]["infer_ms"] for row in timing],
        "native_task_success": steps[-1]["terminated"],
        "truncated": steps[-1]["truncated"],
        "planner_finish": transcript["finish"],
        "real_codex_started": False,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    output = args.output.resolve()
    report = verify(output)
    review = ROOT / "review" / output.name
    review.mkdir(parents=True, exist_ok=True)
    destination = review / "verification.json"
    destination.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    print(destination)


if __name__ == "__main__":
    main()
