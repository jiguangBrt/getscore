"""Saved-record interpretation and the local viewer's file-serving boundary."""

import http.client
import json
import struct
import threading
from pathlib import Path
from urllib.parse import urlencode

import pytest

from run_viewer.parser import parse_run
from run_viewer.server import ViewerServer


def write_json(path, value):
    path.write_text(json.dumps(value), encoding="utf-8")


def mp4(path, frames=18, fps=10):
    def atom(kind, body):
        return struct.pack(">I4s", len(body) + 8, kind) + body

    mdhd = atom(b"mdhd", struct.pack(">6I", 0, 0, 0, fps, frames, 0))
    handler = atom(b"hdlr", bytes(8) + b"vide")
    samples = atom(b"stsz", struct.pack(">3I", 0, 0, frames))
    path.write_bytes(
        atom(
            b"moov",
            atom(
                b"trak",
                atom(b"mdia", mdhd + handler + atom(b"minf", atom(b"stbl", samples))),
            ),
        )
    )


@pytest.fixture
def saved_run(tmp_path):
    root = tmp_path / "run"
    root.mkdir()
    write_json(
        root / "configuration.json",
        {
            "model": "recorded-model",
            "policy_backend": "gawm",
            "config": {"gawm": {"execute_horizon": 8}},
        },
    )
    write_json(
        root / "states.json",
        {
            "steps": [
                {"step_idx": 0},
                {
                    "step_idx": 1,
                    "command": {"action": "move_to", "xyz": [0, 0, 1]},
                    "result": {"steps_used": 3},
                    "elapsed_s": 0.2,
                },
                {
                    "step_idx": 2,
                    "command": {"action": "pi0_pick", "chunks": 2},
                    "result": {"chunks_used": 2},
                    "elapsed_s": 0.4,
                    "terminated": True,
                },
            ]
        },
    )
    mp4(root / "episode.mp4")
    rows = []

    def item(identifier, kind, at, **fields):
        value = {"id": identifier, "type": kind, **fields}
        rows.append(
            {"method": "item/started", "payload": {"started_at_ms": at, "item": value}}
        )
        rows.append(
            {
                "method": "item/completed",
                "payload": {"completed_at_ms": at + 100, "item": value},
            }
        )

    item("prompt", "userMessage", 1000000, content=[{"text": "Recorded initial input"}])
    item("empty", "reasoning", 1000200, summary=[], content=[])
    item("text", "agentMessage", 1000400, text="I will inspect and move.")
    item(
        "move",
        "mcpToolCall",
        1000600,
        tool="move_to",
        arguments={"xyz": [0, 0, 1]},
        result={"content": [{"type": "text", "text": '{"step": 1}'}]},
        status="completed",
    )
    usage = {
        "method": "thread/tokenUsage/updated",
        "payload": {
            "token_usage": {
                "total": {
                    "input_tokens": 100,
                    "cached_input_tokens": 40,
                    "output_tokens": 10,
                    "reasoning_output_tokens": 3,
                }
            }
        },
    }
    rows.extend([usage, usage])  # Repeated cumulative notifications are not replies.
    item(
        "pick",
        "mcpToolCall",
        1000800,
        tool="pi0_pick",
        arguments={"chunks": 2},
        result={"step": 2},
        status="completed",
    )
    item(
        "failed",
        "mcpToolCall",
        1001000,
        tool="pi0_pick",
        arguments={"chunks": 3},
        result={"error": "offline"},
        status="failed",
    )
    item(
        "shell",
        "commandExecution",
        1001200,
        command="echo recorded",
        aggregated_output="recorded",
        exit_code=0,
        status="completed",
    )
    rows.append(
        {
            "method": "thread/tokenUsage/updated",
            "payload": {
                "token_usage": {"total": {"input_tokens": 250, "output_tokens": 30}}
            },
        }
    )
    rows.append(
        {
            "method": "turn/completed",
            "payload": {"turn": {"completed_at": 1001, "status": "completed"}},
        }
    )
    (root / "codex.stream.jsonl").write_text(
        "\n".join(json.dumps(row) for row in rows) + '\n{"incomplete":',
        encoding="utf-8",
    )
    return root


def test_stream_replies_tokens_public_text_and_observations(saved_run):
    report = parse_run(saved_run)
    assert report["outcome"]["status"] == "success"
    assert report["outcome"]["native_success"] is True
    assert report["metrics"]["turns"] == 2
    assert report["metrics"]["tool_calls"] == 4
    assert report["metrics"]["tokens"]["input_tokens"] == 250
    assert report["turns"][1]["usage"]["input_tokens"] == 150
    assert not any(event["title"] == "Codex 推理摘要" for event in report["events"])
    tools = [event for event in report["events"] if event.get("tool")]
    assert (tools[0]["before_state"], tools[0]["after_state"]) == (0, 1)
    assert tools[0]["public_before"] == "I will inspect and move."
    assert (tools[1]["before_state"], tools[1]["after_state"]) == (1, 2)
    assert (
        tools[2]["after_state"] is None
    )  # A failed call does not acquire an invented image.
    assert report["events"][-1]["title"] == "Codex 运行结束"
    assert any("不完整" in warning for warning in report["warnings"])


def test_video_alignment_is_withdrawn_if_native_frames_disagree(saved_run):
    report = parse_run(saved_run)
    clips = [
        event["video_range"] for event in report["events"] if "video_range" in event
    ]
    assert clips == [
        {"start_s": 0, "end_s": 0.2, "frames": 2},
        {"start_s": 0.2, "end_s": 1.8, "frames": 16},
    ]
    mp4(saved_run / "episode.mp4", frames=19)
    report = parse_run(saved_run)
    assert not any("video_range" in event for event in report["events"])
    assert any("帧数不一致" in warning for warning in report["warnings"])


def test_fake_messages_preserve_readonly_tool_arguments(tmp_path):
    write_json(
        tmp_path / "transcript_fake.json",
        {
            "messages": [
                {"role": "user", "content": "Start"},
                {
                    "role": "assistant",
                    "content": [
                        {
                            "type": "tool_use",
                            "id": "a",
                            "name": "segment",
                            "input": {"prompt": "basket"},
                        }
                    ],
                },
                {
                    "role": "tool",
                    "tool_call_id": "a",
                    "name": "segment",
                    "content": '{"found":true}',
                },
            ]
        },
    )
    report = parse_run(tmp_path)
    tool = next(event for event in report["events"] if event.get("tool"))
    assert tool["arguments"] == {"prompt": "basket"}
    assert tool["at_ms"] is None
    assert report["metrics"]["tokens"]["input_tokens"] == 0


def test_native_action_counts_override_tool_estimates(saved_run):
    # Arrival/termination and shortened final chunks can differ from the
    # legacy steps_used/chunks_used estimates despite the same total frames.
    write_json(saved_run / "native-action-counts.json", {
        "total_frames": 18,
        "steps": {"1": {"action": "move_to", "frames": 3},
                  "2": {"action": "pi0_pick", "frames": 15}},
    })
    report = parse_run(saved_run)
    clips = [event["video_range"] for event in report["events"] if "video_range" in event]
    assert [clip["frames"] for clip in clips] == [3, 15]
    assert clips[1]["start_s"] == .3
    assert not any("帧数不一致" in warning for warning in report["warnings"])


def test_pending_recovery_actions_have_clips_without_inventing_model_calls(tmp_path):
    write_json(tmp_path / "configuration.json", {"policy_backend": "openpi"})
    write_json(tmp_path / "states.json", {"steps": [
        {"step_idx": 0},
        {"step_idx": 1, "command": {"action": "resume_pending_policy_actions", "instruction": "grasp"},
         "result": {"executed_steps": 3}},
        {"step_idx": 2, "command": {"action": "move_to", "xyz": [0, 0, 1]},
         "result": {"steps_used": 3}, "terminated": True},
    ]})
    mp4(tmp_path / "episode.mp4", frames=5)
    write_json(tmp_path / "native-action-counts.json", {"total_frames": 5, "steps": {
        "1": {"action": "resume_pending_policy_actions", "frames": 3},
        "2": {"action": "move_to", "frames": 2},
    }})
    row = {"method": "item/completed", "payload": {"completed_at_ms": 1000, "item": {
        "id": "live", "type": "mcpToolCall", "tool": "move_to", "arguments": {"xyz": [0, 0, 1]},
        "result": {"step": 2}, "status": "completed",
    }}}
    (tmp_path / "00-planner.stream.jsonl").write_text(json.dumps(row) + "\n")
    report = parse_run(tmp_path)
    assert report["warnings"] == []
    assert report["metrics"]["tool_calls"] == 1
    actions = [event for event in report["events"] if event.get("action")]
    assert [event["after_state"] for event in actions] == [1, 2]
    assert [event["video_range"]["frames"] for event in actions] == [3, 2]
    assert actions[0]["at_ms"] is None
    assert actions[0]["source"]["file"] == "states.json"
    assert actions[1]["before_state"] == 1


def test_kimi_normalization_excludes_replayed_session_history(tmp_path):
    from scripts.organize_long_benchmark import normalize_events

    def chunk(at, text):
        return {"time_unix": at, "direction": "server", "message": {"method": "session/update", "params": {
            "update": {"sessionUpdate": "agent_message_chunk", "content": {"type": "text", "text": text}},
        }}}

    rows = [chunk(10, "old loaded context"),
            {"time_unix": 20, "direction": "client", "message": {"method": "session/prompt"}},
            chunk(21, "new response"), chunk(22, " continued")]
    (tmp_path / "kimi_acp.stream.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
    (tmp_path / "kimi_tool_calls.jsonl").write_text(json.dumps({
        "time_unix": 23, "tool": "view_env_state", "arguments": {"step": -1}, "result": {"step": 0},
    }) + "\n")
    name = normalize_events(tmp_path, "kimi")
    events = [json.loads(line) for line in (tmp_path / name).read_text().splitlines()]
    assert len(events) == 2
    assert events[0]["payload"]["item"]["text"] == "new response continued"
    assert events[1]["payload"]["item"]["tool"] == "view_env_state"
    assert all(event["source"]["file"] in {"kimi_acp.stream.jsonl", "kimi_tool_calls.jsonl"} for event in events)


def test_absent_usage_is_distinguished_from_recorded_zero_usage(tmp_path):
    write_json(tmp_path / "configuration.json", {"planner": "kimi"})
    assert parse_run(tmp_path)["metrics"]["token_usage_available"] is False
    write_json(tmp_path / "transcript_kimi.json", {"stats": {"total_input_tokens": 0, "total_output_tokens": 0}})
    assert parse_run(tmp_path)["metrics"]["token_usage_available"] is True
    write_json(tmp_path / "configuration.json", {"planner": "none"})
    (tmp_path / "transcript_kimi.json").unlink()
    assert parse_run(tmp_path)["metrics"]["token_usage_available"] is True


@pytest.fixture
def local_server(tmp_path):
    server = ViewerServer(("127.0.0.1", 0), tmp_path)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield server
    server.shutdown()
    server.server_close()
    thread.join()


def request(server, path, body=None, headers=None):
    client = http.client.HTTPConnection("127.0.0.1", server.server_port, timeout=3)
    client.request(
        "POST" if body is not None else "GET",
        path,
        body,
        {"Content-Type": "application/json", **(headers or {})},
    )
    response = client.getresponse()
    status, result, response_headers = (
        response.status,
        response.read(),
        dict(response.getheaders()),
    )
    client.close()
    return status, result, response_headers


def test_experiment_list_uses_natural_name_order(local_server, tmp_path):
    names = ["run-10", "run-9", "run-2", "Alpha-1"]
    for name in names:
        root = tmp_path / name
        root.mkdir()
        write_json(root / "configuration.json", {"model": "recorded-model"})
    # Creation / modification order must not determine the selection list.
    status, body, _ = request(
        local_server, "/api/open", json.dumps({"path": str(tmp_path)})
    )
    assert status == 200
    assert [run["name"] for run in json.loads(body)["runs"]] == [
        "Alpha-1", "run-2", "run-9", "run-10"
    ]


def test_opening_promoted_alias_keeps_the_requested_catalog(local_server, tmp_path):
    recordings = tmp_path / "batch" / "cases"
    recordings.mkdir(parents=True)
    recording = recordings / "long-original-4-kimi-pi05"
    recording.mkdir()
    write_json(recording / "configuration.json", {"planner": "kimi"})
    catalog = tmp_path / "outputs"
    catalog.mkdir()
    alias = catalog / recording.name
    alias.symlink_to(recording, target_is_directory=True)
    other = catalog / "long-swap-4-kimi-pi05"
    other.mkdir()
    write_json(other / "configuration.json", {"planner": "kimi"})
    status, body, _ = request(local_server, "/api/open", json.dumps({"path": str(alias)}))
    data = json.loads(body)
    assert status == 200
    assert data["path"] == str(catalog)
    assert data["report"]["path"] == str(alias)
    assert len(data["runs"]) == 2
    # Media is still constrained to the actual recording, not the catalog.
    assert local_server.runs[data["report"]["run_id"]] == recording.resolve()


def test_outputs_parent_opens_consolidated_catalog(local_server, tmp_path):
    latest = tmp_path / 'long120_1027'
    latest.mkdir()
    run = latest / 'long-original-0-none-pi05'
    run.mkdir()
    write_json(run / 'configuration.json', {'planner': 'none'})
    status, body, _ = request(local_server, '/api/open', json.dumps({'path': str(tmp_path)}))
    data = json.loads(body)
    assert status == 200
    assert data['path'] == str(latest)
    assert len(data['runs']) == 1
    assert data['runs'][0]['path'] == str(run)


def test_catalog_exposes_filters_and_omits_archived_runs(local_server, tmp_path):
    run = tmp_path / "long-original-9-gpt6-gawm"
    run.mkdir()
    write_json(run / "configuration.json", {"task_success": True})
    archived = tmp_path / "_archive" / "runs" / "old-run"
    archived.mkdir(parents=True)
    write_json(archived / "configuration.json", {})
    (tmp_path / "long-original-swap-benchmark-20261023").mkdir()
    status, body, _ = request(local_server, "/api/open", json.dumps({"path": str(tmp_path)}))
    runs = json.loads(body)["runs"]
    assert status == 200 and len(runs) == 1
    assert runs[0]["selection"] == dict(simulation="long-original", experiment="9",
                                      language_model="gpt6", action_model="gawm")
    # A bookmarked experiment still provides the complete picker catalog.
    status, body, _ = request(local_server, "/api/open", json.dumps({"path": str(run)}))
    assert status == 200 and json.loads(body)["runs"] == runs
    assert json.loads(body)["path"] == str(tmp_path)
    # Old bookmarks are resolved to the renamed directory.
    status, body, _ = request(local_server, "/api/open", json.dumps({
        "path": str(tmp_path / "long-original-gpt6-gawm-9")
    }))
    assert status == 200 and json.loads(body)["report"]["path"] == str(run)


@pytest.mark.parametrize(
    "config,orientation",
    [
        ({"suite": "libero_10"}, "libero_calibration"),
        ({"config": {"simulation": {"suite": "libero_object_swap"}}},
         "libero_calibration"),
        ({"suite": "robocasa"}, "saved"),
        ({}, "saved"),
    ],
)
def test_observation_orientation_is_scoped_to_libero(saved_run, config, orientation):
    write_json(saved_run / "configuration.json", config)
    assert parse_run(saved_run)["image_orientation"] == orientation


def test_saved_libero_calibration_is_mirrored_from_policy_when_available():
    outputs = Path(__file__).resolve().parents[1] / "outputs"
    root = outputs / "long120_1027" / "long-original-9-gpt6-gawm"
    if not root.is_dir():
        root = outputs / "long-original-9-gpt6-gawm"
    if not (root / "agentview_policy.png/00.png").is_file():
        pytest.skip("Local robot experiment artifacts are not versioned")
    np = pytest.importorskip("numpy")
    image = pytest.importorskip("PIL.Image")
    with image.open(root / "agentview.png/00.png") as source:
        calibration = np.asarray(source)
    with image.open(root / "agentview_policy.png/00.png") as source:
        policy = np.asarray(source)
    assert np.array_equal(calibration[:, ::-1], policy)
    assert not np.array_equal(calibration, policy)
    report = parse_run(root)
    assert report["image_orientation"] == "libero_calibration"
    assert report["observations"]["0"]["images"] == {
        "agentview": "agentview_high.png/00.png",
        "wrist": "wrist_high.png/00.png",
    }


@pytest.mark.parametrize(
    "task_language,policy_prompt,expected",
    [
        ("recorded task", "policy task", "recorded task"),
        (None, "put the yellow and white mug in the microwave and close it",
         "put the yellow and white mug in the microwave and close it"),
    ],
)
def test_task_description_falls_back_to_configuration(
    saved_run, task_language, policy_prompt, expected
):
    config_path = saved_run / "configuration.json"
    config = json.loads(config_path.read_text())
    config.update(task_language=task_language, policy_prompt=policy_prompt)
    write_json(config_path, config)
    assert parse_run(saved_run)["task"] == expected
    write_json(saved_run / "states.json", {
        "steps": [{"step_idx": 0, "extras": {"task_language": "observed task"}}]
    })
    assert parse_run(saved_run)["task"] == "observed task"


def test_local_http_open_seek_and_file_boundary(local_server, saved_run, tmp_path):
    status, body, _ = request(
        local_server, "/api/open", json.dumps({"path": str(saved_run)})
    )
    assert status == 200
    run = json.loads(body)["report"]["run_id"]
    route = "/api/asset?" + urlencode({"run": run, "path": "episode.mp4"})
    status, body, headers = request(local_server, route, headers={"Range": "bytes=0-7"})
    assert status == 206 and len(body) == 8
    assert headers["Content-Range"].startswith("bytes 0-7/")
    assert request(local_server, route, headers={"Range": "bytes=999999-"})[0] == 416
    external = tmp_path / "external.png"
    external.write_bytes(b"private")
    (saved_run / "escape.png").symlink_to(external)
    for path in ("../external.png", "escape.png", "configuration.json"):
        assert (
            request(
                local_server, "/api/asset?" + urlencode({"run": run, "path": path})
            )[0]
            == 404
        )
    assert (
        request(
            local_server,
            "/api/bootstrap",
            headers={"Origin": "https://external.invalid"},
        )[0]
        == 403
    )
    assert (
        request(local_server, "/api/bootstrap", headers={"Host": "external.invalid"})[0]
        == 403
    )
    assert request(local_server, "/api/open", "[]")[0] == 400
    write_json(saved_run / "configuration.json", [])
    assert (
        request(local_server, "/api/open", json.dumps({"path": str(saved_run)}))[0]
        == 400
    )


@pytest.mark.parametrize(
    "name,elapsed,frames,tools,tokens",
    [
        ("liberopro-gawm-gpt6astra-20261021-162733", 240.8, 447, 32, 1143363),
        ("liberopro-pi05-gpt6astra-20261020-104215", 482.0, 344, 36, 1117616),
    ],
)
def test_existing_successful_records_when_available(
    name, elapsed, frames, tools, tokens
):
    root = Path(__file__).resolve().parents[1] / "outputs" / "_archive" / "runs" / name
    if not root.is_dir():
        pytest.skip("Local robot experiment artifacts are not versioned")
    report = parse_run(root)
    metrics = report["metrics"]
    assert metrics["elapsed_s"] == elapsed
    assert metrics["action_frames"] == frames
    assert metrics["tool_calls"] == tools
    assert (
        metrics["tokens"]["input_tokens"] + metrics["tokens"]["output_tokens"] == tokens
    )
    assert metrics["turns"] == 19
    assert not report["warnings"]
