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
    root = Path(__file__).resolve().parents[1] / "outputs" / name
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
