"""Turn saved RPent/Codex artifacts into a flat, evidence-based timeline."""

from __future__ import annotations

import json
import math
import re
import shlex
import struct
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

TZ = ZoneInfo("Asia/Shanghai")
LOG_TIME = re.compile(r"^(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d)")
TOOL_TYPES = {
    "mcpToolCall",
    "dynamicToolCall",
    "commandExecution",
    "webSearch",
    "fileChange",
}
TOKEN_KEYS = (
    "input_tokens",
    "cached_input_tokens",
    "output_tokens",
    "reasoning_output_tokens",
)


def read_json(path: Path, default=None):
    if not path.is_file():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def read_object(path: Path):
    value = read_json(path, {})
    if not isinstance(value, dict):
        raise TypeError(f"{path.name} 应为 JSON 对象，无法解析该记录。")
    return value


def decode_content(value, depth=0):
    """Unwrap MCP JSON/text envelopes and omit inline binary image payloads."""
    if depth > 16:
        return "内容嵌套过深"
    if isinstance(value, list):
        return [decode_content(part, depth + 1) for part in value]
    if isinstance(value, dict):
        if value.get("type") in {"image", "image_url", "audio"}:
            return {"type": value["type"], "note": "画面从本轮保存的观测文件读取"}
        if value.get("type") == "text" and isinstance(value.get("text"), str):
            try:
                return decode_content(json.loads(value["text"]), depth + 1)
            except (ValueError, TypeError):
                return value["text"]
        return {key: decode_content(part, depth + 1) for key, part in value.items()}
    return value


def find_object(value, key):
    if isinstance(value, dict):
        if key in value:
            return value
        for child in value.values():
            found = find_object(child, key)
            if found is not None:
                return found
    if isinstance(value, list):
        for child in value:
            found = find_object(child, key)
            if found is not None:
                return found
    return None


def public_text(value):
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        return "\n".join(filter(None, (public_text(part) for part in value)))
    if isinstance(value, dict):
        return public_text(
            value.get("text", value.get("summary", value.get("content", "")))
        )
    return ""


def video_metadata(path: Path):
    """Read MP4 track duration and sample count without decoding or dependencies."""
    if not path.is_file():
        return None

    def atoms(stream, start, end):
        while start + 8 <= end:
            stream.seek(start)
            header = stream.read(8)
            if len(header) != 8:
                return
            size, kind = struct.unpack(">I4s", header)
            head = 8
            if size == 1:
                extended = stream.read(8)
                if len(extended) != 8:
                    return
                size, head = struct.unpack(">Q", extended)[0], 16
            elif size == 0:
                size = end - start
            if size < head or start + size > end:
                return
            yield kind, start + head, start + size
            start += size

    def walk(stream, start, end):
        for kind, payload, limit in atoms(stream, start, end):
            yield kind, payload, limit
            if kind in {b"mdia", b"minf", b"stbl"}:
                yield from walk(stream, payload, limit)

    with path.open("rb") as stream:
        for kind, payload, end in atoms(stream, 0, path.stat().st_size):
            if kind != b"moov":
                continue
            for child, track_start, track_end in atoms(stream, payload, end):
                if child != b"trak":
                    continue
                is_video, duration, count = False, None, None
                for atom, data, limit in walk(stream, track_start, track_end):
                    stream.seek(data)
                    body = stream.read(min(limit - data, 40))
                    if atom == b"hdlr" and len(body) >= 12:
                        is_video = body[8:12] == b"vide"
                    elif atom == b"mdhd" and len(body) >= 24:
                        if body[0] == 1 and len(body) >= 32:
                            scale = struct.unpack_from(">I", body, 20)[0]
                            ticks = struct.unpack_from(">Q", body, 24)[0]
                        else:
                            scale, ticks = struct.unpack_from(">II", body, 12)
                        duration = ticks / scale if scale else None
                    elif atom == b"stsz" and len(body) >= 12:
                        count = struct.unpack_from(">I", body, 8)[0]
                if is_video and duration and count is not None:
                    return {
                        "path": "episode.mp4",
                        "duration_s": duration,
                        "frames": count,
                        "fps": count / duration,
                    }
    return None


def state_images(root: Path, state):
    result = {}
    index = state["step_idx"]
    for camera in ("agentview", "wrist"):
        for name in (
            f"{camera}_high.png/{index:02d}.png",
            f"{camera}.png/{index:02d}.png",
        ):
            path = root / name
            if path.is_file() and path.resolve().is_relative_to(root):
                result[camera] = name
                break
    return result


def parse_run(directory: Path):
    root = directory.expanduser().resolve()
    config = read_object(root / "configuration.json")
    transcripts = sorted(root.glob("transcript_*.json"))
    transcript = read_object(transcripts[0]) if transcripts else {}
    streams = sorted(root.glob("*.stream.jsonl"))
    if not config and not transcript and not streams:
        raise ValueError(
            "该目录没有可识别的实验记录，请选择具体实验目录或它的 outputs 上级目录。"
        )
    warnings, events, turns = [], [], {}
    states = read_object(root / "states.json").get("steps", [])
    state_map = {state["step_idx"]: state for state in states}
    initial = states[0]["step_idx"] if states else None
    latest_state = initial
    observations = {
        str(index): {
            "index": index,
            "images": state_images(root, state),
            "terminated": state.get("terminated"),
            "truncated": state.get("truncated"),
        }
        for index, state in state_map.items()
    }
    unused_states = [state for state in states if state.get("command")]
    video = video_metadata(root / "episode.mp4")
    frame_cursor = 0
    backend = config.get("policy_backend", "openpi")
    policy = config.get("config", {}).get(backend, {})
    horizon = policy.get("execute_horizon", 5 if backend == "openpi" else None)
    last_public = ""

    def add(title, at=None, end=None, turn=0, **extra):
        event = {
            "id": f"event-{len(events)}",
            "title": title,
            "at_ms": at,
            "end_ms": end if end is not None else at,
            "turn": turn,
            **extra,
        }
        events.append(event)
        return event

    def link_tool(event, name, arguments, result):
        nonlocal latest_state, frame_cursor
        event.update(before_state=latest_state, after_state=latest_state)
        returned = find_object(result, "step")
        returned_index = returned.get("step") if returned else None
        matched = None
        for candidate in unused_states:
            command = candidate["command"]
            if command.get("action") != name:
                continue
            same_args = all(
                command.get(key) == value for key, value in arguments.items()
            )
            if returned_index == candidate["step_idx"] or same_args:
                matched = candidate
                break
        if matched is not None:
            unused_states.remove(matched)
            latest_state = matched["step_idx"]
            event.update(
                after_state=latest_state,
                action=True,
                execution_s=matched.get("elapsed_s"),
                state_result=matched.get("result"),
                native_success=matched.get("terminated"),
                truncated=matched.get("truncated"),
            )
            outcome = matched.get("result") or {}
            count = None
            if name == "move_to" and isinstance(
                outcome.get("steps_used"), (int, float)
            ):
                # RPent counts the final arrival check as a sample, not an action.
                count = max(0, outcome["steps_used"] - 1)
            elif name in {"pi0_pick", "pi0_doubled"} and horizon:
                count = outcome.get("chunks_used", 0) * horizon
            elif name == "set_gripper":
                count = outcome.get("steps")
            elif name in {"release", "rotate_wrist", "rotate_pitch", "move_pose"}:
                count = outcome.get("steps_used")
            elif name == "execute_action":
                count = 1
            if count is not None and frame_cursor >= 0 and video and video["fps"]:
                event["video_range"] = {
                    "start_s": frame_cursor / video["fps"],
                    "end_s": (frame_cursor + count) / video["fps"],
                    "frames": count,
                }
                frame_cursor += count
            elif video:
                frame_cursor = -1
        elif returned_index in state_map and name == "view_env_state":
            event["reference_state"] = returned_index
        if (
            matched is None
            and event.get("status") == "failed"
            and any(
                state["command"].get("action") == name
                for state in states
                if state.get("command")
            )
        ):
            event["after_state"] = None

    log_path = root / "run.log"
    log_start = log_end = None
    if log_path.is_file():
        for number, line in enumerate(
            log_path.read_text(encoding="utf-8", errors="replace").splitlines(), 1
        ):
            match = LOG_TIME.match(line)
            if not match:
                continue
            stamp = (
                datetime.strptime(match[1], "%Y-%m-%d %H:%M:%S")
                .replace(tzinfo=TZ)
                .timestamp()
                * 1000
            )
            log_start = log_start if log_start is not None else stamp
            log_end = stamp
            label = None
            if "physical agent cmd:" in line:
                label = "启动实验"
            elif "spawned (pid=" in line:
                label = "启动仿真环境"
            elif "HttpMcpServer ready at" in line:
                label = "工具服务就绪"
            elif "invoking Codex SDK model" in line:
                label = "连接 Codex CLI"
            if label:
                add(
                    label,
                    stamp,
                    precision="second",
                    text=line[20:],
                    arguments={
                        "command": config.get("command"),
                        "model": config.get("model"),
                        "reasoning_effort": config.get("reasoning_effort"),
                    }
                    if label == "启动实验"
                    else None,
                    before_state=initial,
                    after_state=initial,
                    source={"file": "run.log", "line": number},
                )
    if not events:
        add(
            "启动配置",
            arguments={
                "command": config.get("command"),
                "model": config.get("model"),
                "config": config.get("config"),
            },
            before_state=initial,
            after_state=initial,
            source={"file": "configuration.json"},
        )

    current_turn, latest_clock = 1, log_start
    usage_total = {key: 0 for key in TOKEN_KEYS}
    seen_usage, starts, completed = set(), {}, set()

    if streams:
        for line_number, line in enumerate(streams[0].open(encoding="utf-8"), 1):
            try:
                raw = json.loads(line)
            except json.JSONDecodeError:
                warnings.append(f"原始事件第 {line_number} 行不完整，已跳过。")
                continue
            method = raw.get("method", raw.get("type", ""))
            payload = raw.get("payload") or {}
            item = payload.get("item") or {}
            item_id, kind = item.get("id"), item.get("type")
            if method == "item/started":
                starts[item_id] = (
                    payload.get("started_at_ms"),
                    item,
                    line_number,
                    current_turn,
                )
            if method == "thread/tokenUsage/updated":
                usage = payload.get("token_usage") or {}
                total = usage.get("total", usage)
                values = {key: int(total.get(key) or 0) for key in TOKEN_KEYS}
                signature = tuple(values.values())
                if not any(signature) or signature in seen_usage:
                    continue
                seen_usage.add(signature)
                delta = {
                    key: max(0, values[key] - usage_total[key]) for key in TOKEN_KEYS
                }
                turn_info = turns.setdefault(current_turn, {})
                turn_info.update(usage=delta)
                add(
                    f"Turn {current_turn} 用量",
                    latest_clock,
                    turn=current_turn,
                    usage=delta,
                    precision="adjacent",
                    source={"file": streams[0].name, "line": line_number},
                )
                usage_total = values
                current_turn += 1
                continue
            if method == "turn/completed":
                turn = payload.get("turn") or {}
                at = turn.get("completed_at")
                add(
                    "Codex 运行结束",
                    max(at * 1000, latest_clock or 0) if at else latest_clock,
                    turn=0,
                    precision="second",
                    status=turn.get("status"),
                    result={
                        "duration_ms": turn.get("duration_ms"),
                        "status": turn.get("status"),
                    },
                    before_state=latest_state,
                    after_state=latest_state,
                    source={"file": streams[0].name, "line": line_number},
                )
                continue
            if method in {"error", "fatal", "timeout"}:
                add(
                    "运行错误",
                    latest_clock,
                    turn=current_turn,
                    status="failed",
                    result=decode_content(payload or raw),
                    precision="adjacent",
                    source={"file": streams[0].name, "line": line_number},
                )
            if method != "item/completed" or item_id in completed:
                continue
            completed.add(item_id)
            start, _, start_line, item_turn = starts.get(
                item_id, (None, None, line_number, current_turn)
            )
            end = payload.get("completed_at_ms")
            latest_clock = max(
                filter(lambda value: value is not None, (latest_clock, end)),
                default=None,
            )
            source = {
                "file": streams[0].name,
                "line": line_number,
                "started_line": start_line,
            }
            common = {
                "turn": item_turn,
                "before_state": latest_state,
                "after_state": latest_state,
                "source": source,
            }
            if kind == "userMessage":
                add(
                    "提交初始提示词",
                    start or end,
                    end,
                    turn=0,
                    text=public_text(item.get("content")),
                    before_state=initial,
                    after_state=initial,
                    source=source,
                )
            elif kind == "agentMessage":
                text = item.get("text") or ""
                if text.strip():
                    last_public = text
                    add("Codex 公开文字", end, text=text, **common)
            elif kind == "reasoning":
                text = public_text(item.get("summary") or item.get("content"))
                if text.strip():
                    add("Codex 推理摘要", end, text=text, **common)
            elif kind in TOOL_TYPES:
                name = item.get("tool", kind)
                if kind == "commandExecution":
                    name = "本地命令"
                    arguments = {"command": item.get("command"), "cwd": item.get("cwd")}
                    result = {
                        "exit_code": item.get("exit_code"),
                        "output": item.get("aggregated_output"),
                    }
                else:
                    arguments = item.get("arguments") or {}
                    result = decode_content(
                        item.get("result", item.get("changes", item))
                    )
                if not isinstance(arguments, dict):
                    arguments = {"input": arguments}
                event = add(
                    name,
                    start or end,
                    end,
                    tool=True,
                    status=item.get("status"),
                    duration_ms=item.get(
                        "duration_ms",
                        (end - start)
                        if end is not None and start is not None
                        else None,
                    ),
                    arguments=arguments,
                    result=result,
                    public_before=last_public,
                    **common,
                )
                link_tool(event, name, arguments, result)
            elif kind not in {None, "plan", "hookPrompt"}:
                add(str(kind), start or end, end, result=decode_content(item), **common)
        for item_id, (start, item, number, turn) in starts.items():
            if item_id not in completed and item.get("type") in TOOL_TYPES:
                add(
                    item.get("tool", item["type"]),
                    start,
                    turn=turn,
                    tool=True,
                    status="inProgress",
                    arguments=item.get("arguments"),
                    before_state=latest_state,
                    after_state=None,
                    source={"file": streams[0].name, "line": number},
                )
    elif transcript:
        warnings.append("没有 Codex 原始事件，按保存的消息顺序展示；未记录的时间留空。")
        saved_calls = {}
        for message in transcript.get("messages", []):
            if message.get("role") == "user":
                add(
                    "保存的用户输入",
                    turn=0,
                    text=public_text(message.get("content")),
                    before_state=initial,
                    after_state=initial,
                )
            elif message.get("role") == "assistant":
                content = message.get("content")
                if isinstance(content, list):
                    for block in content:
                        if isinstance(block, dict) and block.get("type") == "tool_use":
                            saved_calls[block.get("id")] = block.get("input", {})
                for call in message.get("tool_calls", []):
                    function = call.get("function", {})
                    arguments = function.get("arguments", {})
                    if isinstance(arguments, str):
                        try:
                            arguments = json.loads(arguments)
                        except ValueError:
                            arguments = {"input": arguments}
                    saved_calls[call.get("id")] = arguments
                text = public_text(message.get("content"))
                if text:
                    add(
                        "规划器公开文字",
                        turn=current_turn,
                        text=text,
                        before_state=latest_state,
                        after_state=latest_state,
                    )
            elif message.get("role") == "tool":
                name = message.get("name", "工具")
                result = decode_content(
                    {"type": "text", "text": message.get("content", "")}
                )
                command = next(
                    (
                        state["command"]
                        for state in unused_states
                        if state["command"].get("action") == name
                    ),
                    {},
                )
                arguments = saved_calls.get(message.get("tool_call_id")) or {
                    key: value for key, value in command.items() if key != "action"
                }
                event = add(
                    name,
                    turn=current_turn,
                    tool=True,
                    arguments=arguments,
                    result=result,
                    status="completed",
                    before_state=latest_state,
                    after_state=latest_state,
                )
                link_tool(event, name, arguments, result)
                current_turn += 1

    # A raw log's response counter is authoritative for reply boundaries; empty
    # reasoning items are not presented as reconstructed private thoughts.
    for event in events:
        if event["turn"]:
            turns.setdefault(event["turn"], {}).setdefault("events", []).append(
                event["id"]
            )
    native_stats = transcript.get("stats", config.get("stats", {})) or {}
    token_stats = {
        key: int(native_stats.get("total_" + key, usage_total[key]) or 0)
        for key in TOKEN_KEYS
    }
    for key in ("input_tokens", "output_tokens"):
        if usage_total[key] and token_stats[key] != usage_total[key]:
            warnings.append(f"汇总与事件累计 {key} 不一致，页头采用汇总记录。")
    timed = [event["at_ms"] for event in events if event["at_ms"] is not None]
    zero = log_start if log_start is not None else min(timed, default=None)
    for sequence, event in enumerate(events):
        event["sequence"] = sequence
        event["offset_s"] = (
            (event["at_ms"] - zero) / 1000
            if event["at_ms"] is not None and zero is not None
            else None
        )
    # Source order resolves ties. Unknown-time messages retain their recorded
    # order instead of inventing timestamps from adjacent action durations.
    events.sort(
        key=lambda event: (
            event["at_ms"]
            if event["at_ms"] is not None
            else (-math.inf if event["title"] == "启动配置" else math.inf),
            event["sequence"],
        )
    )
    if video and frame_cursor != video["frames"]:
        warnings.append(
            "逐工具动作数与完整视频帧数不一致，保留完整视频，关闭可能错位的动作片段。"
        )
        for event in events:
            event.pop("video_range", None)
    if unused_states:
        warnings.append(
            f"{len(unused_states)} 份动作观测未匹配到完整工具事件，未猜测对应关系。"
        )
    action_s = sum(
        float(state.get("elapsed_s") or 0) for state in states if state.get("command")
    )
    overall = transcript.get("elapsed_s")
    if overall is None:
        overall = native_stats.get("elapsed_s")
    status = (
        "success"
        if states and states[-1].get("terminated")
        else (
            "running"
            if config.get("run_status") in {"starting", "running"}
            else "finished"
        )
    )
    command = config.get("command")
    return {
        "name": root.name,
        "path": str(root),
        "task": states[0].get("extras", {}).get("task_language")
        if states
        else config.get("task_language"),
        "model": config.get("model", transcript.get("model", config.get("planner"))),
        "backend": backend,
        "status": status,
        "metrics": {
            "elapsed_s": overall,
            "wall_elapsed_s": (log_end - zero) / 1000
            if log_end is not None and zero is not None
            else None,
            "action_execution_s": action_s if states else None,
            "action_video_s": video.get("duration_s") if video else None,
            "action_frames": video.get("frames") if video else None,
            "tool_calls": sum(bool(event.get("tool")) for event in events),
            "recorded_tool_calls": native_stats.get("tool_calls"),
            "turns": max(turns, default=0),
            "tokens": token_stats,
        },
        "command": shlex.join(command) if isinstance(command, list) else command,
        "events": events,
        "turns": [{"number": number, **turn} for number, turn in sorted(turns.items())],
        "observations": observations,
        "video": video,
        "warnings": warnings,
        "notes": [
            "Turn 按累计 token 用量更新划分，与 RPent 的回复计数一致。",
            "仅显示记录中公开的模型文字及实际返回的推理摘要；空 reasoning 事件不补写内容。",
            "动作时间为仿真视频时长；工具实耗来自保存的 elapsed_s。",
        ],
    }
