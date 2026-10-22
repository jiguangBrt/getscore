"""Build an offline, evidence-linked report for a recorded Codex LIBERO run."""

from __future__ import annotations

import argparse
import ast
import base64
import hashlib
import json
import os
import re
from collections import Counter
from datetime import datetime
from pathlib import Path
from urllib.parse import quote
from zoneinfo import ZoneInfo

import imageio.v2 as imageio

TZ = ZoneInfo("Asia/Shanghai")
ROOT = Path(__file__).resolve().parents[1]


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def unpack(value):
    """Decode nested MCP text envelopes and omit inline image bytes in the report."""
    if isinstance(value, list):
        return [unpack(item) for item in value]
    if isinstance(value, dict):
        if value.get("type") in {"image", "image_url"}:
            return {
                "type": value["type"],
                "note": "图像数据见原始事件；页面另展示已存档的相机观测。",
            }
        if value.get("type") == "text":
            text = value.get("text", "")
            try:
                return unpack(json.loads(text))
            except (ValueError, TypeError):
                return {"text": text}
        return {key: unpack(item) for key, item in value.items()}
    return value


def build(output):
    output = output.resolve()
    review = ROOT / "review" / output.name
    review.mkdir(parents=True, exist_ok=True)
    [stream] = output.glob("codex_*.txt.stream.jsonl")
    [transcript_path] = output.glob("transcript_*.json")
    transcript = read_json(transcript_path)
    config = read_json(output / "configuration.json")
    command = config["command"]
    recorded_output = Path(command[command.index("--output-dir") + 1])
    states = read_json(output / "states.json")["steps"]
    network = [
        json.loads(line)
        for line in (output / "network-timing.jsonl").read_text().splitlines()
    ]
    backend = config.get("policy_backend", "openpi")
    policy_label = {"gawm": "GAWM", "openpi": "π0.5"}[backend]
    policy_config = config["config"][backend]
    bridge_config = config["config"]["bridge"]
    execute_horizon = policy_config.get("execute_horizon", 5)
    actions = [step for step in states if step.get("command")]
    policy_calls = [step for step in actions if step["command"]["action"].startswith("pi0_")]
    action_counts = Counter(step["command"]["action"] for step in actions)
    frame_count = 0
    with imageio.get_reader(output / "episode.mp4") as reader:
        video_meta = reader.get_meta_data()
        for _frame in reader:
            frame_count += 1
    fps = video_meta["fps"]
    video_seconds = frame_count / fps
    bridge_status = None
    for line in (output / "bridge.log").read_text().splitlines():
        try:
            value = ast.literal_eval(line)
        except (ValueError, SyntaxError):
            continue
        if isinstance(value, dict) and "source" in value:
            bridge_status = value
            break
    if backend == "gawm":
        assert bridge_status["source"] == "starvla-gawm-libero"
        assert bridge_status["num_action_chunks"] == execute_horizon == 8
    native_success = bool(states[-1]["terminated"])
    source_hashes = {
        path.name: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in (
            stream,
            transcript_path,
            output / "states.json",
            output / "configuration.json",
            output / "network-timing.jsonl",
            output / "bridge.log",
            output / "episode.mp4",
        )
    }

    def source(path, field="", line=None):
        path = Path(path)
        if not path.is_absolute():
            path = output / path
        elif path.is_relative_to(recorded_output):
            path = output / path.relative_to(recorded_output)
        return {
            "file": path.name,
            "path": str(path),
            "url": quote(os.path.relpath(path, review), safe="/"),
            "field": field,
            "line": line,
        }

    def stamp(ms):
        return datetime.fromtimestamp(ms / 1000, TZ).strftime("%H:%M:%S.%f")[:-3]

    log_lines = (output / "run.log").read_text().splitlines()
    log_start = datetime.strptime(log_lines[0][:19], "%Y-%m-%d %H:%M:%S").replace(
        tzinfo=TZ
    )
    zero = log_start.timestamp() * 1000
    events = []

    def add(kind, title, at, *, end=None, **kwargs):
        events.append(
            {
                "id": f"e{len(events)}",
                "kind": kind,
                "title": title,
                "at": at,
                "end": end if end is not None else at,
                "clock": stamp(at),
                "endClock": stamp(end) if end is not None else stamp(at),
                "relative": round((at - zero) / 1000, 3),
                **kwargs,
            }
        )

    add(
        "setup",
        "启动本地 RPent 与仿真",
        zero,
        important=True,
        explanation=f"启动器先启动 {policy_label} 适配层，再启动 RPent。LIBERO-PRO 在本机创建环境并渲染；{policy_label} 已在远程服务中运行。此时间来自秒级日志。",
        arguments={
            "rpent_command": config["command"],
            "model": config["model"],
            "reasoning_effort": config["reasoning_effort"],
            "mode": config["mode"],
        },
        sources=[
            source("configuration.json", "command / limits"),
            source("run.log", line=1),
        ],
    )
    for number, line in enumerate(log_lines, 1):
        if "env_server spawned" in line:
            title = "本地仿真环境进程启动"
            explanation = "RPent 创建 LIBERO-PRO 环境子进程；任务与初始化数据来源见 env_server.log。日志中的 spawn 时间不等于渲染完成时间。"
        elif "HttpMcpServer ready" in line:
            title = "工具服务就绪，准备启动 Codex"
            explanation = "RPent 暴露本地 MCP 接口。随后同一秒的日志记录了初始提示词字符数和 Codex SDK 调用；精确的文本提交见下一条 userMessage。"
        elif "I [agent] elapsed:" in line:
            title = "运行结束并汇总归档"
            explanation = f"RPent 保存汇总：elapsed_s={transcript['elapsed_s']}，环境最终 terminated={str(native_success).lower()}、truncated={str(states[-1]['truncated']).lower()}。启动日志区间包含前置启动时间，不能直接等同于该 elapsed_s。"
        else:
            continue
        at = (
            datetime.strptime(line[:19], "%Y-%m-%d %H:%M:%S")
            .replace(tzinfo=TZ)
            .timestamp()
            * 1000
        )
        add(
            "finish" if "汇总" in title else "setup",
            title,
            at,
            important=True,
            explanation=explanation,
            sources=[source("run.log", line=number)],
        )
    starts = {}
    prompt = None
    prompt_line = None
    last_public = ""
    latest_state = 0
    action_index = 0
    frame_cursor = 0
    type_counts = Counter()
    reasoning_items = []
    for line_number, line in enumerate(stream.open(encoding="utf-8"), 1):
        raw = json.loads(line)
        payload = raw.get("payload", {})
        item = payload.get("item", {})
        method = raw.get("method")
        if method == "item/started":
            starts[item["id"]] = {
                "at": payload.get("started_at_ms"),
                "line": line_number,
            }
        if method != "item/completed":
            continue
        typ = item.get("type")
        type_counts[typ] += 1
        end = payload["completed_at_ms"]
        start = starts.get(item["id"], {}).get("at") or end
        evidence = [
            source(stream, f"payload.item ({typ}); item.id={item['id']}", line_number)
        ]
        if typ == "userMessage":
            prompt = "".join(part.get("text", "") for part in item["content"])
            prompt_line = line_number
            add(
                "prompt",
                "RPent 提交初始完整文本",
                start,
                end=end,
                important=True,
                explanation="RPent 将 system_prompt 与 user_message 以两个换行拼接，再调用 thread.turn(prompt)。这个 userMessage 保存了当时提交的原文，不是用现有模板重新生成。",
                sources=evidence
                + [
                    source(
                        ROOT / "RPent/rpent/planner/codex.py",
                        "solve() → thread.turn(prompt)",
                    )
                ],
            )
        elif typ == "agentMessage":
            text = item.get("text", "")
            last_public = text
            add(
                "model",
                "GPT 公开输出",
                end,
                message=text,
                important=False,
                explanation="模型公开的计划、观察或纠错文字；不是完整内部推理。时间是该消息的完成事件时间。",
                sources=evidence,
                state=latest_state,
            )
        elif typ == "reasoning":
            reasoning_items.append(
                {
                    "summary_count": len(item.get("summary") or []),
                    "content_count": len(item.get("content") or []),
                    "line": line_number,
                }
            )
        elif typ == "commandExecution":
            add(
                "tool",
                "commandExecution",
                start,
                end=end,
                arguments={"command": item.get("command")},
                result={"output": item.get("aggregated_output"), "exit_code": item.get("exit_code")},
                status=item.get("status"),
                publicBefore=last_public,
                explanation="Codex CLI 的本地命令工具调用，原始命令和输出见本事件；它不属于机器人动作。",
                sources=evidence,
            )
        elif typ == "mcpToolCall":
            name = item["tool"]
            arguments = item.get("arguments", {})
            kind = "tool"
            fields = {
                "arguments": arguments,
                "result": unpack(item.get("result")),
                "publicBefore": last_public,
                "status": item.get("status"),
                "durationMs": item.get("duration_ms"),
                "sources": evidence,
                "important": name == "finish",
            }
            if name in {"move_to", "pi0_pick", "pi0_doubled", "set_gripper", "release"}:
                action_index += 1
                step = states[action_index]
                assert step["command"]["action"] == name
                assert arguments == {
                    key: value
                    for key, value in step["command"].items()
                    if key != "action"
                }, (name, arguments, step["command"])
                kind = "vla" if name.startswith("pi0_") else "action"
                fields.update(
                    state=step["step_idx"],
                    beforeState=latest_state,
                    result={
                        key: step.get(key)
                        for key in ("result", "terminated", "truncated", "elapsed_s")
                    },
                    explanation=(
                        f"工具名 {name} 沿用 RPent 接口，实际调用 {policy_label}，执行 {step['result'].get('chunks_used', 0)} 个策略动作块。局部 success 是抓取启发式判据，需与图像及原生任务标志分别核对。"
                        if kind == "vla"
                        else f"GPT 给出参数，RPent 脚本执行 {name}；该动作不请求 {policy_label}。对齐的执行前后图像和原始返回见下方。"
                    ),
                    important=True,
                )
                fields["sources"].append(
                    source(
                        "states.json",
                        f"steps[{action_index}].command / result / state / extras",
                    )
                )
                fields["sources"].append(
                    source("object_swap_t2_s0_recipe.jsonl", "动作参数", action_index)
                )
                latest_state = action_index
                if name == "move_to":
                    actual_steps = step["result"]["steps_used"] - 1
                elif name in {"pi0_pick", "pi0_doubled"}:
                    actual_steps = step["result"]["chunks_used"] * execute_horizon
                elif name == "set_gripper":
                    actual_steps = step["result"]["steps"]
                else:
                    actual_steps = step["result"]["steps_used"]
                fields["video"] = {
                    "start": frame_cursor / fps,
                    "end": (frame_cursor + actual_steps) / fps,
                    "startFrame": frame_cursor,
                    "endFrameExclusive": frame_cursor + actual_steps,
                    "actions": actual_steps,
                }
                frame_cursor += actual_steps
                linked = [
                    index + 1
                    for index, entry in enumerate(network)
                    if start <= entry["time_unix"] * 1000 <= end
                ]
                fields["networkRows"] = linked
                if kind == "vla":
                    assert len(linked) == step["result"]["chunks_used"]
            elif name == "view_env_state":
                requested = arguments.get("step", -1)
                fields["state"] = latest_state if requested == -1 else requested
                fields["explanation"] = (
                    "读取已记录的状态及主相机/腕部图像并返回给 GPT；不执行机器人动作。"
                )
            elif name == "read_text_file":
                fields["explanation"] = (
                    "读取指南或记忆，为后续规划补充上下文。实际返回内容在本事件的 result 中；缺少记忆文件会返回错误，但不等于实验失败。"
                )
                path = Path(arguments.get("path", ""))
                if not path.is_absolute():
                    path = ROOT / "RPent" / path
                fields["sources"].append(
                    source(path, "请求读取的文件；历史返回原文以事件 result 为准")
                )
            elif name == "back_project":
                fields["explanation"] = (
                    "将 GPT 选中的图像像素结合已保存的深度与相机标定转换为世界坐标；是几何计算，不是 VLA 动作。"
                )
            elif name == "finish":
                kind = "finish"
                fields["explanation"] = (
                    "GPT 声明结束并停止工具循环。status=success 是模型声明；本轮另有环境 terminated=true 的独立证据。"
                )
            else:
                fields["explanation"] = "RPent 工具执行的实际调用及返回。"
            add(
                kind,
                f"{name}"
                + (f" · 观测 {latest_state:02d}" if kind in {"action", "vla"} else ""),
                start,
                end=end,
                **fields,
            )

    assert prompt is not None
    assert action_index == len(actions) and frame_cursor == frame_count
    if backend == "gawm":
        assert "GAWM policy for this run" in prompt
    (review / "initial-prompt.txt").write_text(prompt, encoding="utf-8")
    for index, entry in enumerate(network, 1):
        at = entry["time_unix"] * 1000
        parent = next(
            event for event in events if index in event.get("networkRows", [])
        )
        add(
            "network",
            f"{policy_label} 请求 #{index:02d}",
            at,
            end=at + entry["wall_ms"],
            important=False,
            parent=parent["id"],
            state=parent["beforeState"],
            result=entry,
            explanation=f"属于 {parent['title']}。本条时间由本地适配层在请求开始时记录；耗时是本地测得的完整等待时间。网络差值估计含其他开销，不是精确单向延迟。",
            sources=[source("network-timing.jsonl", f"第 {index} 次请求", index)],
        )
    events.sort(key=lambda event: (event["at"], event["end"]))
    images = {}
    for state in states:
        i = state["step_idx"]
        images[str(i)] = {}
        for camera in ("agentview", "wrist"):
            file = output / f"{camera}.png/{i:02d}.png"
            images[str(i)][camera] = (
                "data:image/png;base64," + base64.b64encode(file.read_bytes()).decode()
            )
    prompt_meta = {
        "chars": len(prompt),
        "sha256": hashlib.sha256(prompt.encode()).hexdigest(),
        "source": source(
            stream,
            "payload.item.content[0].text；第一条完成的 userMessage",
            prompt_line,
        ),
        "export": source(review / "initial-prompt.txt", "从历史事件逐字提取，未重建"),
    }
    files = [
        (
            "initial-prompt.txt",
            "初始完整输入",
            f"本次从原始 userMessage 导出的 {len(prompt):,} 字符 RPent 提交文本。",
        ),
        (
            stream.name,
            "原始事件",
            "初始输入、公开输出、工具参数/返回、毫秒时间、token；逐行 JSON，部分返回含编码图像。",
        ),
        (
            "states.json",
            "环境证据",
            f"{len(states)} 份观测；每次动作的参数、结果、原始任务、终止标志；观测号不等于底层动作步数。",
        ),
        (
            "episode.mp4",
            "执行视频",
            f"{frame_count} 个实际动作的连续画面，{fps:g}fps，{video_seconds:g} 秒；没有包含模型等待时间。",
        ),
        (
            "codex_object_swap_t2_s0.txt",
            "公开文字",
            "RPent 实时整理 Codex SDK 事件写出，工具结果仅为摘要。",
        ),
        (
            "codex_object_swap_t2_s0.txt.last",
            "最后回复",
            "仅保留最后一次公开文字，不包含整个任务过程。",
        ),
        (
            transcript_path.name,
            "汇总结果",
            "结果、用量和模型文字；字符串中的换行被转义。environment_success=null 不应替代 states 中的终止证据。",
        ),
        (
            transcript_path.with_suffix(".md").name,
            "可读汇总",
            "此前补生成的按 turn 分段记录，含实际动作参数及返回值。",
        ),
        (
            "configuration.json",
            "启动与预算",
            "精确 RPent 命令、模型、low、有效 limits、run_status、task_success 与累计用量。",
        ),
        (
            "network-timing.jsonl",
            "网络与推理",
            f"{len(network)} 次 {policy_label} 请求开始时间及各项耗时；通过同机时间戳与 VLA 工具区间对齐。",
        ),
        (
            "object_swap_t2_s0.json",
            "模型事后分析",
            "GPT 主动写出的定位、策略、失败与恢复分析；应与原始状态交叉核对。",
        ),
        (
            "object_swap_t2_s0_recipe.jsonl",
            "动作清单",
            f"{len(actions)} 条实际动作的工具名与参数，不含结果；便于核对脚本控制与 VLA 的边界。",
        ),
        (
            "run.log",
            "带时间日志",
            f"启动、MCP 地址、{len(prompt):,} 字符计数、公开文字与工具摘要；时间精确到秒。",
        ),
        ("console.log", "控制台", "命令输出与警告；与 run.log 部分重复。"),
        (
            "env_server.log",
            "仿真启动",
            "LIBERO-PRO 配置、任务语言、原生目标条件、环境 RPC 地址与启动警告。",
        ),
        (
            "bridge.log",
            "适配层日志",
            f"{policy_label} 桥接服务启动、后端标识及模型元数据；动作请求时间看 network-timing。",
        ),
        (
            "agentview_high.png",
            "主相机原图目录",
            f"00–{states[-1]['step_idx']:02d}：初始场景及工具执行后高分辨率画面；同号对应 states.steps 的 step_idx。",
        ),
        (
            "wrist_high.png",
            "腕部原图目录",
            "同号腕部近景；可核对篮子内部和夹爪附近目标。",
        ),
        (
            "agentview_policy.png",
            "策略视角目录",
            "核对送往策略的图像方向；本页缩略图采用标准主相机/腕部图像。",
        ),
        (
            "agentview_depth.npz",
            "深度数组目录",
            "逐像素深度；腕部对应 wrist_depth.npz。",
        ),
        (
            "agentview_world_high.npz",
            "世界坐标目录",
            "逐像素世界坐标，供 back_project；另有标准分辨率和腕部版本。",
        ),
        (
            "agentview_metadata.json",
            "相机参数目录",
            "内外参、分辨率等；腕部对应 wrist_metadata.json。",
        ),
        (
            "task-flow-data.json",
            "本页整理数据",
            "从历史文件提取的事件、索引及来源，不是另一次实验结果。",
        ),
    ]
    from codex_cli_bin import bundled_codex_path

    endpoint_match = re.search(r"HttpMcpServer ready at (\S+)", "\n".join(log_lines))
    policy_action_count = sum(step["result"]["chunks_used"] * execute_horizon for step in policy_calls)
    presentation = {
        "policy": policy_label,
        "backend": backend,
        "executeHorizon": execute_horizon,
        "actionCalls": len(actions),
        "policyCalls": len(policy_calls),
        "networkRequests": len(network),
        "frames": frame_count,
        "fps": fps,
        "videoSeconds": video_seconds,
        "elapsedSeconds": transcript["elapsed_s"],
        "policyActions": policy_action_count,
        "scriptActions": frame_count - policy_action_count,
        "actionCounts": dict(action_counts),
        "nativeSuccess": native_success,
        "truncated": bool(states[-1]["truncated"]),
        "finalState": states[-1]["step_idx"],
        "mcpEndpoint": endpoint_match[1] if endpoint_match else "未单独记录",
        "bridgeEndpoint": f"http://{bridge_config['host']}:{bridge_config['port']}",
        "policyEndpoint": f"ws://{policy_config['host']}:{policy_config['port']}",
        "date": log_start.strftime("%Y.%m.%d"),
        "bridgeStatus": {
            **(bridge_status or {}),
            "metadata": {k: v for k, v in (bridge_status or {}).get("metadata", {}).items() if k != "task_vocabulary"},
        },
    }
    data = {
        "title": f"沙拉酱入篮 · GPT + {policy_label} 数据流",
        "presentation": presentation,
        "run": output.name,
        "path": str(output),
        "reviewPath": str(review),
        "assetBase": quote(os.path.relpath(output, review), safe="/") + "/",
        "task": states[0]["extras"]["task_language"],
        "config": config,
        "stats": transcript["stats"],
        "states": states,
        "events": events,
        "images": images,
        "prompt": prompt,
        "promptMeta": prompt_meta,
        "typeCounts": dict(type_counts),
        "reasoningItems": reasoning_items,
        "video": "data:video/mp4;base64,"
        + base64.b64encode((output / "episode.mp4").read_bytes()).decode(),
        "files": [
            {
                **source(
                    review / name
                    if name in {"initial-prompt.txt", "task-flow-data.json"}
                    or name.endswith(".md")
                    else name
                ),
                "role": role,
                "description": desc,
            }
            for name, role, desc in files
        ],
        "cliBin": str(bundled_codex_path()),
        "launchSources": [
            source(ROOT / "scripts/run-liberopro.sh"),
            source(ROOT / "scripts/run_libero_codex.py"),
            source(ROOT / f"scripts/{backend}_rpent_bridge.py"),
            source(ROOT / "RPent/rpent/planner/codex.py"),
            source(
                ROOT / ".venv/lib/python3.11/site-packages/openai_codex/client.py",
                "CodexClient.start(): app-server --listen stdio://",
                250,
            ),
        ],
        "sourceHashes": source_hashes,
        "notes": [
            "时间统一为 Asia/Shanghai。事件有毫秒时间；启动日志只有秒级精度。工具区间取 item/started 与 item/completed，网络时间取本地 time_unix。",
            f"页头采用 RPent 原生 elapsed_s={transcript['elapsed_s']}，不把包含前置启动的日志区间当作相同计时起点。",
            f"视频位置由动作帧数推导，不是墙钟时间。move_to.steps_used 含一次未执行动作的到达检查，本轮 {action_counts['move_to']} 次移动各减 1；策略执行 {policy_action_count} 个动作，脚本执行 {frame_count-policy_action_count} 个，合计 {frame_count} 帧，已实际逐帧解码核对。",
            f"原生统计记载 {transcript['stats']['turns_used']} 次模型回复、{transcript['stats']['tool_calls']} 次工具调用。事件包含 {type_counts['mcpToolCall']} 次 MCP 调用和 {type_counts['commandExecution']} 次本地命令调用；模型公开文字与内部推理记录分别保存。",
            f"原始事件包含 {len(reasoning_items)} 条 reasoning 事件；累计 reasoning token 为 {transcript['stats'].get('total_reasoning_output_tokens', 0)}。事件字段为空时，不能从 token 数还原内部推理全文。",
            "初始提示词是 RPent 提交给 Codex 的完整 turn 文本，不等于模型服务端全部输入。Codex 基础指令和 MCP 工具声明另外生效；指南与记忆是之后经工具读取的上下文。",
            "定位是否调用 SAM3 以时间轴中的 segment 调用为准；配置了 SAM3 endpoint 不等于实际使用了分割。",
            f"pi0_pick / pi0_doubled 是沿用的工具接口名，本轮实际策略为 {policy_label}；配置、bridge.log 后端元数据和逐次网络记录共同提供证据。工具返回的局部 success 与环境原生成功分别记录。",
            "页面说明为事后整理；代码链接用于解释机制，历史实际内容以原始事件和状态记录为准。未运行新实验，也未新增 GPT 调用。",
        ],
    }
    saved = {
        key: value for key, value in data.items() if key not in {"images", "video"}
    }
    (review / "task-flow-data.json").write_text(
        json.dumps(saved, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    template = (ROOT / "scripts/templates/libero-run-report.html").read_text(
        encoding="utf-8"
    )
    payload = json.dumps(data, ensure_ascii=False).replace("<", "\\u003c")
    target = review / "task-flow.html"
    target.write_text(template.replace("__RUN_DATA__", payload), encoding="utf-8")
    for name, checksum in source_hashes.items():
        assert hashlib.sha256((output / name).read_bytes()).hexdigest() == checksum
    print(
        f"{target}\nPrompt: {len(prompt)} chars; {len(events)} aligned events; {len(actions)} action records; {frame_count} video frames."
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    build(parser.parse_args().output)
