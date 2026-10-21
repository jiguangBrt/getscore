"""Build an offline report for the recorded first Codex LIBERO run on 2026-10-20."""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
from collections import Counter
from datetime import datetime
from pathlib import Path
from urllib.parse import quote
from zoneinfo import ZoneInfo

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
    source_hashes = {
        path.name: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in (
            stream,
            transcript_path,
            output / "states.json",
            output / "configuration.json",
            output / "network-timing.jsonl",
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
        explanation="启动器先启动 OpenPI 适配层，再启动 RPent。LIBERO-PRO 在本机创建环境并渲染；π0.5 已在远程服务中运行。此时间来自秒级日志。",
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
            explanation = "RPent 保存汇总：elapsed_s=482.0，环境最终 terminated=true、truncated=false。启动日志到本条约 490 秒，包含不在该 elapsed_s 中的前置启动时间。"
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
    action_notes = {
        1: "GPT 主动到篮子上方观察内部，不是误抓篮子。腕部视角用于纠正主相机选点落在内壁的问题。",
        2: "GPT 规划的中间航点，分段移动以满足工具对单次横向位移的限制。",
        3: "GPT 用脚本控制移到沙拉酱上方，再通过腕部画面和反投影核对目标。",
        4: "第一次 VLA 抓取。GPT 传入缩短的抓取指令；π0.5 生成动作，机械臂偏向番茄酱且没有夹住物体。8 次推理均通信成功，但局部抓取失败。",
        5: "GPT 观察失败画面后，用 move_to 返回沙拉酱上方；这次修正由脚本执行。",
        6: "第二次 VLA 抓取。改用完整任务指令，起始位置和 lift_thresh 也变化；达到局部抓取判据。不能用本轮结果证明单独改指令就是成功原因。",
        7: "GPT 调用脚本持续闭合夹爪，保持瓶子。",
        8: "GPT 调用脚本抬升，避开桌面物体。",
        9: "GPT 调用脚本经过中间航点搬运，不调用 π0.5。",
        10: "GPT 调用脚本将瓶子对准已测量的篮子内部中心。",
        11: "GPT 调用 release 在篮子上方松开夹爪。环境返回 terminated=true；这一轮任务实际成功。",
    }
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
                    explanation=action_notes[action_index],
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
                elif name == "pi0_pick":
                    actual_steps = step["result"]["chunks_used"] * 5
                elif name == "set_gripper":
                    actual_steps = step["result"]["steps"]
                else:
                    actual_steps = step["result"]["steps_used"]
                fields["video"] = {
                    "start": frame_cursor / 20,
                    "end": (frame_cursor + actual_steps) / 20,
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
                    assert len(linked) == step["result"]["chunks_used"] == 8
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

    assert prompt is not None and len(prompt) == 33730
    assert action_index == 11 and frame_cursor == 344
    assert len(reasoning_items) == 16 and all(
        not r["summary_count"] and not r["content_count"] for r in reasoning_items
    )
    (review / "initial-prompt.txt").write_text(prompt, encoding="utf-8")
    for index, entry in enumerate(network, 1):
        at = entry["time_unix"] * 1000
        parent = next(
            event for event in events if index in event.get("networkRows", [])
        )
        add(
            "network",
            f"π0.5 请求 #{index:02d}",
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
            "本次从原始 userMessage 导出的 33,730 字符 RPent 提交文本。",
        ),
        (
            stream.name,
            "原始事件",
            "初始输入、公开输出、工具参数/返回、毫秒时间、token；逐行 JSON，部分返回含编码图像。",
        ),
        (
            "states.json",
            "环境证据",
            "12 份观测；每次动作的参数、结果、原始任务、终止标志；观测号不等于底层动作步数。",
        ),
        (
            "episode.mp4",
            "执行视频",
            "344 个实际动作的连续画面，20fps，17.2 秒；没有包含模型等待时间。",
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
            "16 次请求开始时间及各项耗时；通过同机时间戳与两次 VLA 工具区间对齐。",
        ),
        (
            "object_swap_t2_s0.json",
            "模型事后分析",
            "GPT 主动写出的定位、策略、失败与恢复分析；应与原始状态交叉核对。",
        ),
        (
            "object_swap_t2_s0_recipe.jsonl",
            "动作清单",
            "11 条实际动作的工具名与参数，不含结果；便于核对脚本控制与 VLA 的边界。",
        ),
        (
            "run.log",
            "带时间日志",
            "启动、MCP 地址、33,730 字符计数、公开文字与工具摘要；时间精确到秒。",
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
            "OpenPI 桥接服务启动与故障排查；动作请求时间看 network-timing。",
        ),
        (
            "agentview_high.png",
            "主相机原图目录",
            "00–11：初始场景及工具执行后高分辨率画面；同号对应 states.steps 的 step_idx。",
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

    data = {
        "title": "沙拉酱入篮 · 一次真实实验的数据流",
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
            "启动日志 10:42:16 到汇总日志 10:50:26 约 490 秒；RPent 自己报告的 elapsed_s=482.0 使用稍晚开始的计时区间。页头采用该原生统计，不把两者当作相同计时起点。",
            "视频位置由保存动作帧数推导，不是墙钟时间。move_to.steps_used 含一次未执行动作的到达检查；本轮 7 次 move_to 各减 1；两次 VLA 各执行 8×5 个动作，合计 344 帧。",
            "19 次模型回复是 tokenUsage 更新计数；不等于 19 个顶层 SDK turn，也不等于 19 条可见文字。原始记录含 18 条 agentMessage、36 条工具调用、16 条 reasoning 事件。",
            "16 条 reasoning 事件的 summary 与 content 均为空。可以还原公开计划、工具决策与结果，无法还原完整内部推理。194 个 reasoning token 的计数不意味着存在可读推理全文。",
            "初始提示词是 RPent 提交给 Codex 的完整 turn 文本，不等于模型服务端全部输入。Codex 基础指令和 MCP 工具声明另外生效；指南与记忆是之后经工具读取的上下文。",
            "本轮无 segment 调用，SAM3 虽已配置，但实际定位采用 GPT 图像选点 + back_project；不能把部署了 SAM3 说成这轮用它做了分割。",
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
        f"{target}\nPrompt: {len(prompt)} chars; {len(events)} aligned events; 11 action records; 344 video frames."
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    build(parser.parse_args().output)
