"""Write readable companions to native RPent LIBERO transcripts."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]


def _block(value: Any, language: str = "json") -> str:
    text = (
        value
        if isinstance(value, str)
        else json.dumps(value, ensure_ascii=False, indent=2)
    )
    fence = "`" * max(
        3, max((len(m[0]) for m in re.finditer(r"`+", text)), default=0) + 1
    )
    return f"{fence}{language}\n{text}\n{fence}\n"


def export_transcripts(output: Path) -> list[Path]:
    """Render saved task text, action evidence and model messages without model calls.

    Original JSON files are read only. Companions go to review/<run-name>/.
    Missing states are allowed for runs that stopped before artifact writing.
    """
    output = output.resolve()
    review = ROOT / "review" / output.name
    state_path = output / "states.json"
    steps = (
        json.loads(state_path.read_text(encoding="utf-8"))["steps"]
        if state_path.exists()
        else []
    )
    written = []
    for source in sorted(output.glob("transcript_*.json")):
        record = json.loads(source.read_text(encoding="utf-8"))
        task = steps[0].get("extras", {}).get("task_language") if steps else None
        sections = [
            "# LIBERO-PRO 任务记录\n",
            f"原始汇总：[{source.name}]({source})；本文件由已有记录生成，不调用模型。\n",
            f"原始运行目录：`{output}`；复盘文件保存在 `review/{output.name}/`。\n",
            "## 原始任务描述\n",
            _block(task or "未保存任务原文，请检查环境记录。", "text"),
            "来源：`states.json → steps[0].extras.task_language`。\n",
            "## 运行结果\n",
            (
                "`finish` 是规划器的声明；`final_observation` 来自环境记录。"
                "抓取工具的 `success` 只表示局部抓取判据，不等于整项任务成功。\n"
            ),
            _block(
                {
                    **{
                        key: value for key, value in record.items() if key != "messages"
                    },
                    "final_observation": {
                        key: steps[-1].get(key)
                        for key in ("step_idx", "terminated", "truncated")
                    }
                    if steps
                    else None,
                }
            ),
            "## 实际动作与返回结果\n",
            (
                "按 `states.json` 的记录顺序列出。编号是观测记录号，不是底层动作步数。"
                "`move_to`、夹爪及释放工具执行脚本控制；`pi0_pick`/`pi0_doubled` 调用 VLA。\n"
            ),
        ]
        for step in steps:
            command = step.get("command")
            if command:
                sections.extend(
                    [
                        f"### 观测 {step['step_idx']}：{command.get('action', 'unknown')}\n",
                        _block(
                            {
                                key: step.get(key)
                                for key in (
                                    "command",
                                    "result",
                                    "terminated",
                                    "truncated",
                                )
                            }
                        ),
                    ]
                )
        sections.extend(
            [
                "## GPT 文字与工具摘要\n",
                (
                    "以下保留模型公开输出，按 turn 分段；不代表完整内部推理。"
                    "工具摘要不含完整参数时，以前面的状态记录为准；"
                    "未改变状态的工具可查 `codex_*.txt.stream.jsonl` 原始事件。\n"
                ),
            ]
        )
        for index, message in enumerate(record.get("messages", []), 1):
            sections.append(f"### 消息 {index}：{message.get('role', 'unknown')}\n")
            content = message.get("content")
            if isinstance(content, str):
                parts = re.split(r"(?m)^\[agent\] === (turn \d+/\d+) ===\s*$", content)
                if parts[0].strip():
                    sections.append(_block(parts[0].strip(), "text"))
                for offset in range(1, len(parts), 2):
                    sections.extend(
                        [
                            f"#### {parts[offset]}\n",
                            _block(parts[offset + 1].strip(), "text"),
                        ]
                    )
            else:
                sections.append(_block(content))
        review.mkdir(parents=True, exist_ok=True)
        target = review / source.with_suffix(".md").name
        target.write_text("\n".join(sections), encoding="utf-8")
        written.append(target)
    return written


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    for path in export_transcripts(args.output):
        print(path)
