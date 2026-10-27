"""Describe task outcomes from environment flags and recorded stopping evidence."""

import re


def task_outcome(config, transcript, states):
    final = states[-1] if states else {}
    evidence = []
    native = None
    for source, field, value in (
        ("states.json", "terminated", final.get("terminated")),
        ("transcript", "environment_success", transcript.get("environment_success")),
        ("configuration.json", "task_success", config.get("task_success")),
    ):
        if isinstance(value, bool):
            native = value
            evidence.append(dict(source=source, field=field, value=value))
            break
    truncated = final.get("truncated") is True
    if isinstance(final.get("truncated"), bool):
        evidence.append(dict(source="states.json", field="truncated", value=final["truncated"]))
    finish = transcript.get("finish") or config.get("planner_finish") or {}
    error = transcript.get("error") or config.get("planner_error") or config.get("error")
    run_status = config.get("run_status")
    active = run_status in {"starting", "running"} and not finish and not error and not truncated
    ended = run_status in {"finished", "completed", "failed", "infrastructure_error", "stopped", "cancelled"} or bool(finish) or bool(error) or truncated
    result = dict(status="unknown", label="结果未确认", failure_type=None,
                  failure_label=None, detail="记录中没有足够的环境结果或结束信息。",
                  native_success=native, evidence=evidence,
                  model_summary=finish.get("summary"), error=str(error) if error else None)

    if native is True:
        result.update(status="success", label="成功", detail="环境已确认任务目标达成。")
        return result
    if active:
        result.update(status="running", label="运行中", detail="实验仍在运行，尚未确认成功。")
        return result
    if not ended:
        return result

    if not truncated and (error or run_status in {"infrastructure_error", "stopped", "cancelled"}):
        result.update(status="interrupted", label="运行中断")
    elif native is False:
        result.update(status="failure", label="失败")
    else:
        result["detail"] = "实验已结束，但没有保存环境成功标记；模型自述不能代替环境判定。"
        return result

    limits = config.get("limits", {})
    simulation = config.get("config", {}).get("simulation", {})
    max_steps = config.get("max_episode_steps") or limits.get("max_episode_steps") or simulation.get("max_episode_steps")
    max_turns = config.get("max_turns") or limits.get("max_turns")
    stats = transcript.get("stats") or config.get("stats") or {}
    turns = stats.get("turns_used")
    if truncated:
        label = "达到动作步数上限" + (f"（{max_steps} 步）" if max_steps else "")
        result.update(failure_type="step_limit", failure_label=label,
                      detail="动作预算已耗尽，环境任务目标尚未达成。")
    elif error:
        timeout = re.search(r"TimeoutError|TimeoutExpired|timed out|exceeded .*wall budget", str(error), re.I)
        result.update(failure_type="runtime_timeout" if timeout else "runtime_error",
                      failure_label="运行超时" if timeout else "运行异常",
                      detail="运行被超时限制中断。" if timeout else "运行记录报告异常；具体错误见下方运行说明。")
        evidence.append(dict(source="运行记录", field="error", value=str(error)))
    elif finish:
        result.update(failure_type="planner_stopped", failure_label="模型结束，目标未达成",
                      detail="模型结束了本轮实验，环境没有确认任务成功。")
        evidence.append(dict(source="transcript", field="finish.status", value=finish.get("status")))
    elif isinstance(turns, (int, float)) and isinstance(max_turns, (int, float)) and turns >= max_turns:
        result.update(failure_type="turn_limit", failure_label=f"达到规划轮数上限（{max_turns} 轮）",
                      detail="规划轮数已用尽，环境任务目标尚未达成。")
        evidence.append(dict(source="transcript.stats / configuration.json", field="turns_used / max_turns", value=[turns, max_turns]))
    elif run_status in {"infrastructure_error", "stopped", "cancelled"}:
        result.update(failure_type="runtime_interrupted", failure_label="运行中断",
                      detail="实验被中断，记录中没有更具体的结束原因。")
    else:
        result.update(failure_type="goal_not_achieved", failure_label="目标未达成，结束原因未记录",
                      detail="环境没有确认任务成功；未保存更具体的失败原因。")
    return result
