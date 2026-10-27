"""Task success and stopping reasons must come from recorded evidence."""

import pytest

from run_viewer.outcomes import task_outcome


@pytest.mark.parametrize("config,transcript,final,status,reason", [
    ({"run_status": "finished"}, {}, {"terminated": True, "truncated": True}, "success", None),
    ({"run_status": "running"}, {}, {"terminated": False}, "running", None),
    ({"run_status": "finished", "max_episode_steps": 1000}, {},
     {"terminated": False, "truncated": True}, "failure", "step_limit"),
    ({"run_status": "finished"}, {"finish": {"status": "failure", "summary": "Grasp failed"}},
     {"terminated": False, "truncated": False}, "failure", "planner_stopped"),
    ({"run_status": "finished"}, {"finish": {"status": "success"}},
     {"terminated": False}, "failure", "planner_stopped"),
    ({"run_status": "finished", "task_success": True}, {},
     {"terminated": False}, "failure", "goal_not_achieved"),
    ({"run_status": "finished", "limits": {"max_turns": 80}}, {"stats": {"turns_used": 80}},
     {"terminated": False}, "failure", "turn_limit"),
    ({"run_status": "infrastructure_error", "error": "RuntimeError: service unavailable"}, {},
     {"terminated": False}, "interrupted", "runtime_error"),
    ({"run_status": "finished"}, {"error": "Codex SDK timed out after 1200s"},
     {"terminated": False}, "interrupted", "runtime_timeout"),
    ({"run_status": "finished"}, {"error": "TimeoutError: Kimi ACP exceeded 1200s wall budget"},
     {}, "interrupted", "runtime_timeout"),
    ({"run_status": "stopped"}, {}, {"terminated": False}, "interrupted", "runtime_interrupted"),
    ({"run_status": "finished"}, {}, {"terminated": False}, "failure", "goal_not_achieved"),
    ({"run_status": "finished"}, {"finish": {"status": "failure"}}, {}, "unknown", None),
    ({"run_status": "finished"}, {"finish": {"status": "success"}}, {}, "unknown", None),
    ({}, {}, {"terminated": False}, "unknown", None),
    ({"run_status": "running"}, {}, {"terminated": False, "truncated": True}, "failure", "step_limit"),
    ({"run_status": "finished"}, {"error": "cleanup failed"}, {"terminated": True}, "success", None),
])
def test_task_result_and_stop_reason(config, transcript, final, status, reason):
    result = task_outcome(config, transcript, [final])
    assert result["status"] == status
    assert result["failure_type"] == reason


def test_model_failure_summary_is_preserved_as_self_report_without_semantic_classification():
    result = task_outcome(
        {"run_status": "finished", "max_episode_steps": 1000},
        {"finish": {"status": "failure", "summary": "The mug slipped during carry."}},
        [{"terminated": False, "truncated": True}],
    )
    assert result["model_summary"] == "The mug slipped during carry."
    assert result["failure_label"] == "达到动作步数上限（1000 步）"
    assert result["evidence"] == [
        dict(source="states.json", field="terminated", value=False),
        dict(source="states.json", field="truncated", value=True),
    ]


def test_environment_result_can_fall_back_to_recorded_native_success():
    assert task_outcome({}, {"environment_success": True}, [])["status"] == "success"
    assert task_outcome({"task_success": False, "run_status": "finished"}, {}, [])["status"] == "failure"


def test_ordinary_tool_error_is_not_a_task_failure_reason():
    result = task_outcome({"run_status": "finished"}, {}, [
        {"terminated": False, "result": {"error": "temporarily unavailable"}},
        {"terminated": True},
    ])
    assert result["status"] == "success" and result["failure_type"] is None
