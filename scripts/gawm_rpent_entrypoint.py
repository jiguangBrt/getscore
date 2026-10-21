"""Run the existing RPent CLI with an explicit GAWM instruction constraint."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "RPent"))

from robots.libero import robot_spec
from rpent.cli.main import main
from rpent.prompt.utils import format_prompt

original = robot_spec.system_prompt


def system_prompt(variables):
    return (
        format_prompt(original(variables), variables=variables)
        + """

GAWM policy for this run:
The pi0_pick and pi0_doubled tool identifiers are retained for RPC compatibility,
but their backend is the copied StarVLA GAWM LIBERO policy, not Pi0.5.
GAWM uses a fixed training-task vector vocabulary. For this single-task run,
always pass this exact policy prompt: pick up the salad dressing and place it in the basket
Do not send rewritten sub-instructions such as 'pick up the salad dressing';
unknown task strings are rejected. Use scripted motion/gripper tools for
individual corrective motions and observe the result before continuing.
Each policy chunk executes 8 actions. Use short, bounded calls and judge success
only from the native environment termination flag and observations.
"""
    )


robot_spec.system_prompt = system_prompt

if __name__ == "__main__":
    main()
