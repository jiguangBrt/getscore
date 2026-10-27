"""Use RPent's normal artifact/runtime path for the Long benchmark matrix."""
from __future__ import annotations

import importlib
import json
import math
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT/'RPent')]

from rpent.planner.base import Planner, PlannerResult
from rpent.prompt.utils import format_prompt
from robots.libero import robot_spec


class PurePolicyPlanner(Planner):
    """Execute unmodified policy chunks until native success or truncation."""

    def __init__(self, output_dir):
        self.output_dir = Path(output_dir)

    def solve(self, *, system_prompt, user_message, toolkit, max_turns, **kwargs):
        started = time.monotonic()
        prompt = os.environ['LONG_POLICY_PROMPT']
        horizon = int(os.environ['LONG_POLICY_HORIZON'])
        steps = int(os.environ['LONG_MAX_STEPS'])
        args = {'prompt': prompt, 'max_chunks': math.ceil(steps/horizon)}
        (self.output_dir/'pure_policy_prompt.txt').write_text(prompt+'\n')
        result = toolkit.execute_tool('pi0_doubled', args)
        event = {'tool':'pi0_doubled','arguments':args,'result':result.data}
        (self.output_dir/'pure_policy_events.jsonl').write_text(json.dumps(event,default=str)+'\n')
        error = str(result.data['error']) if result.is_error else None
        success = toolkit.solved()
        finish = {'status':'success' if success else 'failure',
                  'summary':'Pure policy, one episode; native environment success='+str(success)}
        return PlannerResult(finish_result=finish,messages=[event],error=error,
            stats={'backend':'pure_policy','elapsed_s':round(time.monotonic()-started,2),
                   'tool_calls':1,'total_input_tokens':0,'total_output_tokens':0})


def main():
    cli = importlib.import_module('rpent.cli.main')
    original_parser = cli._build_argparser
    original_factory = cli.build_planner
    original_prompt = robot_spec.system_prompt
    from robots.libero.env_client import LiberoEnvClient
    original_reset = LiberoEnvClient.reset
    original_step = LiberoEnvClient.step
    original_chunk = LiberoEnvClient.chunk_step

    def reset(self):
        result = original_reset(self)
        self._benchmark_native_steps = int(result[1].get('recovery',{}).get('resumed_steps',0))
        return result

    def step(self, action):
        result = original_step(self, action)
        self._benchmark_native_steps += 1
        return result

    def chunk(self, actions, **kwargs):
        remaining = int(os.environ['LONG_MAX_STEPS']) - self._benchmark_native_steps
        actions = actions[:remaining]
        result = original_chunk(self, actions, **kwargs)
        self._benchmark_native_steps += len(actions)
        return result

    LiberoEnvClient.reset = reset
    LiberoEnvClient.step = step
    LiberoEnvClient.chunk_step = chunk

    def parser():
        result = original_parser()
        for action in result._actions:
            if action.dest == 'planner':
                action.choices = [*action.choices,'none','kimi']
        return result

    def factory(planner_type, **kwargs):
        if planner_type == 'none':
            return PurePolicyPlanner(kwargs['output_dir'])
        if planner_type == 'kimi':
            from scripts.long_benchmark_kimi import KimiPlanner
            return KimiPlanner(**kwargs)
        return original_factory(planner_type,**kwargs)

    def prompt(variables):
        text = format_prompt(original_prompt(variables),variables=variables)
        text += '\n\nBenchmark protocol: one continuous episode, no reset. External task memory is empty. Do not read other experiment outputs, benchmark initial-state files, simulator state, or task definitions. Use observations and the provided perception tools to localize objects.\n'
        text += f'Native-action limit for this episode: {os.environ["LONG_MAX_STEPS"]}. Do not treat an older 1000-action limit as the current limit.\n'
        if os.environ['LONG_POLICY_BACKEND'] == 'gawm':
            text += ('The pi0_pick and pi0_doubled tool names in this run call the GAWM policy, not Pi0.5. '
                'GAWM accepts a fixed training-task vocabulary. For every policy tool call use exactly this instruction: '
                + os.environ['LONG_POLICY_PROMPT'] + '\n'
                'Do not shorten or paraphrase this instruction. Each chunk executes 8 native actions. '
                'Use bounded calls, inspect observations, and use the existing motion/gripper tools for corrections.\n')
        return text

    robot_spec.system_prompt = prompt
    cli._build_argparser = parser
    cli.build_planner = factory
    return cli.main()


if __name__ == '__main__':
    sys.exit(main())
