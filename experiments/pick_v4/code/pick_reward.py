"""Finite-horizon pick reward, with bounded progress and recoverable boundaries."""
import copy
import numpy as np

VERSION = 'single-white-mug-pick-v4-20261106'


class RewardLedger:
    def __init__(self, cfg, skill, initial):
        assert skill == 'pick'
        self.cfg = copy.deepcopy(cfg)
        self.initial = copy.deepcopy(initial)
        self.prev = copy.deepcopy(initial)
        self.step = self.hold = self.stable = 0
        self.peak = initial['clearance_m']
        self.path = self.descent = self.total = 0.
        self.earned = {}
        self.events = set()
        self.hold_heights = []
        self.done = False

    def advance(self, m):
        assert not self.done
        c, s = self.cfg, self.cfg['pick']
        self.step += 1
        parts = {'step': -c['reward']['step_penalty']}

        def credit(name, score, budget):
            score = float(np.clip(score, 0, 1))
            before = self.earned.get(name, 0.)
            parts[name] = budget * max(0., score - before)
            self.earned[name] = max(before, score)

        xy = float(np.linalg.norm(np.array(m['object_pos'][:2]) - self.initial['object_pos'][:2]))
        self.path += float(np.linalg.norm(np.array(m['object_pos'][:2]) - self.prev['object_pos'][:2]))
        h = m['clearance_m']
        self.peak = max(self.peak, h)
        # This is occupancy cost: returning to the free region stops the cost.
        excess = max(0., xy - s['soft_boundary_free_radius_m'])
        fraction = min(1., excess / (s['soft_boundary_full_radius_m'] - s['soft_boundary_free_radius_m']))
        parts['soft_boundary'] = -s['soft_boundary_cost_per_second'] * fraction**2 / c['control_hz']
        d0 = max(self.initial['finger_object_distance_m'], .005)
        credit('approach', 1. - m['finger_object_distance_m'] / d0, s['approach_reward_budget'])
        if m['bilateral_contact_steps'] >= s['contact_confirm_steps'] and 'bilateral_contact' not in self.events:
            self.events.add('bilateral_contact')
            parts['bilateral_contact'] = s['contact_event_bonus']
        # Ground contact is not an airborne grasp. Reward height only with verified hold.
        score = (h - s['grasp_min_clearance_m']) / (s['lift_reward_max_m'] - s['grasp_min_clearance_m']) if m['grasped'] else 0.
        credit('height', score, s['height_reward_budget'])
        # Provide feedback before the strict 2s hold is reached. This is bounded
        # progress credit, never a per-step reward for indefinitely hovering.
        if m['grasped'] and h >= s['success_min_lift_m'] and xy < s['success_xy_radius_m']:
            linear_score=max(0.,1.-m['linear_speed']/s['settling_linear_scale_m_s'])
            angular_score=max(0.,1.-m['angular_speed']/s['settling_angular_scale_rad_s'])
            credit('settling',linear_score*angular_score,s['settling_reward_budget'])
        if self.earned.get('height', 0) > 0:
            before = max(0., self.descent - s['descent_deadband_m'])
            self.descent += max(0., self.prev['clearance_m'] - h)
            parts['descent'] = -s['descent_penalty_per_m'] * (max(0., self.descent - s['descent_deadband_m']) - before)
        picked = (m['grasped'] and h >= s['success_min_lift_m']
                  and xy < s['success_xy_radius_m']
                  and m['linear_speed'] <= s['hold_linear_speed_m_s']
                  and m['angular_speed'] <= s['hold_angular_speed_rad_s'])
        if picked:
            heights = self.hold_heights + [h]
            if max(heights) - min(heights) > s['hold_height_tolerance_m']:
                heights = [h]
            self.hold_heights = heights
        else:
            self.hold_heights = []
        self.hold = len(self.hold_heights)
        credit('hold', self.hold / c['success']['pick_hold_steps'], s['hold_progress_reward_budget'])
        failure = h < -.10 or m['object_pos'][2] < m['table_z'] - .10
        reason = 'fell_off_table' if failure else None
        if xy >= s['hard_boundary_radius_m']:
            failure, reason = True, 'severe_xy_boundary'
        success = bool(self.hold >= c['success']['pick_hold_steps'] and not failure)
        timeout = self.step >= c['max_native_steps_per_skill'] and not success and not failure
        if success:
            parts['success'], reason = c['reward']['terminal_bonus'], 'success'
        if failure:
            parts['failure'] = -c['reward']['failure_penalty']
        if timeout:
            parts['timeout'], reason = -c['reward']['timeout_penalty'], 'task_deadline_200'
        terminated = bool(success or failure or timeout)
        reward = float(sum(parts.values()))
        self.total += reward
        self.done = terminated
        self.prev = copy.deepcopy(m)
        phase = 'hold' if self.hold else ('lift' if m['grasped'] else ('contact' if m['bilateral_contact'] else 'approach'))
        info = dict(m, native_step=self.step, remaining_steps=max(0, c['max_native_steps_per_skill'] - self.step),
                    object_lift_m=h, max_object_lift_m=self.peak, xy_offset_m=xy, xy_path_m=self.path,
                    hold_steps=self.hold, stable_count=0, pick_count=self.hold, phase=phase,
                    reward_components=parts, reward=reward, episode_reward=self.total,
                    skill_success=success, failure=bool(failure), timeout=bool(timeout),
                    termination_reason=reason, terminated=terminated, truncated=False,
                    earned=dict(self.earned), events=sorted(self.events), cumulative_descent_m=self.descent)
        return reward, terminated, False, info
