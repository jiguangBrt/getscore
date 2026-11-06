"""Unwanted gripper contacts, sampled after every MuJoCo physics substep.

Finger/pad contact with the requested target is the intended grasp. Palm-target,
table, floor and other external contacts are costs. Internal gripper pairs are
excluded. No reward or contact force is altered in the simulator.
"""
import numpy as np


def contact_severity(force, cfg):
    return float(np.clip((force-cfg['force_deadband_n']) /
                         (cfg['full_force_n']-cfg['force_deadband_n']), 0., 1.))


class ContactMonitor:
    def __init__(self, sim, gripper, target_ids, cfg):
        self.sim, self.cfg = sim, cfg
        self.gripper_ids = {sim.model.geom_name2id(n) for n in gripper.contact_geoms}
        self.finger_ids = {sim.model.geom_name2id(n)
                           for key in ('left_finger', 'right_finger', 'left_fingerpad', 'right_fingerpad')
                           for n in gripper.important_geoms[key]}
        self.target_ids = set(target_ids)
        self.dt = float(sim.model.opt.timestep)

    def sample(self):
        import mujoco
        pairs = []
        force_sum = 0.
        intended = 0.
        for i, c in enumerate(self.sim.data.contact[:self.sim.data.ncon]):
            a, b = int(c.geom1), int(c.geom2)
            if a not in self.gripper_ids and b not in self.gripper_ids:
                continue
            if a in self.gripper_ids and b in self.gripper_ids:
                continue
            g, other = (a, b) if a in self.gripper_ids else (b, a)
            f = np.zeros(6)
            mujoco.mj_contactForce(self.sim.model._model, self.sim.data._data, i, f)
            normal = max(0., float(f[0]))
            if g in self.finger_ids and other in self.target_ids:
                intended += normal
                continue
            force_sum += normal
            if normal > .01:
                pairs.append(dict(gripper_geom=self.sim.model.geom_id2name(g),
                                  other_geom=self.sim.model.geom_id2name(other),
                                  normal_force_n=normal, penetration_m=max(0., -float(c.dist)),
                                  category='palm_target' if other in self.target_ids else 'external'))
        severity = contact_severity(force_sum, self.cfg)
        return dict(force_n=force_sum, intended_force_n=intended, severity=severity, pairs=pairs)

    def summarize(self, samples):
        assert samples, 'collision monitor received no physics substeps'
        # Deduplicate records while retaining the peak for each pair.
        pairs = {}
        for s in samples:
            for p in s['pairs']:
                key = p['gripper_geom'], p['other_geom']
                if key not in pairs or p['normal_force_n'] > pairs[key]['normal_force_n']:
                    pairs[key] = p
        return dict(gripper_collision_cost=self.cfg['max_cost_per_second']*self.dt*sum(s['severity'] for s in samples),
                    gripper_collision_peak_force_n=max(s['force_n'] for s in samples),
                    gripper_intended_contact_peak_force_n=max(s['intended_force_n'] for s in samples),
                    gripper_collision_seconds=self.dt*sum(s['severity'] > 0 for s in samples),
                    gripper_collision_impulse_ns=self.dt*sum(s['force_n'] for s in samples),
                    collision_physics_substeps=len(samples), gripper_collision_pairs=list(pairs.values()))
