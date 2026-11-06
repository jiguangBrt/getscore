"""Tests for reward semantics that can silently invalidate a long RL run."""
import copy
import json
from pathlib import Path
import unittest
from pick_reward import RewardLedger

C=json.loads((Path(__file__).parents[1]/'configs/pilot.json').read_text())


def state(**changes):
    m=dict(object_pos=[0.,0.,.85],table_z=.8,clearance_m=0.,finger_object_distance_m=.15,
           bilateral_contact_steps=0,bilateral_contact=False,grasped=False,linear_speed=0.,angular_speed=0.)
    m.update(changes)
    return m


class RewardTests(unittest.TestCase):
    def test_hold_potential_cannot_farm_partial_holds(self):
        c=copy.deepcopy(C);c['pick']['hold_progress_mode']='discounted_potential'
        l=RewardLedger(c,'pick',state());discounted=0.;saw_break=False
        held=state(object_pos=[0,0,.95],clearance_m=.1,grasped=True,bilateral_contact=True,bilateral_contact_steps=3,finger_object_distance_m=0.)
        for i in range(200):
            m=held if i%21<20 else state()
            _,_,_,r=l.advance(m)
            discounted+=c['gamma_per_native_step']**i*r['reward_components']['hold']
            if i%21==20:saw_break|=r['reward_components']['hold']<0
        self.assertTrue(saw_break)
        self.assertAlmostEqual(discounted,0.,places=12)
        self.assertTrue(r['timeout'])

    def test_hold_potential_terminal_success_budget(self):
        c=copy.deepcopy(C);c['pick']['hold_progress_mode']='discounted_potential'
        l=RewardLedger(c,'pick',state());discounted=0.
        held=state(object_pos=[0,0,.95],clearance_m=.1,grasped=True,bilateral_contact=True,bilateral_contact_steps=3,finger_object_distance_m=0.)
        for i in range(40):
            _,term,_,r=l.advance(held)
            discounted+=c['gamma_per_native_step']**i*sum(v for k,v in r['reward_components'].items() if k in ('hold','hold_success'))
        self.assertTrue(term and r['skill_success'])
        self.assertAlmostEqual(discounted,.4*c['gamma_per_native_step']**39,places=12)

    def test_collision_is_cost_and_recovery_allowed(self):
        l=RewardLedger(C,'pick',state())
        _,term,_,i=l.advance(state(gripper_collision_cost=.01,gripper_collision_seconds=.05))
        self.assertFalse(term)
        self.assertEqual(i['reward_components']['gripper_collision'],-.01)
        _,term,_,i=l.advance(state())
        self.assertFalse(term)
        self.assertEqual(i['reward_components']['gripper_collision'],0.)
        self.assertAlmostEqual(i['cumulative_collision_cost'],.01)

    def test_soft_boundary_recovers(self):
        l=RewardLedger(C,'pick',state())
        _,term,trunc,i=l.advance(state(object_pos=[.051,0,.85]))
        self.assertFalse(term or trunc)
        self.assertLess(i['reward_components']['soft_boundary'],0)
        _,term,_,i=l.advance(state())
        self.assertFalse(term)
        self.assertEqual(i['reward_components']['soft_boundary'],0)

    def test_severe_and_drop(self):
        for m in (state(object_pos=[.2,0,.85]),state(object_pos=[0,0,.5],clearance_m=-.3)):
            _,term,trunc,i=RewardLedger(C,'pick',state()).advance(m)
            self.assertTrue(term and i['failure'])
            self.assertFalse(trunc)

    def test_deadline_is_terminal(self):
        l=RewardLedger(C,'pick',state())
        for _ in range(200):r,term,trunc,i=l.advance(state())
        self.assertTrue(term and i['timeout'])
        self.assertFalse(trunc)
        self.assertAlmostEqual(l.total,-.4)

    def test_contact_on_table_not_success(self):
        l=RewardLedger(C,'pick',state())
        for _ in range(199):
            _,term,_,i=l.advance(state(bilateral_contact=True,bilateral_contact_steps=3))
            self.assertFalse(term)
        self.assertEqual(l.earned['height'],0)
        self.assertEqual(l.hold,0)

    def test_verified_grasp_hold_and_no_repeat_credit(self):
        l=RewardLedger(C,'pick',state())
        m=state(object_pos=[0,0,.95],clearance_m=.1,grasped=True,bilateral_contact=True,bilateral_contact_steps=3,finger_object_distance_m=0.)
        for _ in range(40):r,term,trunc,i=l.advance(m)
        self.assertTrue(term and i['skill_success'])
        self.assertAlmostEqual(l.total,3.28)
        l=RewardLedger(C,'pick',state())
        l.advance(m)
        l.advance(state())
        _,_,_,i=l.advance(m)
        self.assertEqual(i['reward_components']['height'],0)
        self.assertEqual(i['reward_components']['approach'],0)
        self.assertEqual(i['reward_components']['settling'],0)


if __name__=='__main__':unittest.main()
