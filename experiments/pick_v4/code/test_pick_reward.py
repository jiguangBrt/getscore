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
