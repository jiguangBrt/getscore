import copy
import unittest
import numpy as np
from task_advantages import normalize_by_object


class TaskAdvantageTests(unittest.TestCase):
    def test_task_baseline_shift_does_not_change_actor_signal(self):
        rows=[dict(object_type=t,advantage=v) for t,values in [('easy',[1,2,4,5]),('hard',[-3,-1])] for v in values]
        shifted=copy.deepcopy(rows)
        for r in shifted:
            if r['object_type']=='hard':r['advantage']+=12.
        stats=normalize_by_object(rows);normalize_by_object(shifted)
        np.testing.assert_allclose([r['normalized_actor_advantage'] for r in rows],
                                   [r['normalized_actor_advantage'] for r in shifted],atol=1e-12)
        self.assertAlmostEqual(stats['easy']['task_weight']*4,stats['hard']['task_weight']*2)

    def test_constant_advantages_do_not_create_noise(self):
        rows=[dict(object_type='mug',advantage=3.) for _ in range(8)]
        normalize_by_object(rows)
        self.assertTrue(all(r['normalized_actor_advantage']==0 for r in rows))


if __name__=='__main__':unittest.main()
