from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
from v3_zero_model_candidate import OFFSETS_DEG,J2_LIMITS_DEG,legacy_angles


class RezeroJointLimitsTest(unittest.TestCase):
    def test_full_legacy_j2_range_preserved_by_each_arm(self):
        for side in ('left','right'):
            for old_j2 in np.linspace(-105.,105.,101):
                new_j2=OFFSETS_DEG[side][1]-old_j2
                lower,upper=J2_LIMITS_DEG[side]
                self.assertGreaterEqual(new_j2,lower-1e-10)
                self.assertLessEqual(new_j2,upper+1e-10)
                pose=np.zeros(7);pose[1]=np.deg2rad(new_j2)
                self.assertAlmostEqual(np.rad2deg(legacy_angles(side,pose)[1]),old_j2)

    def test_home_and_unloading_references_in_limits(self):
        for side,new_angles in [('left',[105.,-65.,115.,-85.,120.]),('right',[-105.,65.,-115.,85.,-120.])]:
            lower,upper=J2_LIMITS_DEG[side]
            self.assertTrue(all(lower<=value<=upper for value in new_angles))


if __name__=='__main__':unittest.main()
