import io
from pathlib import Path
import sys
import unittest
import xml.etree.ElementTree as ET

import numpy as np
from scipy.spatial.transform import Rotation
import yourdfpy

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
from v3_dual_zero_review import ZERO_DEG,load_review,old_to_review


class DualZeroReviewTest(unittest.TestCase):
    def test_zero_geometry_and_new_positive_j6_axes(self):
        path=Path(__file__).resolve().parents[1]/'generated/zero_model_candidate_20261018/robot_rezero_candidate.urdf'
        definition=ET.parse(path)
        for side in ('left','right'):definition.find(f"joint[@name='{side}_tool0_fixed']/origin").set('xyz','0 0 0.191')
        old=yourdfpy.URDF.load(io.BytesIO(ET.tostring(definition.getroot())),load_meshes=False)
        new=load_review(path,False)
        zero={f'{side}_joint{i}':np.deg2rad(value) for side in ('left','right') for i,value in enumerate(ZERO_DEG[side],1)}
        old.update_cfg(zero);new.update_cfg({name:0. for name in new.actuated_joint_names})
        for link in old.link_map:
            np.testing.assert_allclose(old.get_transform(link,old.base_link),new.get_transform(link,new.base_link),atol=1e-8)
        for side in ('left','right'):
            name=side+'_joint6';joint=new.joint_map[name]
            before=new.get_transform(joint.child,new.base_link).copy()
            axis=before[:3,:3]@joint.axis
            self.assertGreater(axis[2],.999)
            new.update_cfg({name:.001})
            after=new.get_transform(joint.child,new.base_link)
            increment=Rotation.from_matrix(after[:3,:3]@before[:3,:3].T).as_rotvec()
            self.assertAlmostEqual(float(increment@axis),.001,places=9)
            new.update_cfg({name:0.})
        rng=np.random.default_rng(24)
        for _trial in range(20):
            values={name:float(rng.uniform(-.4,.4)) for name in old.actuated_joint_names}
            old.update_cfg(values);new.update_cfg(old_to_review(values))
            for side in ('left','right'):
                np.testing.assert_allclose(old.get_transform(side+'_tool0',old.base_link),new.get_transform(side+'_tool0',new.base_link),atol=1e-8)


if __name__=='__main__':unittest.main()
