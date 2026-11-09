"""Physical and frame-contract checks for the two-controller shared plant."""
import unittest

import mujoco
import numpy as np

from duel import Controller, impact_velocity, load_scene


class DuelTests(unittest.TestCase):
    def test_mirrored_robots_have_matching_local_observations_and_disjoint_actuators(self):
        scene,cfg,_=load_scene()
        p1,p2=scene.players
        s1,s2=p1.state(),p2.state()
        np.testing.assert_allclose(s1.base_pos_w,s2.base_pos_w,atol=1e-12)
        self.assertAlmostEqual(abs(np.dot(s1.base_quat_w,s2.base_quat_w)),1.)
        np.testing.assert_array_equal(s1.q,s2.q)
        self.assertEqual(np.intersect1d(p1.act,p2.act).size,0)
        self.assertEqual(scene.model.nu,62)
        self.assertEqual(np.count_nonzero(scene.model.jnt_type==mujoco.mjtJoint.mjJNT_FREE),3)
        controls=[Controller(cfg,p) for p in scene.players]
        for control in controls:
            control.tick(None,0)
        np.testing.assert_allclose(p1.q_des,p2.q_des,atol=1e-6)
        self.assertFalse(np.shares_memory(controls[0].last_action,controls[1].last_action))
        scene.step()
        self.assertAlmostEqual(scene.data.time,cfg.control_dt)

    def test_both_rackets_can_contact_the_one_physical_ball(self):
        scene,_,_=load_scene()
        for i in range(2):
            scene.reset()
            site=mujoco.mj_name2id(scene.model,mujoco.mjtObj.mjOBJ_SITE,f"p{i+1}_right_racket")
            scene.set_ball(scene.data.site_xpos[site].copy(),[0,0,0])
            contacts,_,_=scene.step()
            self.assertIn(i,contacts)
            self.assertNotIn(1-i,contacts)

    def test_moving_paddle_normal_impulse_reconstructs_desired_outgoing_velocity(self):
        incoming=np.array([-2.,.1,-1.])
        outgoing=np.array([3.,.3,1.5])
        e=.654
        paddle=impact_velocity(incoming,outgoing,e)
        normal=(outgoing-incoming)/np.linalg.norm(outgoing-incoming)
        actual=incoming-(1+e)*np.dot(incoming-paddle,normal)*normal
        np.testing.assert_allclose(actual,outgoing,atol=1e-12)


if __name__=="__main__":
    unittest.main()
