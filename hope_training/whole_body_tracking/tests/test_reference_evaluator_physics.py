"""Regression coverage for the standalone physical-ball evaluator (no Isaac)."""
from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace
import sys
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
SCRIPTS = ROOT / "hope_training/whole_body_tracking/scripts"
sys.path[:0] = [str(SCRIPTS), str(ROOT / "a3_deploy/a3_deploy_example/reference")]
spec = importlib.util.spec_from_file_location("reference_eval_physics_test", SCRIPTS / "mujoco_eval_onnx.py")
evaluator = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = evaluator
spec.loader.exec_module(evaluator)

from a3_deploy_onnx_ref_pingpong.racket_command import RacketCommand


class PredictorTests(unittest.TestCase):
    def setUp(self):
        self.metric = evaluator._load_success_metric(ROOT)
        self.physics = self.metric.BallPhysics.from_config()
        self.scene = SimpleNamespace(near_edge_x=.30, table_height=.76, ball_radius=.02,
                                     length=2.74, width=1.525, net_x_table=1.37,
                                     offset=np.array([.30, .7625, .76]),
                                     table_restitution=.9215)
        self.args = SimpleNamespace(strike_plane_x=0.)

    def predict(self, position, velocity):
        return evaluator._predict_command(np.array(position), np.array(velocity), 1,
                                          self.scene, self.physics, self.args, 1, 0, RacketCommand)

    def test_first_seed_zero_ball_is_not_predicted_underground(self):
        # Exact first incoming ball from the failing continuous seed-0 run.
        cmd = self.predict([2.2204867619680972, -.3848443525253613, 1.0726540478400546],
                           [-1.9633086954072048, .14827576778241097, 4.613717110404098])
        self.assertIsNotNone(cmd)
        self.assertGreater(cmd.position[2], self.scene.table_height + self.physics.ball_radius)
        self.assertGreater(cmd.time_to_strike, 0.)

    def test_departed_ball_does_not_create_a_zero_time_command(self):
        self.assertIsNone(self.predict([-.1, -.2, 1.], [-2., 0., 0.]))

    def test_ball_below_surface_is_not_an_actionable_target(self):
        self.assertIsNone(self.predict([1., -.2, .2], [-2., 0., -1.]))

    def test_direct_airborne_intercept_still_works(self):
        cmd = self.predict([.6, -.2, 1.2], [-3., 0., 0.])
        self.assertIsNotNone(cmd)
        self.assertAlmostEqual(cmd.position[0], .3)
        self.assertGreater(cmd.position[2], .78)


class SceneTests(unittest.TestCase):
    def test_standalone_scene_does_not_inherit_the_gate3_robot_table(self):
        from a3_deploy_onnx_ref_pingpong.config import RuntimeConfig
        from a3_deploy_onnx_ref_pingpong.joint_order import JOINT_NAMES
        from mujoco_pingpong_scene import PingPongRealPhysicsScene
        cfg = RuntimeConfig.load(ROOT / "a3_deploy/a3_deploy_example/config/hope_pingpong_runtime.yaml")
        metric = evaluator._load_success_metric(ROOT)
        scene = PingPongRealPhysicsScene(str(cfg.model_xml_path), metric.load_ball_physics_config(), JOINT_NAMES)
        for name in ("gate3_table_collision", "gate3_net_collision"):
            gid = scene._geom_id(name)
            self.assertEqual(int(scene.model.geom_contype[gid]), 0)
            self.assertEqual(int(scene.model.geom_conaffinity[gid]), 0)
        self.assertEqual(scene.model.geom_rgba[scene._geom_id("gate3_table_visual"), 3], 0.)
        # The standalone table and paddle/ball pair must remain enabled.
        self.assertNotEqual(int(scene.model.geom_contype[scene._geom_id("table_geom")]), 0)
        self.assertGreater(scene.model.npair, 0)
        scene.close()


class CancelledInterceptTests(unittest.TestCase):
    def test_lost_intercept_recovers_without_reengaging_the_old_ball(self):
        from a3_deploy_onnx_ref_pingpong.lifecycle import SwingLifecycle, Phase
        life = SwingLifecycle()
        state = SimpleNamespace(base_pos_w=np.array([0., 0., 1.]))
        command = RacketCommand(7, 0, 1, [.58, -.44, 1.1], [1.8, .2, 1.], .5)
        life.update(command, state)
        self.assertEqual(life.phase, Phase.SWING)
        self.assertFalse(life.cancel_active_task(6))
        self.assertTrue(life.cancel_active_task(7))
        # A stale mailbox value for the same ball must not start a second swing.
        target = life.update(command, state)
        self.assertEqual(life.phase, Phase.RECOVERY)
        self.assertEqual(life._last_engaged_task_id, 7)
        np.testing.assert_array_equal(target.vel_w, np.zeros(3))
        np.testing.assert_allclose(target.pos_w, state.base_pos_w + [.4, .2, -.05])
        self.assertFalse(life.cancel_active_task(7))

    def test_command_loss_after_the_strike_does_not_interrupt_follow_through(self):
        from a3_deploy_onnx_ref_pingpong.lifecycle import SwingLifecycle, Phase
        life = SwingLifecycle()
        state = SimpleNamespace(base_pos_w=np.array([0., 0., 1.]))
        life.update(RacketCommand(7, 0, 1, [.58, -.44, 1.1], [1.8, .2, 1.], .01), state)
        self.assertEqual(life.phase, Phase.FOLLOW_THROUGH)
        self.assertFalse(life.cancel_active_task(7))
        self.assertEqual(life.phase, Phase.FOLLOW_THROUGH)


if __name__ == "__main__":
    unittest.main()
