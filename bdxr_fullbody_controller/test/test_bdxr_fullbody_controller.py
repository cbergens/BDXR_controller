# SPDX-FileCopyrightText: 2026 NVIDIA CORPORATION & AFFILIATES
# SPDX-License-Identifier: Apache-2.0

"""Tests for the BDX-R full-body controller."""

import unittest
from collections import deque

import numpy as np
from geometry_msgs.msg import Twist
from bdxr_fullbody_controller.bdxr_fullbody_controller import BDXRFullbodyController
from sensor_msgs.msg import Imu, JointState


def build_controller():
    """Build a controller without ROS, holding only the observation states"""

    controller = object.__new__(BDXRFullbodyController)
    controller._obs_histories = {}
    controller._history_length = 5
    controller.foot_force_window = deque([[float("inf"), float("inf")]] * 3, maxlen=3)
    controller._foot_contact_threshold = 1.0
    controller._cmd_vel = Twist()
    controller._head_pose_cmd = np.zeros(3)
    controller._policy_step = 0
    controller._policy_period = 0.02
    controller._gait_cmd_thresh = 0.05
    controller._gait_period = 0.476
    controller._base_mass = 7.113365
    controller._previous_action = np.zeros(14)
    return controller


class TestBDXRFullbodyController(unittest.TestCase):
    """Test policy observation construction."""

    def test_obs_shape_and_gravity(self):
        """Upright robot with no command gives 295 finite numbers and a vertical gravity reading"""

        # An identity quaternion means the IMU is level, so gravity in the body frame is [0, 0, -1].
        # An empty JointState and a zero command check that missing inputs read as 0 instead of raising.
        imu = Imu()
        imu.orientation.w = 1.0
        obs = build_controller()._compute_observation(JointState(), imu)

        self.assertEqual(obs.shape, (295,)) # Observation Shape
        self.assertTrue(np.all(np.isfinite(obs)))   # Finite values
        np.testing.assert_allclose(obs[15:30], np.tile([0.0, 0.0, -1.0], 5))    # Gravity points down in all 5 frames
        np.testing.assert_array_equal(obs[210:220], np.zeros(10))   # Check that gait clock is off close to 0

    def test_gait_clock_when_walking(self):
        """A walk command opens the clock at the phase the step counter gives."""

        # Step 57 matches the gait clock demo, so the expected values are easy to cross-check.
        # The newest frame is the last 2 numbers of the 10-number gait slice.
        controller = build_controller()
        controller._cmd_vel.linear.x = 0.3
        controller._policy_step = 57
        imu = Imu()
        imu.orientation.w = 1.0
        obs = controller._compute_observation(JointState(), imu)

        phi = 2.0 * np.pi * 57 * 0.02 / 0.476
        np.testing.assert_allclose(obs[218:220], [np.sin(phi), np.cos(phi)])


    def test_leg_joints_found_by_name(self):
        """Joint values land in policy order even when the message lists joints in another order."""

        # Reverse the names and give each joint its message index as its position.
        # The policy order then reads 9 down to 0, which proves the lookup uses names.
        controller = build_controller()
        joint_state = JointState()
        joint_state.name = list(reversed(controller.leg_joints))
        joint_state.position = [float(i) for i in range(10)]
        joint_state.velocity = [0.0] * 10
        imu = Imu()
        imu.orientation.w = 1.0
        obs = controller._compute_observation(joint_state, imu)

        np.testing.assert_array_equal(obs[125:135], np.arange(9.0, -1.0, -1.0))


if __name__ == "__main__":
    unittest.main()
