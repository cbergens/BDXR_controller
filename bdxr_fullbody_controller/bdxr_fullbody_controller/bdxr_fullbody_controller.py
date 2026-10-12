#!/usr/bin/env python3

# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# NumPy does the math and torch runs the trained policy.
# message_filters pairs up joint and IMU messages that share a timestamp.
import io
import os
import time

import numpy as np
import rclpy
import torch
from collections import deque
from geometry_msgs.msg import Twist
from message_filters import Subscriber, TimeSynchronizer
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from std_msgs.msg import Float32MultiArray
from sensor_msgs.msg import Imu, JointState
from ament_index_python.packages import get_package_share_directory


class BDXRFullbodyController(Node):
    """Fullbody controller for BDX-R humanoid robot.

    This ROS 2 node subscribes to velocity commands and synchronized joint/IMU
    data, processes the data through a neural network policy, and publishes
    joint commands for controlling the BDX-R robot's movements.
    """

    # The joint order the trained policy expects. Do not reorder this list.
    # It must line up with default_pos and with the 14 numbers the network outputs.
    # These lists are outside the __init__ call so test files can access them.
    
    # NOTE: Order Sensitive - expected by trained policy
    leg_joints = [
        "Left_Hip_Yaw",
        "Right_Hip_Yaw",
        "Left_Hip_Roll",
        "Right_Hip_Roll",
        "Left_Hip_Pitch",
        "Right_Hip_Pitch",
        "Left_Knee",
        "Right_Knee",
        "Left_Ankle",
        "Right_Ankle",
    ]

    # NOTE: Order Sensitive - expected by trained policy
    head_joints = [
        "Neck_Pitch",
        "Head_Pitch",
        "Head_Yaw",
        "Head_Roll",
    ]

    # NOTE: Order Sensitive - expected by trained policy
    obs_names = [
        "imu_ang_vel_b",
        "gravity_b",
        "head_joint_pos",
        "head_joint_vel",
        "head_pos_cmd",
        "leg_joint_pos",
        "leg_joint_vel",
        "feet_contact",
        "vel_cmd",
        "gait_phase",
        "base_mass",
        "actions"
    ]

    joint_names = leg_joints + head_joints


    def __init__(self):
        """Initialize the BDX-R fullbody controller node."""

        super().__init__("bdxr_fullbody_controller")

        # Get the installed copy of the policy using the ament index
        default_policy_path = os.path.join(
            get_package_share_directory("bdxr_fullbody_controller"), "policy", "bdxr_policy.pt"
        )

        # ROS parameters instead of constants in the code.
        # The launch file can point at a different policy file without an edit here.
        self.declare_parameter("foot_contact_force_threshold", 1.0)
        self.declare_parameter("publish_period_ms", 5)
        self.declare_parameter("policy_path", default_policy_path)
        self.declare_parameter("base_mass", 7.113365)   #kg

        self._logger = self.get_logger()

        # QoS (quality of service) settings must match the Isaac Sim side.
        # If they do not match, ROS 2 drops the messages and this node hears nothing.
        sim_qos_profile = rclpy.qos.QoSProfile(
            reliability=rclpy.qos.ReliabilityPolicy.RELIABLE,
            durability=rclpy.qos.DurabilityPolicy.VOLATILE,
            history=rclpy.qos.HistoryPolicy.KEEP_ALL,
        )

        # The walk command arrives here, from teleop or from a navigation stack.
        # A queue depth of 10 is the normal setting for cmd_vel.
        self._cmd_vel_subscription = self.create_subscription(Twist, "cmd_vel", self._cmd_vel_callback, qos_profile=10)

        # Subscriber to foot contacts topics from isaac sim action graph
        self._foot_force_subscription = self.create_subscription(Float32MultiArray, "foot_forces", self._foot_force_callback, qos_profile=qos_profile_sensor_data)

        # Joint targets go out on this topic, Isaac Sim reads it and moves the robot.
        self._joint_publisher = self.create_publisher(JointState, "joint_command", qos_profile=sim_qos_profile)

        # Joint and IMU messages arrive at slightly different times.
        # These wrappers hold each message until a matching one shows up.
        self._imu_sub_filter = Subscriber(
            self,
            Imu,
            "imu",
            qos_profile=sim_qos_profile,
        )
        self._joint_states_sub_filter = Subscriber(
            self,
            JointState,
            "joint_states",
            qos_profile=sim_qos_profile,
        )
        queue_size = 10
        subscribers = [self._joint_states_sub_filter, self._imu_sub_filter]

        # Observation history of 5 for policy input. Initialize
        # observation term history dict with no frames.
        self._history_length = 5
        self._obs_histories = {}

        # Foot contact sensor history window. Initialize with infinite force on the
        # contact sensors to ensure they begin reading contact. The full deque shall be full.
        self.foot_force_window = deque([[float('inf'), float('inf')]] * 3, maxlen=3)

        # _tick runs once for each matched pair of messages.
        # The control loop therefore follows the simulator clock, not a wall clock timer.
        self.sync = TimeSynchronizer(subscribers, queue_size)
        self.sync.registerCallback(self._tick)

        # Load the trained network once, at startup.
        # Loading it later will add a delay in the middle of the control loop.
        self.policy_path = self.get_parameter("policy_path").value
        self.load_policy()

        # State that the node carries from one control step to the next.
        # The messages are reused, so each step does not build new objects.
        self._joint_state = JointState()
        self._joint_command = JointState()
        self._cmd_vel = Twist()
        self._imu = Imu()
        self._action_scale = 0.25  # Scale factor for policy output
        self._previous_action = np.zeros(14)
        self.action = np.zeros(14)
        self._policy_period = 0.02
        self._gait_period = 0.476   # Do not change gait parameters, these are trained into the control policy
        self._gait_cmd_thresh = 0.05
        self._policy_step = 0
        self._last_tick_time = self.get_clock().now().nanoseconds * 1e-9
        self._decimation = 4
        self._tick_count = 0
        self._foot_contact_threshold = self.get_parameter("foot_contact_force_threshold").value
        self._base_mass = self.get_parameter("base_mass").value
        # TODO: Implement head control command observation
        self._head_pose_cmd = np.zeros(3)

        if not 6.648365 <= self._base_mass <= 8.975365:
            base_mass_warn_msg = f"Robot base mass ({self._base_mass} kg) is outside the trained range of 6.648365 to 8.975365 kg"
            self._logger.warning(base_mass_warn_msg)
            print(f"WARN: {base_mass_warn_msg}")

        # The standing pose the policy was trained around, in radians.
        # The network outputs offsets from this pose, so a zero output means stand still.
        self.default_pos = np.zeros(14)

        self._logger.info("Initializing BDXRFullbodyController")

    def _cmd_vel_callback(self, msg):
        """Store the latest velocity command."""

        # Save the command and nothing else. The control step reads it later.
        # A fast or slow command publisher therefore does not change the control rate.
        self._cmd_vel = msg

    def _foot_force_callback(self, msg):
        """Store the latest foot contact sensor data."""

        # Guard misaligned messages
        if len(msg.data) != 2:
            self._logger.warning(f"foot_forces has {len(msg.data)} entries, expected 2")
            return

        # Append, process the window on each policy update.
        self.foot_force_window.append(np.array(msg.data, dtype = np.float64))

    def _history_update(self, name: str, value: list) -> np.ndarray:
        """Update the observation history for a given term and return the flattened history."""

        new_frame = np.array(value, dtype=np.float64)
        obs_history = self._obs_histories.get(name)

        # In the case of no history, initialize the obs term history
        # with a history full of the initial frame in the same way
        # isaaclab does
        if obs_history is None:
            obs_history = deque([new_frame] * self._history_length, maxlen = self._history_length)
            self._obs_histories[name] = obs_history

        # Otherwise, just append the new and drop the oldest history
        else:
            obs_history.append(new_frame)

        # Flatten the history and return
        return np.concatenate(obs_history)
        

    def _tick(self, joint_state: JointState, imu: Imu):
        """Process synchronized joint state and IMU data to generate robot commands."""

        # The clock runs backwards when someone resets the simulation.
        now = self.get_clock().now().nanoseconds * 1e-9
        if now < self._last_tick_time:
            self._logger.error(f"{self._get_stamp_prefix()} Time jumped backwards. Resetting.")
            self._tick_count = 0
            self._policy_step = 0
            self._last_tick_time = now
            self._previous_action = np.zeros(14)
            self._obs_histories = {}
            self.action = np.zeros(14)
            return

        # Store time for reset check
        self._last_tick_time = now

        # Force policy and hardware to step in sync
        if self._tick_count % self._decimation == 0:
            # Build the observation and run the policy
            self.forward(joint_state, imu)
        self._tick_count += 1

        

        # Fill in the timestamp and the joint names.
        # A JointState message has no fixed joint order, so the names travel with the numbers.
        self._joint_command.header.stamp = self.get_clock().now().to_msg()
        self._joint_command.name = self.joint_names

        # Joint angle = standing pose + scaled policy output.
        # Velocity and effort stay zero because the robot runs in position mode.
        action_pos = self.default_pos + self.action * self._action_scale
        self._joint_command.position = action_pos.tolist()
        self._joint_command.velocity = np.zeros(len(self.joint_names)).tolist()
        self._joint_command.effort = np.zeros(len(self.joint_names)).tolist()
        self._joint_publisher.publish(self._joint_command)

    def _compute_observation(self, joint_state: JointState, imu: Imu):
        """Compute the policy observation vector from robot state."""

        ang_vel_b = np.array([imu.angular_velocity.x, imu.angular_velocity.y, imu.angular_velocity.z])
        self._history_update(self.obs_names[0], ang_vel_b)    # Update history
        """Base angular velocity"""

        # B to I quat
        quat_I = imu.orientation
        quat_array = np.array([quat_I.w, quat_I.x, quat_I.y, quat_I.z])
        # Convert to rotation matrix
        R_IB = self.quat_to_rot_matrix(quat_array)
        # Take gravity vector
        gravity_b = -(R_IB[2, :])
        self._history_update(self.obs_names[1], gravity_b)    # Update history
        """Base-frame gravity unit vector"""

        # Initialize head_joint_pos and head_joint_vel
        head_joint_pos = np.zeros(4)
        head_joint_vel = np.zeros(4)
        
        # Search for each joint in the joint_state message and grab
        # position and velocity for each
        for i, name in enumerate(self.head_joints):
            if name in joint_state.name:
                idx = joint_state.name.index(name)
                head_joint_pos[i] = joint_state.position[idx]
                head_joint_vel[i] = joint_state.velocity[idx]  

        self._history_update(self.obs_names[2], head_joint_pos)  # Update History
        self._history_update(self.obs_names[3], head_joint_vel)
        """Head joint positions and velocities, ordered for policy"""

        # TODO: Implement head_pose_cmd observation term rather than current zeros
        # TODO: Change to roll, pitch, yaw ordering (policy side)
        self._history_update(self.obs_names[4], self._head_pose_cmd)  # Update history
        """Head pose command (pitch, roll, yaw)"""

        # Initialize leg_joint_pos and leg_joint_vel
        leg_joint_pos = np.zeros(10)
        leg_joint_vel = np.zeros(10)

        for i, name in enumerate(self.leg_joints):
            if name in joint_state.name:
                idx = joint_state.name.index(name)
                leg_joint_pos[i] = joint_state.position[idx]
                leg_joint_vel[i] = joint_state.velocity[idx]

        self._history_update(self.obs_names[5], leg_joint_pos)  # Update History
        self._history_update(self.obs_names[6], leg_joint_vel)
        """Leg joint positions and velocities, ordered for policy"""

        feet_contact = (np.max(self.foot_force_window, axis=0) 
                        > self._foot_contact_threshold).astype(np.float64)
        self._history_update(self.obs_names[7], feet_contact)   # Update History
        """Feet contact"""

        vel_cmd = np.array([self._cmd_vel.linear.x, self._cmd_vel.linear.y, self._cmd_vel.angular.z])
        self._history_update(self.obs_names[8], vel_cmd)  # Update History
        """Velocity commands"""


        sim_t = self._policy_step * self._policy_period
        phi = 2.0 * np.pi * sim_t / self._gait_period

        # Use cos to differentiate similar late phase and early phase sin values.
        # helpful for the neural net.
        gait_phase = np.array([np.sin(phi), np.cos(phi)])

        # Silence the clock if there is a small movement command
        if np.hypot(vel_cmd[0], vel_cmd[1]) <= self._gait_cmd_thresh:
            gait_phase[:] = 0.0

        self._history_update(self.obs_names[9], gait_phase)
        """Gait Phase"""

        self._history_update(self.obs_names[10], [self._base_mass])
        """Base Mass"""

        self._history_update(self.obs_names[11], self._previous_action)
        """Actions"""

        # NOTE: Order sensitive
        obs = np.concatenate([np.concatenate(self._obs_histories[name]) for name in self.obs_names])
        assert obs.shape == (295,), f"Observation is {obs.shape}, but expected to be (295,)"
        return obs

    def _compute_action(self, obs):
        """Run the neural network policy to compute an action from the observation."""

        # Run the network on one observation.
        # no_grad turns off the training bookkeeping, which costs time and does nothing here.
        with torch.no_grad():
            obs = torch.from_numpy(obs).view(1, -1).float()
            action = self.policy(obs).detach().view(-1).numpy()
        return action

    def forward(self, joint_state: JointState, imu: Imu):
        """Process sensor data and compute control actions."""

        obs = self._compute_observation(joint_state, imu)
        self.action = self._compute_action(obs)
        self._previous_action = self.action.copy()
        self._policy_step += 1  # Advance the gait clock

    def quat_to_rot_matrix(self, quat: np.ndarray) -> np.ndarray:
        """Convert input quaternion to rotation matrix."""

        # An all-zero quaternion points nowhere, and one arrives before the IMU starts up.
        # Return "no rotation" instead of dividing by zero.
        q = np.array(quat, dtype=np.float64, copy=True)
        nq = np.dot(q, q)
        if nq < 1e-10:
            return np.identity(3)

        # The standard formula that turns a quaternion into a 3x3 rotation matrix.
        # It is written out here so this node does not need a transform library.
        q *= np.sqrt(2.0 / nq)
        q = np.outer(q, q)
        return np.array(
            (
                (1.0 - q[2, 2] - q[3, 3], q[1, 2] - q[3, 0], q[1, 3] + q[2, 0]),
                (q[1, 2] + q[3, 0], 1.0 - q[1, 1] - q[3, 3], q[2, 3] - q[1, 0]),
                (q[1, 3] - q[2, 0], q[2, 3] + q[1, 0], 1.0 - q[1, 1] - q[2, 2]),
            ),
            dtype=np.float64,
        )

    def load_policy(self):
        """Load the neural network policy from the specified path."""

        # Read the file into memory, then hand the bytes to torch.
        # The file closes right away instead of staying open while the node runs.
        with open(self.policy_path, "rb") as f:
            buffer = io.BytesIO(f.read())
        self.policy = torch.jit.load(buffer)

    def _get_stamp_prefix(self) -> str:
        """Create a timestamp prefix for logging."""

        # Print wall clock time and simulator time together.
        # The gap between the two shows when the simulator pauses or resets.
        now = time.time()
        now_ros = self.get_clock().now().nanoseconds / 1e9
        return f"[{now}][{now_ros}]"

    def header_time_in_seconds(self, header) -> float:
        """Convert a ROS message header timestamp to seconds."""

        # A ROS header splits time into seconds and nanoseconds.
        # Join them into one float so timestamps are easy to compare.
        return header.stamp.sec + header.stamp.nanosec * 1e-9


def main(args=None):
    """Main function to initialize and run the BDXR fullbody controller node."""

    # The standard ROS 2 startup: start up, spin, then shut down.
    # spin blocks here because all the work happens in the callbacks.
    rclpy.init(args=args)
    node = BDXRFullbodyController()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
