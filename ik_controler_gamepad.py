# Copyright 2026 Sachin Kumar.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import ast
import threading
from dataclasses import asdict, dataclass, field

import numpy as np
import rclpy
import zenoh
from builtin_interfaces.msg import Duration
from ikpy.chain import Chain
from rclpy.node import Node
from sensor_msgs.msg import JointState
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

# -------------------------------------------------------------------
# Tuning & Configuration
# -------------------------------------------------------------------
STEP_SIZE = 0.005  # meters per joystick update
MAX_JOINT_DELTA = 0.15  # rad – reject IK solution if any joint jumps more than this
TRAJECTORY_SEC = 2  # seconds for each trajectory point
DEADZONE = 0.1  # ignore joystick drift

ZENOH_ENDPOINT = "tcp/10.147.19.185:7447"
ZENOH_TOPIC = "joy/topic"

# Joint names the ros2_control trajectory controller expects
ARM_JOINT_NAMES = [
    "joint_1",
    "joint_2",
    "joint_3",
    "joint_4",
    "joint_5",
    "joint_6",
]

ACTIVE_LINKS_MASK = [True, True, True, True, True, True, True]


# -------------------------------------------------------------------
# Data Structures
# -------------------------------------------------------------------
def apply_deadzone(value: float):
    return value if abs(value) > DEADZONE else 0.0


@dataclass
class JoyMsg:
    x_axis: float = 0.0
    y_axis: float = 0.0
    z_axis: float = 0.0
    button: int = -1
    hat: tuple = (0, 0)

    @classmethod
    def from_dict(cls, data: dict):
        return cls(**data)


@dataclass
class Pose:
    position: list = field(default_factory=lambda: [0.0, 0.0, 0.0])
    orientation: list = field(default_factory=lambda: [0.0, 0.0, 0.0, 1.0])


# -------------------------------------------------------------------
# Communication Class
# -------------------------------------------------------------------
class ZTransport:
    def __init__(self, joy_callback):
        config = zenoh.Config()
        config.insert_json5("mode", '"client"')
        config.insert_json5("connect/endpoints", f'["{ZENOH_ENDPOINT}"]')

        print(f"Connecting to Zenoh at {ZENOH_ENDPOINT}...")
        self.session = zenoh.open(config)
        self.joy_sub = self.session.declare_subscriber(ZENOH_TOPIC, joy_callback)


# -------------------------------------------------------------------
# Main Controller Node
# -------------------------------------------------------------------
class IkController(Node):
    def __init__(self, urdf_file_path: str):
        super().__init__("ik_controller")

        # 1. Setup IKPy Chain (using your active links mask)
        self.chain = Chain.from_urdf_file(
            urdf_file_path,
            active_links_mask=ACTIVE_LINKS_MASK,
        )

        # 2. ROS 2 Setup
        self._joint_state_sub = self.create_subscription(
            JointState, "joint_states", self._joint_state_cb, 10
        )
        self._joint_traj_pub = self.create_publisher(
            JointTrajectory,
            "/denso_joint_trajectory_controller/joint_trajectory",
            10,
        )

        # 3. State Initialization
        self._chain_len = len(self.chain.links)
        self.last_joint_angles: np.ndarray | None = None
        self.current_pose = Pose()
        self._pose_initialised = False
        self._lock = threading.Lock()

        # 4. Start Zenoh Communication
        self.z_transport = ZTransport(self.joy_topic_cb)
        self.get_logger().info("IK Controller with Zenoh Transport initialized.")

    # --- Zenoh Callback ---
    def joy_topic_cb(self, sample):
        payload_str = sample.payload.to_string()
        try:
            payload_dict = ast.literal_eval(payload_str)
            joy_msg = JoyMsg.from_dict(payload_dict)

            dx = apply_deadzone(joy_msg.x_axis) * STEP_SIZE
            dy = apply_deadzone(joy_msg.y_axis) * STEP_SIZE
            dz = apply_deadzone(joy_msg.z_axis) * STEP_SIZE

            if dx == 0.0 and dy == 0.0 and dz == 0.0:
                return

            with self._lock:
                if not self._pose_initialised or self.last_joint_angles is None:
                    self.get_logger().warn(
                        "Waiting for joint states to initialize pose..."
                    )
                    return

                # Calculate the new target based on current pose + joystick delta
                target = [
                    self.current_pose.position[0] + dx,
                    self.current_pose.position[1] + dy,
                    self.current_pose.position[2] + dz,
                ]
                warm_start = self.last_joint_angles.copy()

            # Execute the IK logic from your keyboard script
            self._go_to_position(target, warm_start)

        except Exception as e:
            self.get_logger().error(f"Failed to process joystick data: {e}")

    # --- IK Logic (Identical to your Keyboard script) ---
    def _joint_state_cb(self, msg: JointState):
        name_to_pos = dict(zip(msg.name, msg.position))
        if not all(name in name_to_pos for name in ARM_JOINT_NAMES):
            return

        angles = np.zeros(self._chain_len)
        for i, name in enumerate(ARM_JOINT_NAMES):
            angles[i + 1] = name_to_pos[name]

        with self._lock:
            self.last_joint_angles = angles

        if not self._pose_initialised:
            self._update_pose_from_fk(angles)

    def _update_pose_from_fk(self, angles: np.ndarray):
        try:
            T = self.chain.forward_kinematics(angles)
            with self._lock:
                self.current_pose.position = T[:3, 3].tolist()
                self.current_pose.orientation = T[:3, :3].flatten().tolist()
                self._pose_initialised = True
            self.get_logger().info(
                f"Pose initialised: {np.round(self.current_pose.position, 4)}"
            )
        except Exception as e:
            self.get_logger().error(f"FK failed: {e}")

    def _go_to_position(self, target_position: list, warm_start: np.ndarray):
        try:
            new_angles = self.chain.inverse_kinematics(
                target_position=target_position,
                initial_position=warm_start,
            )
        except Exception as e:
            self.get_logger().error(f"IK failed: {e}")
            return

        # Reject solution if any arm joint would jump too far
        delta = np.abs(new_angles[1:7] - warm_start[1:7])
        max_delta = float(np.max(delta))
        if max_delta > MAX_JOINT_DELTA:
            self.get_logger().warn(
                f"IK Rejected: jump of {max_delta:.3f} rad > {MAX_JOINT_DELTA}"
            )
            return

        # Update pose for next calculation
        with self._lock:
            self.current_pose.position = target_position

        self._publish_trajectory(new_angles[1:7])

    def _publish_trajectory(self, arm_angles: np.ndarray):
        msg = JointTrajectory()
        msg.joint_names = ARM_JOINT_NAMES
        point = JointTrajectoryPoint()
        point.positions = arm_angles.tolist()
        point.velocities = [0.0] * 6
        point.time_from_start = Duration(sec=TRAJECTORY_SEC, nanosec=0)
        msg.points = [point]
        self._joint_traj_pub.publish(msg)


# -----------------------------------------------------------------------
def main(args=None):
    rclpy.init(args=args)
    urdf_file_path = "/home/asus/zzzzz/ros2/personal_projects/colcon_ws/src/denso_robot_ros2/denso_robot_descriptions/denso_robot.urdf"

    node = IkController(urdf_file_path)

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if hasattr(node, "z_transport"):
            node.z_transport.session.close()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
