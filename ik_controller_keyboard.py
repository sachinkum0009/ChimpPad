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

import sys
import termios
import threading
import tty
from dataclasses import dataclass, field

import numpy as np
import rclpy
from builtin_interfaces.msg import Duration
from ikpy.chain import Chain
from rclpy.node import Node
from sensor_msgs.msg import JointState
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

# -------------------------------------------------------------------
# Tuning
# -------------------------------------------------------------------
STEP_SIZE = 0.005  # metres per keypress
MAX_JOINT_DELTA = 0.15  # rad – reject IK solution if any joint jumps more than this
TRAJECTORY_SEC = 2  # seconds for each trajectory point

KEY_BINDINGS = {
    "w": (1, 0, 0),  # +X (Forward)
    "s": (-1, 0, 0),  # -X (Back)
    "a": (0, 1, 0),  # +Y (Left)
    "d": (0, -1, 0),  # -Y (Right)
    " ": (0, 0, 1),  # +Z (Up)
    "x": (0, 0, -1),  # -Z (Down)
    "q": None,
}

HELP_TEXT = """
Keyboard Controls:
------------------
  W / S  →  +X / -X  (forward / back)
  A / D  →  +Y / -Y  (left / right)
  Space  →  +Z        (up)
  X      →  -Z        (down)
  Q      →  quit
------------------
"""

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


def get_key() -> str:
    """Read one raw keypress from stdin (no Enter needed)."""
    fd = sys.stdin.fileno()
    old = termios.tcgetattr(fd)
    try:
        tty.setraw(fd)
        ch = sys.stdin.read(1)
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old)
    return ch


@dataclass
class Pose:
    position: list = field(default_factory=lambda: [0.0, 0.0, 0.0])
    orientation: list = field(default_factory=lambda: [0.0, 0.0, 0.0, 1.0])


class IkController(Node):
    def __init__(self, urdf_file_path: str):
        super().__init__("ik_controller")

        # Build chain – finger link is passive so IK ignores it
        self.chain = Chain.from_urdf_file(
            urdf_file_path,
            active_links_mask=ACTIVE_LINKS_MASK,
        )
        self.get_logger().info("IKPy chain links:")
        for i, lnk in enumerate(self.chain.links):
            self.get_logger().info(f"  [{i}] {lnk.name}  active={ACTIVE_LINKS_MASK[i]}")

        self._joint_state_sub = self.create_subscription(
            JointState, "joint_states", self._joint_state_cb, 10
        )
        self._joint_traj_pub = self.create_publisher(
            JointTrajectory,
            "/denso_joint_trajectory_controller/joint_trajectory",
            10,
        )

        self._chain_len = len(self.chain.links)  # 8
        self.last_joint_angles: np.ndarray | None = None  # length 8
        self.current_pose = Pose()
        self._pose_initialised = False
        self._lock = threading.Lock()

        print(HELP_TEXT)
        threading.Thread(target=self._keyboard_loop, daemon=True).start()

    # ------------------------------------------------------------------
    # Joint state callback
    # ------------------------------------------------------------------
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

    # ------------------------------------------------------------------
    # FK → initialise current_pose once
    # ------------------------------------------------------------------
    def _update_pose_from_fk(self, angles: np.ndarray):
        try:
            T = self.chain.forward_kinematics(angles)
            with self._lock:
                self.current_pose.position = T[:3, 3].tolist()
                self.current_pose.orientation = T[:3, :3].flatten().tolist()
                self._pose_initialised = True
            self.get_logger().info(
                f"Pose initialised from FK: "
                f"{[f'{v:.4f}' for v in self.current_pose.position]}"
            )
        except Exception as e:
            self.get_logger().error(f"FK failed: {e}")

    # ------------------------------------------------------------------
    # Keyboard loop (background thread)
    # ------------------------------------------------------------------
    def _keyboard_loop(self):
        while rclpy.ok():
            key = get_key().lower()

            if key == "q":
                self.get_logger().info("Quit key pressed — shutting down.")
                rclpy.shutdown()
                break

            direction = KEY_BINDINGS.get(key)
            if direction is None:
                continue

            with self._lock:
                if not self._pose_initialised or self.last_joint_angles is None:
                    self.get_logger().warn(
                        "Pose not yet initialised — waiting for joint states."
                    )
                    continue

                target = [
                    self.current_pose.position[0] + direction[0] * STEP_SIZE,
                    self.current_pose.position[1] + direction[1] * STEP_SIZE,
                    self.current_pose.position[2] + direction[2] * STEP_SIZE,
                ]
                warm_start = self.last_joint_angles.copy()

            self.get_logger().info(
                f"Key '{key}' → target  "
                f"x:{target[0]:.4f}  y:{target[1]:.4f}  z:{target[2]:.4f}"
            )
            self._go_to_position(target, warm_start)

    # ------------------------------------------------------------------
    # IK → safety check → publish
    # ------------------------------------------------------------------
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
                f"IK solution rejected — max joint delta {max_delta:.3f} rad "
                f"> limit {MAX_JOINT_DELTA:.3f} rad."
            )
            return

        self.get_logger().info(
            f"Joint angles (arm): {np.round(new_angles[1:7], 3).tolist()}"
        )

        # Accept → update warm-start and pose for next keypress
        with self._lock:
            # self.last_joint_angles = new_angles
            self.current_pose.position = target_position

        self._publish_trajectory(new_angles[1:7])

    # ------------------------------------------------------------------
    # Publish trajectory
    # ------------------------------------------------------------------
    def _publish_trajectory(self, arm_angles: np.ndarray):
        msg = JointTrajectory()
        msg.joint_names = ARM_JOINT_NAMES

        point = JointTrajectoryPoint()
        point.positions = arm_angles.tolist()
        point.velocities = [0.0] * 6
        point.time_from_start = Duration(sec=TRAJECTORY_SEC, nanosec=0)
        msg.points = [point]
        self.get_logger().info(
            f"Current Joints -> {np.round(self.last_joint_angles, 4).tolist()}"
        )

        self.get_logger().info(f"Publishing → {np.round(arm_angles, 4).tolist()}")
        self._joint_traj_pub.publish(msg)


# -----------------------------------------------------------------------
def main(args=None):
    rclpy.init(args=args)
    # urdf_file_path = "/home/asus/zzzzz/ros2/personal_projects/denso_ws2/denso.urdf"
    urdf_file_path = "/home/asus/zzzzz/ros2/personal_projects/colcon_ws/src/denso_robot_ros2/denso_robot_descriptions/denso_robot.urdf"
    # denso_chain = Chain.from_urdf_file(urdf_file_path)
    # Get the names of all links
    # link_names = [link.name for link in denso_chain.links]
    # print("Links:", link_names)

    # Get the names of all joints
    # joint_names = [joint.name for joint in denso_chain.joints]
    # print("Joints:", joint_names)
    # print(denso_chain.get_link_names())
    # urdf_file_path = "/home/asus/zzzzz/ros2/personal_projects/denso_ws2/denso.urdf"
    node = IkController(urdf_file_path)
    rclpy.spin(node)
    rclpy.shutdown()


if __name__ == "__main__":
    main()
