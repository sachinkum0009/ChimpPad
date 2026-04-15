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

import logging
import sys

import pygame
import zenoh

from utils import JoyMsg

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    filename="app.log",
)

logger = logging.getLogger(__name__)

# Add a StreamHandler to log to the console
console_handler = logging.StreamHandler(sys.stdout)
console_handler.setLevel(logging.INFO)
console_handler.setFormatter(
    logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
)

# Add the console handler to the logger
logger.addHandler(console_handler)


class ZenohTransport:
    def __init__(self):
        config = zenoh.Config()
        config.insert_json5("mode", '"client"')
        config.insert_json5("connect/endpoints", '["tcp/10.147.19.232:7447"]')
        self.session = zenoh.open(config)
        self.joy_key = "joy/topic"
        self.joy_pub = self.session.declare_publisher(self.joy_key)

    def publish(self, msg):
        buf = f"{msg}"
        self.joy_pub.put(buf)

    def __del__(self):
        if self.session:
            self.session.close()


class XController(ZenohTransport):
    def __init__(self):
        """Initialize the important variables"""
        super().__init__()

        pygame.joystick.init()

        joy_count = pygame.joystick.get_count()
        if joy_count == 0:
            logger.error(
                "Unable to find any joystick. Please make sure it connected..."
            )
            exit(1)
        elif joy_count == 1:
            self.joystick = pygame.joystick.Joystick(0)
        else:
            # List available joysticks
            print("Multiple joysticks detected. Please select one:")
            for i in range(joy_count):
                joystick = pygame.joystick.Joystick(i)
                print(f"{i}: {joystick.get_name()}")

            # Prompt user for selection
            while True:
                try:
                    choice = int(input("Enter the number of the joystick to use: "))
                    if 0 <= choice < joy_count:
                        self.joystick = pygame.joystick.Joystick(choice)
                        break
                    else:
                        print(f"Please enter a number between 0 and {joy_count - 1}.")
                except ValueError:
                    print("Please enter a valid number.")

        self.msg = JoyMsg()

    def monitor_controller(self):
        self.joystick.init()
        try:
            while True:
                for event in pygame.event.get():
                    if event.type == pygame.JOYBUTTONDOWN:
                        self.msg.button = event.button
                        logger.info(f"type {type(self.msg.to_dict())}")
                        self.publish(self.msg.to_dict())
                        logger.info(f"Button {event.button} pressed")
                        # rumble joystick
                        res = self.joystick.rumble(0, 0.7, 500)
                        if res:
                            logger.info("rumbled success")
                        else:
                            logger.info("rumbled failure")
                        self.msg.button = -1  # reset
                    elif event.type == pygame.JOYAXISMOTION:
                        if event.axis == 0:  # X-axis
                            self.msg.x_axis = event.value
                        elif event.axis == 1:  # Y-axis
                            self.msg.y_axis = event.value
                        elif event.axis == 2:  # Z-axis (if applicable)
                            self.msg.z_axis = event.value
                        if abs(event.value) > 0.1:
                            self.publish(self.msg.to_dict())
                            logger.info(f"Axis {event.axis} moved to {event.value:.2f}")
                            # reset values after publising
                            self.msg.x_axis = 0.0
                            self.msg.y_axis = 0.0
                            self.msg.z_axis = 0.0
                    elif event.type == pygame.JOYHATMOTION:
                        self.msg.hat = event.value
                        self.publish(self.msg.to_dict())
                        logger.info(f"D-pad moved to {event.value}")
                        self.msg.hat = (0, 0)  # reset

        except KeyboardInterrupt:
            logger.info("\nStopping...")


def main():
    pygame.init()

    x_controller = XController()
    x_controller.monitor_controller()
    pygame.quit()


if __name__ == "__main__":
    main()
