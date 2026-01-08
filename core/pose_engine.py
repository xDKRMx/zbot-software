# core/pose_engine.py

from config.servos import SERVOS
from core.serial_comm import Arduino

class PoseEngine:
    def __init__(self, arduino: Arduino):
        self.arduino = arduino

    def apply_pose(self, pose: dict):
        for joint, angle in pose.items():
            servo_id = SERVOS[joint]
            self.arduino.send_servo(servo_id, angle)
