# core/climber.py
import time
from config.poses import POSES
from core.pose_engine import PoseEngine

class Climber:
    def __init__(self, pose_engine: PoseEngine):
        self.pe = pose_engine

    def climb_step(self):
        self.pe.apply_pose(POSES["HOLD"])
        time.sleep(1)

        self.pe.apply_pose(POSES["SWING_FL"])
        time.sleep(1)

        self.pe.apply_pose(POSES["HOLD"])
        time.sleep(1)
