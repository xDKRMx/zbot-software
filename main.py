# main.py
from core.serial_comm import Arduino
from core.pose_engine import PoseEngine
from core.climber import Climber

arduino = Arduino()
pose_engine = PoseEngine(arduino)
climber = Climber(pose_engine)

print("Z-BOT STARTED")
climber.climb_step()
