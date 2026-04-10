"""
Mock Robot Localization API for testing world coordinate calculation.

This simulates a robot that moves in a simple pattern and provides its position.
Run this on your PC, then point infer_rpi.py to http://localhost:5000/robot/position

Usage:
    python mock_robot_api.py

Then in another terminal:
    python src/infer_rpi.py --robot-api-url http://localhost:5000/robot/position --show
"""

import time
import math
from flask import Flask, jsonify

app = Flask(__name__)

# Simulated robot state
robot_state = {
    "x": 0.0,  # meters (or can be lat/lon)
    "y": 0.0,  # meters
    "heading": 0.0,  # degrees (0=North, 90=East, 180=South, 270=West)
    "start_time": time.time(),
}


@app.route("/robot/position", methods=["GET"])
def get_robot_position():
    """Return current robot position and heading."""
    # Simulate robot moving in a circle
    elapsed = time.time() - robot_state["start_time"]
    
    # Simple circular motion: radius=5m, period=60s
    radius = 5.0
    period = 60.0
    angle = (elapsed / period) * 360.0  # degrees
    angle_rad = math.radians(angle)
    
    robot_state["x"] = radius * math.cos(angle_rad)
    robot_state["y"] = radius * math.sin(angle_rad)
    robot_state["heading"] = (angle + 90.0) % 360.0  # tangent to circle
    
    return jsonify({
        "x": round(robot_state["x"], 3),
        "y": round(robot_state["y"], 3),
        "heading": round(robot_state["heading"], 1),
        "timestamp": time.time(),
    })


@app.route("/robot/position/static", methods=["GET"])
def get_static_position():
    """Return a fixed position for testing."""
    return jsonify({
        "x": 10.0,
        "y": 5.0,
        "heading": 45.0,
        "timestamp": time.time(),
    })


if __name__ == "__main__":
    print("=" * 60)
    print("Mock Robot Localization API")
    print("=" * 60)
    print("Endpoints:")
    print("  GET /robot/position        - Simulated moving robot")
    print("  GET /robot/position/static - Fixed position (x=10, y=5, heading=45)")
    print()
    print("Test with:")
    print("  curl http://localhost:5000/robot/position")
    print()
    print("Use with infer_rpi.py:")
    print("  python src/infer_rpi.py --robot-api-url http://localhost:5000/robot/position --show")
    print("=" * 60)
    app.run(host="0.0.0.0", port=5000, debug=False)
