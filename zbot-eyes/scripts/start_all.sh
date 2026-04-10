#!/bin/bash
# Z-Bot Vision Auto-Starter for Raspberry Pi

# Navigate to project root
cd "$(dirname "$0")/.."

# Activate virtual environment if it exists
if [ -d ".venv" ]; then
    source .venv/bin/activate
fi

echo "Starting Z-Bot Vision Subsystems..."

# Ensure outputs/shared exists
mkdir -p outputs/shared

# Run everything in the background, logging to outputs/
python scripts/run_webcam.py --camera 0 --fps 5.0 > outputs/webcam.log 2>&1 &
WEBCAM_PID=$!

python scripts/run_thermal.py --camera 1 --fps 2.0 > outputs/thermal.log 2>&1 &
THERMAL_PID=$!

# Give cameras a second to initialize and emit events
sleep 2

python scripts/run_orchestrator.py --glm-interval 10.0 > outputs/orchestrator.log 2>&1 &
ORCH_PID=$!

echo "Subsystems started. WebCam PID: $WEBCAM_PID, Thermal PID: $THERMAL_PID, Orchestrator PID: $ORCH_PID"

# Function to gracefully shutdown
cleanup() {
    echo "Shutting down all subsystems..."
    kill $WEBCAM_PID
    kill $THERMAL_PID
    kill $ORCH_PID
    exit 0
}

# Trap termination signals
trap cleanup SIGINT SIGTERM

# Wait indefinitely (until killed)
wait
