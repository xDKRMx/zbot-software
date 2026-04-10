"""Orchestrator Process (GLM & Audio).
Connects to MQTT to receive detection events, aggregating them for GLM processing.
Reads shared images from disk to formulate responses.
"""

import argparse
import json
import sys
import time
from pathlib import Path

src_dir = str(Path(__file__).resolve().parent.parent / "src")
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

try:
    from dotenv import load_dotenv
    env_path = Path(__file__).resolve().parent.parent / ".env"
    load_dotenv(dotenv_path=env_path, override=True)
except ImportError:
    pass

try:
    import paho.mqtt.client as mqtt
except ImportError:
    print("[ERROR] paho-mqtt is required for the orchestrator. Run: pip install paho-mqtt")
    sys.exit(1)

from net_inspector.config import AppConfig
from net_inspector.orchestrator import DetectionEvent, UnifiedOrchestrator

def run_orchestrator(
    broker_ip: str = "127.0.0.1",
    broker_port: int = 1883,
    topic: str = "zbot/vision",
    glm_interval_s: float = 10.0,
    speak: bool = True
):
    print(f"[ORCHESTRATOR_RUNNER] Starting MQTT loop and GLM worker...")
    
    config = AppConfig()
    orchestrator = UnifiedOrchestrator(
        config=config,
        glm_interval_s=glm_interval_s,
        enable_audio_output=speak
    )
    
    # Setup MQTT Subscriber
    def on_connect(client, userdata, flags, rc, properties=None):
        if rc == 0:
            print(f"[ORCHESTRATOR_RUNNER] Connected to MQTT broker at {broker_ip}:{broker_port}")
            client.subscribe(topic)
        else:
            print(f"[ERROR] Failed to connect to MQTT broker, return code {rc}")
            
    def on_message(client, userdata, msg):
        try:
            payload = json.loads(msg.payload.decode())
            # Convert JSON back to DetectionEvent
            if "event_type" in payload and "confidence" in payload:
                event = DetectionEvent(
                    timestamp=payload.get("timestamp", ""),
                    source=payload.get("source", "unknown"),
                    event_type=payload.get("event_type", "UNKNOWN"),
                    confidence=float(payload.get("confidence", 0.0)),
                    metadata=payload.get("metadata", {})
                )
                orchestrator.submit_event(event)
        except Exception as e:
            # Not a valid detection event JSON, ignore
            pass

    client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
    client.on_connect = on_connect
    client.on_message = on_message
    
    connected = False
    print(f"[ORCHESTRATOR_RUNNER] Attempting to connect to MQTT Broker at {broker_ip}:{broker_port}...")
    while not connected:
        try:
            client.connect(broker_ip, broker_port, 60)
            connected = True
        except Exception as e:
            print(f"[WARNING] Cannot connect to MQTT Broker: {e}")
            print(f"Retrying in 5 seconds... (Ensure Mosquitto is running on port {broker_port})")
            time.sleep(5.0)
        
    orchestrator.start()
    
    try:
        client.loop_forever()
    except KeyboardInterrupt:
        print("[ORCHESTRATOR_RUNNER] Interrupted by user.")
    finally:
        client.loop_stop()
        client.disconnect()
        orchestrator.stop()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--broker", type=str, default="127.0.0.1", help="MQTT Broker IP")
    parser.add_argument("--port", type=int, default=1883, help="MQTT Broker Port")
    parser.add_argument("--topic", type=str, default="zbot/vision", help="MQTT Topic")
    parser.add_argument("--glm-interval", type=float, default=10.0, help="Seconds between GLM requests")
    parser.add_argument("--no-speak", action="store_true", help="Disable TTS output")
    args = parser.parse_args()
    
    run_orchestrator(
        broker_ip=args.broker,
        broker_port=args.port,
        topic=args.topic,
        glm_interval_s=args.glm_interval,
        speak=not args.no_speak
    )
