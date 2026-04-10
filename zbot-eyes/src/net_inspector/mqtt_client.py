"""Simple MQTT publisher for Net Inspector."""

import json
import logging
from typing import Dict, Any, Optional

try:
    import paho.mqtt.client as mqtt
except ImportError:
    mqtt = None

logger = logging.getLogger(__name__)

class MessagePublisher:
    """Publishes detection results to an MQTT broker and logs to stdout."""

    def __init__(
        self,
        broker_ip: str = "127.0.0.1",
        broker_port: int = 1883,
        topic: str = "zbot/vision/detections",
        client_id: str = "net_inspector_pi"
    ) -> None:
        self.broker_ip = broker_ip
        self.broker_port = broker_port
        self.topic = topic
        self.client_id = client_id
        
        self.mqtt_available = mqtt is not None
        self.client: Optional[mqtt.Client] = None
        
        if self.mqtt_available:
            self.client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2, client_id=self.client_id)
            try:
                self.client.connect(self.broker_ip, self.broker_port, 60)
                self.client.loop_start()
                logger.info(f"Connected to MQTT broker at {broker_ip}:{broker_port}")
            except Exception as e:
                logger.warning(f"Failed to connect to MQTT broker: {e}. Falling back to stdout logging only.")
                self.client = None
        else:
            logger.warning("paho-mqtt not installed. Falling back to stdout logging only. (pip install paho-mqtt)")

    def publish(self, payload: Dict[str, Any]) -> None:
        """Publish a dictionary payload as JSON."""
        json_payload = json.dumps(payload)
        
        # Always log to stdout (useful for running headless via systemctl/journalctl)
        logger.info(f"OUT: {json_payload}")

        # Publish to MQTT if available
        if self.client:
            try:
                self.client.publish(self.topic, json_payload)
            except Exception as e:
                logger.error(f"Failed to publish MQTT message: {e}")

    def close(self) -> None:
        """Clean up the connection."""
        if self.client:
            self.client.loop_stop()
            self.client.disconnect()
