import json
import time
from pathlib import Path

import requests


class Watchdog:
    """
    The Active Control Plane.
    Polls the custom metrics server for critical alerts and enforces PAUSE/HALT actions.
    """

    def __init__(
        self,
        metrics_url: str = "http://localhost:8000",
        control_file_path: str = "/tmp/training_control.flag",
        poll_interval: int = 5,
        loss_threshold: float = 10.0,
    ):
        self.metrics_url = metrics_url
        self.control_file = Path(control_file_path)
        self.poll_interval = poll_interval
        self.loss_threshold = loss_threshold
        self.running = True

        print("✓ Watchdog Initialized")
        print(f"  Monitoring: {self.metrics_url}")
        print(f"  Control File: {self.control_file}")
        print(f"  Loss Threshold: {self.loss_threshold}")

    def check_alerts(self):
        """
        Query the custom metrics server and evaluate alert rules locally.
        """
        try:
            response = requests.get(
                f"{self.metrics_url}/query", params={"metric": "training_loss"}
            )
            data = response.json()

            if "value" in data:
                value = float(data["value"])
                if value > self.loss_threshold:
                    print(
                        f"⚠️  CRITICAL ALERT: Loss Divergence Detected (Value: {value:.2f})"
                    )
                    self.trigger_pause(reason=f"Loss Divergence (Value: {value:.2f})")

        except Exception as e:
            print(f"Watchdog Error connecting to metrics server: {e}")

    def trigger_pause(self, reason: str):
        """
        Write the Control Flag to pause training.
        """
        if not self.control_file.exists():
            with open(self.control_file, "w") as f:
                payload = {
                    "action": "PAUSE",
                    "reason": reason,
                    "timestamp": time.time(),
                }
                json.dump(payload, f)
            print(f"⛔ PAUSE TRIGGERED: {reason}")
            print(f"   Control flag written to {self.control_file}")

    def run(self):
        print("Watchdog Service Running...")
        try:
            while self.running:
                self.check_alerts()
                time.sleep(self.poll_interval)
        except KeyboardInterrupt:
            print("Watchdog Stopped.")


if __name__ == "__main__":
    # For testing, we can run this standalone
    wd = Watchdog()
    wd.run()
