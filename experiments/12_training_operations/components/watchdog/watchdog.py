
import time
import os
import requests
import json
from pathlib import Path

class Watchdog:
    """
    The Active Control Plane.
    Polls Prometheus for critical alerts and enforces PAUSE/HALT actions.
    """
    def __init__(self, 
                 prometheus_url: str = "http://localhost:9090", 
                 control_file_path: str = "/tmp/training_control.flag",
                 poll_interval: int = 5):
        self.prom_url = prometheus_url
        self.control_file = Path(control_file_path)
        self.poll_interval = poll_interval
        self.running = True
        
        print(f"✓ Watchdog Initialized")
        print(f"  Monitoring: {self.prom_url}")
        print(f"  Control File: {self.control_file}")

    def check_alerts(self):
        """
        Query Prometheus AlertManager or Evaluate Rules Locally.
        For POC, we check a simple metric condition directly.
        """
        # Example Rule: Pause if Loss > 10.0 (Divergence)
        query = 'training_loss > 10.0'
        
        try:
            response = requests.get(
                f"{self.prom_url}/api/v1/query",
                params={'query': query}
            )
            data = response.json()
            
            if data['status'] == 'success' and len(data['data']['result']) > 0:
                # Condition Met!
                value = float(data['data']['result'][0]['value'][1])
                print(f"⚠️  CRITICAL ALERT: Loss Divergence Detected (Value: {value:.2f})")
                self.trigger_pause(reason=f"Loss Divergence (Value: {value:.2f})")
            else:
                # All good, ensure we are not unnecessarily paused
                # (Optional: Implement auto-resume or manual-only resume logic)
                pass
                
        except Exception as e:
            print(f"Watchdog Error connecting to Prometheus: {e}")

    def trigger_pause(self, reason: str):
        """
        Write the Control Flag to pause training.
        """
        if not self.control_file.exists():
            with open(self.control_file, "w") as f:
                payload = {
                    "action": "PAUSE",
                    "reason": reason,
                    "timestamp": time.time()
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
