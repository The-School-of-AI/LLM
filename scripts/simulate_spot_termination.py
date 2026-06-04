#!/usr/bin/env python3
"""
Simulate EC2 spot termination notice for testing checkpoint behavior.

This starts a local HTTP server on port 8111 that mimics the EC2 instance
metadata endpoint. Point SpotTerminationListener at this URL to test.

Usage:
    # Terminal 1: Start the simulator
    python scripts/simulate_spot_termination.py

    # Terminal 2: Trigger termination (after training is running)
    curl -X POST http://localhost:8111/trigger

    # Or with a delay (simulates "termination in 120s"):
    curl -X POST http://localhost:8111/trigger?delay=120

    # Check status:
    curl http://localhost:8111/status

    # Reset (clear termination notice):
    curl -X POST http://localhost:8111/reset

To use with training, set the environment variable:
    export SPOT_METADATA_URL=http://localhost:8111/latest/meta-data/spot/instance-action
"""

import argparse
import json
import threading
import time
from datetime import datetime, timedelta, timezone
from http.server import BaseHTTPRequestHandler, HTTPServer


class SpotSimulatorState:
    def __init__(self):
        self.termination_active = False
        self.termination_time = None
        self.lock = threading.Lock()

    def trigger(self, delay_seconds: int = 120):
        with self.lock:
            self.termination_active = True
            self.termination_time = (
                datetime.now(timezone.utc) + timedelta(seconds=delay_seconds)
            ).strftime("%Y-%m-%dT%H:%M:%SZ")
            print(f"[SpotSim] Termination triggered! Time: {self.termination_time}")

    def reset(self):
        with self.lock:
            self.termination_active = False
            self.termination_time = None
            print("[SpotSim] Termination notice cleared.")

    def get_response(self):
        with self.lock:
            if self.termination_active:
                return {"action": "terminate", "time": self.termination_time}
            return None


state = SpotSimulatorState()


class SpotSimHandler(BaseHTTPRequestHandler):
    def log_message(self, format, *args):
        # Suppress default logging
        pass

    def do_GET(self):
        if self.path.startswith("/latest/meta-data/spot/instance-action"):
            resp = state.get_response()
            if resp is None:
                self.send_response(404)
                self.end_headers()
                self.wfile.write(b"Not Found")
            else:
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps(resp).encode())

        elif self.path == "/status":
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            resp = state.get_response()
            status = {
                "termination_active": state.termination_active,
                "termination_time": state.termination_time,
                "response": resp,
            }
            self.wfile.write(json.dumps(status, indent=2).encode())

        # IMDSv2 token endpoint (always succeed)
        elif self.path.startswith("/latest/api/token"):
            self.send_response(200)
            self.end_headers()
            self.wfile.write(b"fake-imds-token")
        else:
            self.send_response(404)
            self.end_headers()

    def do_PUT(self):
        # IMDSv2 token request
        if self.path.startswith("/latest/api/token"):
            self.send_response(200)
            self.end_headers()
            self.wfile.write(b"fake-imds-token")
        else:
            self.send_response(404)
            self.end_headers()

    def do_POST(self):
        if self.path.startswith("/trigger"):
            delay = 120
            if "?" in self.path:
                params = dict(
                    p.split("=") for p in self.path.split("?")[1].split("&") if "=" in p
                )
                delay = int(params.get("delay", 120))
            state.trigger(delay_seconds=delay)
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(
                json.dumps(
                    {
                        "status": "triggered",
                        "termination_time": state.termination_time,
                        "delay_seconds": delay,
                    }
                ).encode()
            )

        elif self.path == "/reset":
            state.reset()
            self.send_response(200)
            self.end_headers()
            self.wfile.write(b'{"status": "reset"}')
        else:
            self.send_response(404)
            self.end_headers()


def main():
    parser = argparse.ArgumentParser(description="EC2 Spot Termination Simulator")
    parser.add_argument("--port", type=int, default=8111, help="Port to listen on")
    parser.add_argument(
        "--auto-trigger",
        type=int,
        default=0,
        help="Auto-trigger termination after N seconds (0=disabled)",
    )
    args = parser.parse_args()

    server = HTTPServer(("0.0.0.0", args.port), SpotSimHandler)
    print(f"[SpotSim] Listening on http://0.0.0.0:{args.port}")
    print(
        f"[SpotSim] Metadata URL: http://localhost:{args.port}/latest/meta-data/spot/instance-action"
    )
    print(f"[SpotSim] Trigger:  curl -X POST http://localhost:{args.port}/trigger")
    print(f"[SpotSim] Status:   curl http://localhost:{args.port}/status")
    print(f"[SpotSim] Reset:    curl -X POST http://localhost:{args.port}/reset")

    if args.auto_trigger > 0:

        def _auto():
            time.sleep(args.auto_trigger)
            state.trigger()

        t = threading.Thread(target=_auto, daemon=True)
        t.start()
        print(f"[SpotSim] Auto-trigger in {args.auto_trigger}s")

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n[SpotSim] Shutting down.")
        server.shutdown()


if __name__ == "__main__":
    main()
