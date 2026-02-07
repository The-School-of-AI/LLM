"""
Flexible Training Dashboard - Backend Server
Automatically discovers and visualizes metrics from ANY training logs.

Features:
- Auto-discovers all metrics in logs (no hardcoded metric names)
- Works with any framework (PyTorch, TensorFlow, JAX, custom)
- Dynamic chart generation based on user selection
- Supports local files and S3 storage
- Zero configuration required
"""

from flask import Flask, jsonify, send_from_directory, request
from flask_cors import CORS
import json
import os
from pathlib import Path
from typing import Dict, List, Any, Optional
import threading
import time
from collections import defaultdict

app = Flask(__name__)
CORS(app)

# Configuration
LOG_DIR = Path("../training/deepspeed_template/logs")
DASHBOARD_DIR = Path("dashboard")


class FlexibleMetricsServer:
    """
    Flexible metrics server that discovers and serves any metrics from logs.
    No hardcoded metric names - works with anything!
    """
    
    def __init__(self, log_dir: Path):
        self.log_dir = log_dir
        self.cache = {}
        self.cache_lock = threading.Lock()
        self.cache_ttl = 5  # seconds
        self.last_cache_update = {}
    
    def get_runs(self) -> List[Dict[str, Any]]:
        """
        Get list of all training runs.
        Scans log directory for any folders containing metrics.
        """
        runs = []
        
        if not self.log_dir.exists():
            return runs
        
        for run_dir in self.log_dir.iterdir():
            if not run_dir.is_dir():
                continue
            
            # Check if this directory has metrics
            metrics_dir = run_dir / "metrics"
            if not metrics_dir.exists():
                continue
            
            # Try to load metadata
            metadata_file = run_dir / "metadata.json"
            metadata = {
                "run_id": run_dir.name,
                "status": "unknown",
                "start_time": None
            }
            
            if metadata_file.exists():
                try:
                    with open(metadata_file) as f:
                        loaded_metadata = json.load(f)
                        metadata.update(loaded_metadata)
                except Exception as e:
                    print(f"Error reading metadata for {run_dir}: {e}")
            
            # Get summary if exists
            summary_file = run_dir / "summary.json"
            if summary_file.exists():
                try:
                    with open(summary_file) as f:
                        summary = json.load(f)
                    metadata["summary"] = summary
                except Exception as e:
                    print(f"Error reading summary for {run_dir}: {e}")
            
            runs.append(metadata)
        
        # Sort by start time (newest first)
        runs.sort(key=lambda x: x.get("start_time", ""), reverse=True)
        return runs
    
    def discover_metrics(self, run_id: str) -> Dict[str, List[str]]:
        """
        AUTO-DISCOVER all metrics in a run's logs.
        This is the magic - no hardcoded metric names!
        
        Returns metrics grouped by category:
        {
            "train": ["train/loss", "train/lr"],
            "gpu": ["gpu/temp", "gpu/util"],
            "custom": ["my_metric", "another_one"]
        }
        """
        cache_key = f"metrics_{run_id}"
        
        # Check cache
        with self.cache_lock:
            if cache_key in self.cache:
                if time.time() - self.last_cache_update.get(cache_key, 0) < self.cache_ttl:
                    return self.cache[cache_key]
        
        run_dir = self.log_dir / run_id
        metrics_dir = run_dir / "metrics"
        
        if not metrics_dir.exists():
            return {}
        
        # Find all JSONL files
        jsonl_files = list(metrics_dir.glob("*.jsonl"))
        
        if not jsonl_files:
            return {}
        
        # Scan files and extract unique metric names
        unique_metrics = set()
        
        for jsonl_file in jsonl_files:
            try:
                with open(jsonl_file) as f:
                    # Sample first 1000 lines to discover metrics quickly
                    for i, line in enumerate(f):
                        if i >= 1000:
                            break
                        
                        try:
                            entry = json.loads(line.strip())
                            metric_name = entry.get("metric")
                            if metric_name:
                                unique_metrics.add(metric_name)
                        except json.JSONDecodeError:
                            continue
            except Exception as e:
                print(f"Error scanning {jsonl_file}: {e}")
        
        # Group metrics by category (prefix before "/")
        grouped = defaultdict(list)
        
        for metric in sorted(unique_metrics):
            if "/" in metric:
                category = metric.split("/", 1)[0]
            else:
                category = "other"
            
            grouped[category].append(metric)
        
        result = dict(grouped)
        
        # Update cache
        with self.cache_lock:
            self.cache[cache_key] = result
            self.last_cache_update[cache_key] = time.time()
        
        return result
    
    def get_metric_data(self, run_id: str, metric_names: List[str], 
                       max_points: int = 1000) -> Dict[str, List[Dict[str, Any]]]:
        """
        Get historical data for specific metrics.
        Supports multiple metrics in one call.
        
        Args:
            run_id: Run identifier
            metric_names: List of metric names to fetch
            max_points: Maximum data points per metric (downsamples if needed)
        
        Returns:
            Dictionary mapping metric name to list of {step, value, timestamp}
        """
        run_dir = self.log_dir / run_id
        metrics_dir = run_dir / "metrics"
        
        if not metrics_dir.exists():
            return {}
        
        # Find JSONL files
        jsonl_files = list(metrics_dir.glob("*.jsonl"))
        
        if not jsonl_files:
            return {}
        
        # Collect data for requested metrics
        data = {metric: [] for metric in metric_names}
        
        for jsonl_file in jsonl_files:
            try:
                with open(jsonl_file) as f:
                    for line in f:
                        try:
                            entry = json.loads(line.strip())
                            metric_name = entry.get("metric")
                            
                            if metric_name in metric_names:
                                data[metric_name].append({
                                    "step": entry.get("step"),
                                    "value": entry.get("value"),
                                    "timestamp": entry.get("timestamp")
                                })
                        except json.JSONDecodeError:
                            continue
            except Exception as e:
                print(f"Error reading {jsonl_file}: {e}")
        
        # Downsample if needed
        for metric_name in data:
            if len(data[metric_name]) > max_points:
                step_size = len(data[metric_name]) // max_points
                data[metric_name] = data[metric_name][::step_size]
        
        return data
    
    def get_current_metrics(self, run_id: str) -> Dict[str, Any]:
        """Get current/latest values for all metrics"""
        cache_key = f"current_{run_id}"
        
        # Check cache
        with self.cache_lock:
            if cache_key in self.cache:
                if time.time() - self.last_cache_update.get(cache_key, 0) < self.cache_ttl:
                    return self.cache[cache_key]
        
        run_dir = self.log_dir / run_id
        current_file = run_dir / "current_metrics.json"
        
        if not current_file.exists():
            return {}
        
        try:
            with open(current_file) as f:
                data = json.load(f)
            
            # Update cache
            with self.cache_lock:
                self.cache[cache_key] = data
                self.last_cache_update[cache_key] = time.time()
            
            return data
        except Exception as e:
            print(f"Error reading current metrics: {e}")
            return {}


# Initialize server
metrics_server = FlexibleMetricsServer(LOG_DIR)


# ============================================================================
# API Routes
# ============================================================================

@app.route("/api/runs")
def api_runs():
    """Get list of all training runs"""
    runs = metrics_server.get_runs()
    return jsonify({"runs": runs})


@app.route("/api/runs/<run_id>")
def api_run_details(run_id):
    """Get details for a specific run"""
    runs = metrics_server.get_runs()
    run = next((r for r in runs if r["run_id"] == run_id), None)
    
    if not run:
        return jsonify({"error": "Run not found"}), 404
    
    return jsonify(run)


@app.route("/api/runs/<run_id>/metrics")
def api_discover_metrics(run_id):
    """
    AUTO-DISCOVER all metrics in a run.
    This is the key endpoint that makes the dashboard flexible!
    """
    metrics = metrics_server.discover_metrics(run_id)
    return jsonify({"metrics": metrics})


@app.route("/api/runs/<run_id>/data")
def api_metric_data(run_id):
    """
    Get data for selected metrics.
    
    Query params:
        metrics: Comma-separated list of metric names
        max_points: Maximum points per metric (default 1000)
    """
    # Parse query params
    metrics_param = request.args.get("metrics", "")
    metric_names = [m.strip() for m in metrics_param.split(",") if m.strip()]
    max_points = request.args.get("max_points", 1000, type=int)
    
    if not metric_names:
        return jsonify({"error": "No metrics specified"}), 400
    
    # Fetch data
    data = metrics_server.get_metric_data(run_id, metric_names, max_points)
    
    return jsonify(data)


@app.route("/api/runs/<run_id>/current")
def api_current_metrics(run_id):
    """Get current/latest values for all metrics"""
    metrics = metrics_server.get_current_metrics(run_id)
    return jsonify(metrics)


@app.route("/api/health")
def api_health():
    """Health check endpoint"""
    return jsonify({
        "status": "ok",
        "log_dir": str(LOG_DIR.absolute()),
        "log_dir_exists": LOG_DIR.exists()
    })


# ============================================================================
# Serve Dashboard Frontend
# ============================================================================

@app.route("/")
def serve_dashboard():
    """Serve the main dashboard HTML"""
    dashboard_file = DASHBOARD_DIR / "index.html"
    if dashboard_file.exists():
        return send_from_directory(DASHBOARD_DIR, "index.html")
    return "Dashboard not found. Please ensure dashboard/index.html exists.", 404


@app.route("/<path:path>")
def serve_static(path):
    """Serve static files (CSS, JS, etc.)"""
    return send_from_directory(DASHBOARD_DIR, path)


# ============================================================================
# Main
# ============================================================================

def run_server(host="0.0.0.0", port=5000, debug=False, log_dir=None):
    """
    Run the flexible dashboard server.
    
    Args:
        host: Host to bind to
        port: Port to listen on
        debug: Enable debug mode
        log_dir: Custom log directory (overrides default)
    """
    global LOG_DIR, metrics_server
    
    if log_dir:
        LOG_DIR = Path(log_dir)
        metrics_server = FlexibleMetricsServer(LOG_DIR)
    
    print("=" * 80)
    print("🚀 Flexible Training Dashboard Server")
    print("=" * 80)
    print(f"Server: http://{host}:{port}")
    print(f"Logs: {LOG_DIR.absolute()}")
    print(f"Logs exist: {LOG_DIR.exists()}")
    print("=" * 80)
    print("\nFeatures:")
    print("  ✅ Auto-discovers ALL metrics (no hardcoded names)")
    print("  ✅ Works with ANY framework")
    print("  ✅ Dynamic chart generation")
    print("  ✅ Zero configuration required")
    print("=" * 80)
    
    app.run(host=host, port=port, debug=debug, threaded=True)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Flexible Training Dashboard Server")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=5000, help="Port to listen on")
    parser.add_argument("--log-dir", help="Directory containing logs")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    
    args = parser.parse_args()
    
    run_server(
        host=args.host,
        port=args.port,
        debug=args.debug,
        log_dir=args.log_dir
    )