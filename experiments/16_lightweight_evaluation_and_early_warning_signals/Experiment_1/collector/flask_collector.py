"""
Optional Flask-based result collector.

Team members POST their result JSON to this server, which stores it and
serves a live summary page.

Usage (on the central machine):
    python collector/flask_collector.py --port 5001

Team members submit results:
    curl -X POST http://<host>:5001/submit \\
         -H "Content-Type: application/json" \\
         -d @results/raw/my_result.json

View live summary:
    http://<host>:5001/
"""
from __future__ import annotations

import argparse
import json
import logging
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = ROOT / "results" / "raw"
AGG_FILE = ROOT / "results" / "aggregated_results.json"

logger = logging.getLogger(__name__)


def create_app() -> "Flask":
    try:
        from flask import Flask, request, jsonify, render_template_string
        from flask_cors import CORS
    except ImportError:
        raise ImportError("flask and flask-cors are required: pip install flask flask-cors")

    app = Flask(__name__)
    CORS(app)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    SUMMARY_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
  <title>Team 16 Early Warning — Eval Dashboard</title>
  <meta http-equiv="refresh" content="30">
  <style>
    body { font-family: monospace; background: #1e1e2e; color: #cdd6f4; padding: 2em; }
    h1 { color: #89b4fa; }
    table { border-collapse: collapse; width: 100%; margin-top: 1em; }
    th { background: #313244; color: #89dceb; padding: 8px 12px; text-align: left; }
    td { padding: 6px 12px; border-bottom: 1px solid #313244; }
    tr:hover td { background: #313244; }
    .good { color: #a6e3a1; }
    .bad  { color: #f38ba8; }
    .warn { color: #fab387; }
    .meta { color: #6c7086; font-size: 0.85em; margin-top: 2em; }
  </style>
</head>
<body>
  <h1>🔍 Team 16 Early Warning — Checkpoint Eval Dashboard</h1>
  <p class="meta">Auto-refreshes every 30s | {{ total_runs }} runs | Last updated: {{ generated_at }}</p>
  <table>
    <tr>
      <th>Checkpoint</th><th>Quant</th>
      <th>MMLU ↑</th><th>PPL ↓</th><th>Code ↑</th><th>Math ↑</th><th>Consist ↑</th>
      <th>Host</th><th>Time (UTC)</th>
    </tr>
    {% for run in runs %}
    <tr>
      <td>{{ run.checkpoint_name }}</td>
      <td>{{ run.quant_mode }}</td>
      <td>{{ run.metrics.get('mmlu', {}).get('value', 'N/A') }}</td>
      <td>{{ run.metrics.get('language_modeling', {}).get('value', 'N/A') }}</td>
      <td>{{ run.metrics.get('code_continuation', {}).get('value', 'N/A') }}</td>
      <td>{{ run.metrics.get('math_prose', {}).get('value', 'N/A') }}</td>
      <td>{{ run.metrics.get('consistency', {}).get('value', 'N/A') }}</td>
      <td>{{ run.hostname }}</td>
      <td>{{ run.timestamp_utc[:16] }}</td>
    </tr>
    {% endfor %}
  </table>
</body>
</html>
"""

    def _aggregate() -> dict:
        from collector.collect_results import load_all_results, aggregate
        records = load_all_results(RESULTS_DIR)
        return aggregate(records)

    @app.route("/", methods=["GET"])
    def index():
        agg = _aggregate()
        return render_template_string(SUMMARY_TEMPLATE, **agg)

    @app.route("/submit", methods=["POST"])
    def submit():
        data = request.get_json(force=True, silent=True)
        if not data:
            return jsonify({"error": "Invalid JSON"}), 400

        run_id = data.get("run_id", f"unknown_{datetime.utcnow().strftime('%Y%m%dT%H%M%SZ')}")
        safe_id = "".join(c if c.isalnum() or c in "-_." else "_" for c in run_id)
        out_file = RESULTS_DIR / f"{safe_id}.json"

        with open(out_file, "w") as f:
            json.dump(data, f, indent=2)

        logger.info(f"Received result: {safe_id} → {out_file}")
        return jsonify({"status": "ok", "saved_as": str(out_file)}), 200

    @app.route("/api/results", methods=["GET"])
    def api_results():
        agg = _aggregate()
        return jsonify(agg)

    @app.route("/health", methods=["GET"])
    def health():
        return jsonify({"status": "ok", "timestamp": datetime.utcnow().isoformat()})

    return app


def main() -> None:
    p = argparse.ArgumentParser(description="Flask result collector server")
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--port", type=int, default=5001)
    p.add_argument("--debug", action="store_true")
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO)
    app = create_app()
    print(f"Starting collector server on http://{args.host}:{args.port}")
    app.run(host=args.host, port=args.port, debug=args.debug)


if __name__ == "__main__":
    main()
