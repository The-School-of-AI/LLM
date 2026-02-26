#!/usr/bin/env python3
"""
monitor_report.py — HTML Report with Charts from Monitor Logs
=============================================================
Reads dstat.csv and generates a self-contained HTML file with
interactive charts (CPU, Memory, Disk I/O, Network).

Usage:
    python3 scripts/monitor_report.py
    python3 scripts/monitor_report.py /path/to/logs
    python3 scripts/monitor_report.py /path/to/logs --upload s3://bucket/prefix

Requirements:
    No external dependencies — uses only Python stdlib.
    Charts are rendered via embedded Chart.js (CDN).
"""

import argparse
import csv
import html
import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path


def parse_dstat_csv(filepath: str) -> dict:
    """Parse dstat CSV into structured data for charting."""
    timestamps = []
    cpu_usr = []
    cpu_sys = []
    cpu_idl = []
    cpu_wai = []
    cpu_stl = []
    disk_read = []
    disk_write = []
    net_recv = []
    net_send = []

    with open(filepath, "r") as f:
        reader = csv.reader(f)
        header_found = False
        col_map = {}

        for row in reader:
            if not row:
                continue
            # Skip dstat metadata lines
            if row[0].startswith('"') or "Dstat" in str(row):
                continue
            # Find the header row with column names
            if "usr" in row or "usr" in str(row):
                for i, col in enumerate(row):
                    col = col.strip().strip('"').lower()
                    col_map[col] = i
                header_found = True
                continue
            if not header_found:
                continue

            try:
                idx = len(timestamps)
                timestamps.append(idx)

                usr_i = col_map.get("usr", 0)
                sys_i = col_map.get("sys", 1)
                idl_i = col_map.get("idl", 2)
                wai_i = col_map.get("wai", 3)
                stl_i = col_map.get("stl", 4)

                cpu_usr.append(float(row[usr_i]))
                cpu_sys.append(float(row[sys_i]))
                cpu_idl.append(float(row[idl_i]))
                cpu_wai.append(float(row[wai_i]))
                cpu_stl.append(float(row[stl_i]))

                # dstat disk columns: read, writ
                read_i = col_map.get("read", 5)
                writ_i = col_map.get("writ", 6)
                # Convert bytes to MB/s
                disk_read.append(float(row[read_i]) / 1048576)
                disk_write.append(float(row[writ_i]) / 1048576)

                # dstat net columns: recv, send
                recv_i = col_map.get("recv", 7)
                send_i = col_map.get("send", 8)
                net_recv.append(float(row[recv_i]) / 1048576)
                net_send.append(float(row[send_i]) / 1048576)
            except (ValueError, IndexError):
                continue

    return {
        "timestamps": timestamps,
        "cpu_usr": cpu_usr,
        "cpu_sys": cpu_sys,
        "cpu_idl": cpu_idl,
        "cpu_wai": cpu_wai,
        "cpu_stl": cpu_stl,
        "disk_read": disk_read,
        "disk_write": disk_write,
        "net_recv": net_recv,
        "net_send": net_send,
    }


def compute_summary(data: dict) -> dict:
    """Compute summary statistics."""

    def avg(lst):
        return sum(lst) / len(lst) if lst else 0

    def peak(lst):
        return max(lst) if lst else 0

    cpu_util = [u + s for u, s in zip(data["cpu_usr"], data["cpu_sys"])]

    return {
        "samples": len(data["timestamps"]),
        "cpu_avg_util": round(avg(cpu_util), 1),
        "cpu_peak_util": round(peak(cpu_util), 1),
        "cpu_avg_steal": round(avg(data["cpu_stl"]), 2),
        "cpu_peak_steal": round(peak(data["cpu_stl"]), 2),
        "cpu_avg_iowait": round(avg(data["cpu_wai"]), 2),
        "cpu_peak_iowait": round(peak(data["cpu_wai"]), 2),
        "disk_avg_read": round(avg(data["disk_read"]), 1),
        "disk_peak_read": round(peak(data["disk_read"]), 1),
        "disk_avg_write": round(avg(data["disk_write"]), 1),
        "disk_peak_write": round(peak(data["disk_write"]), 1),
        "net_avg_recv": round(avg(data["net_recv"]), 1),
        "net_peak_recv": round(peak(data["net_recv"]), 1),
    }


def check_threshold(value, op, threshold):
    """Return pass/fail/warn status."""
    if op == ">=":
        return "pass" if value >= threshold else "fail"
    elif op == "<":
        return "pass" if value < threshold else "fail"
    return "warn"


def generate_html(data: dict, summary: dict, log_dir: str) -> str:
    """Generate self-contained HTML report."""
    now = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")

    checks = [
        ("CPU Utilization (avg)", summary["cpu_avg_util"], "%", ">=", 80),
        ("CPU Steal (peak)", summary["cpu_peak_steal"], "%", "<", 1),
        ("CPU I/O Wait (avg)", summary["cpu_avg_iowait"], "%", "<", 5),
        ("Disk Read (peak)", summary["disk_peak_read"], "MB/s", ">=", 0),
        ("Net RX (peak)", summary["net_peak_recv"], "MB/s", ">=", 0),
    ]

    checks_html = ""
    for name, value, unit, op, thresh in checks:
        status = check_threshold(value, op, thresh)
        icon = {"pass": "✅", "fail": "❌", "warn": "⚠️"}[status]
        color = {"pass": "#22c55e", "fail": "#ef4444", "warn": "#eab308"}[status]
        checks_html += f"""
        <tr>
          <td>{icon}</td>
          <td>{html.escape(name)}</td>
          <td style="color:{color};font-weight:bold">
            {value}{unit}
          </td>
          <td>{op} {thresh}{unit}</td>
        </tr>"""

    chart_data = json.dumps(data, separators=(",", ":"))

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Infrastructure Monitoring Report</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4"></script>
<style>
  * {{ margin:0; padding:0; box-sizing:border-box; }}
  body {{
    font-family: -apple-system,BlinkMacSystemFont,
      'Segoe UI',Roboto,sans-serif;
    background: #0f172a; color: #e2e8f0;
    padding: 2rem;
  }}
  h1 {{ color: #38bdf8; margin-bottom: 0.5rem; }}
  h2 {{
    color: #94a3b8; font-size: 1.1rem;
    margin-bottom: 2rem;
  }}
  .grid {{
    display: grid;
    grid-template-columns: repeat(auto-fit,minmax(500px,1fr));
    gap: 1.5rem; margin-bottom: 2rem;
  }}
  .card {{
    background: #1e293b; border-radius: 12px;
    padding: 1.5rem;
    border: 1px solid #334155;
  }}
  .card h3 {{
    color: #38bdf8; margin-bottom: 1rem;
    font-size: 1rem;
  }}
  canvas {{ max-height: 300px; }}
  table {{
    width: 100%; border-collapse: collapse;
    margin-top: 1rem;
  }}
  th, td {{
    padding: 0.6rem 1rem; text-align: left;
    border-bottom: 1px solid #334155;
  }}
  th {{ color: #94a3b8; font-weight: 600; }}
  .summary-grid {{
    display: grid;
    grid-template-columns: repeat(auto-fit,minmax(200px,1fr));
    gap: 1rem; margin-bottom: 2rem;
  }}
  .stat {{
    background: #1e293b; border-radius: 8px;
    padding: 1rem; text-align: center;
    border: 1px solid #334155;
  }}
  .stat .value {{
    font-size: 2rem; font-weight: 700;
    color: #38bdf8;
  }}
  .stat .label {{
    color: #94a3b8; font-size: 0.85rem;
    margin-top: 0.3rem;
  }}
</style>
</head>
<body>

<h1>🖥️ Infrastructure Monitoring Report</h1>
<h2>Generated: {now} | Samples: {summary['samples']} |
   Log dir: {html.escape(log_dir)}</h2>

<div class="summary-grid">
  <div class="stat">
    <div class="value">{summary['cpu_avg_util']}%</div>
    <div class="label">Avg CPU Utilization</div>
  </div>
  <div class="stat">
    <div class="value">{summary['cpu_peak_steal']}%</div>
    <div class="label">Peak CPU Steal</div>
  </div>
  <div class="stat">
    <div class="value">{summary['cpu_avg_iowait']}%</div>
    <div class="label">Avg I/O Wait</div>
  </div>
  <div class="stat">
    <div class="value">{summary['net_peak_recv']} MB/s</div>
    <div class="label">Peak Network RX</div>
  </div>
  <div class="stat">
    <div class="value">{summary['disk_peak_read']} MB/s</div>
    <div class="label">Peak Disk Read</div>
  </div>
  <div class="stat">
    <div class="value">{summary['disk_peak_write']} MB/s</div>
    <div class="label">Peak Disk Write</div>
  </div>
</div>

<div class="card" style="margin-bottom:2rem">
  <h3>Threshold Checks</h3>
  <table>
    <tr><th></th><th>Metric</th>
        <th>Value</th><th>Threshold</th></tr>
    {checks_html}
  </table>
</div>

<div class="grid">
  <div class="card">
    <h3>CPU Utilization Over Time</h3>
    <canvas id="cpuChart"></canvas>
  </div>
  <div class="card">
    <h3>CPU Steal & I/O Wait</h3>
    <canvas id="stealChart"></canvas>
  </div>
  <div class="card">
    <h3>Disk I/O (MB/s)</h3>
    <canvas id="diskChart"></canvas>
  </div>
  <div class="card">
    <h3>Network Throughput (MB/s)</h3>
    <canvas id="netChart"></canvas>
  </div>
</div>

<script>
const D = {chart_data};
const labels = D.timestamps.map(i => i);
const chartOpts = {{
  responsive: true,
  animation: false,
  scales: {{
    x: {{ display: false }},
    y: {{ beginAtZero: true,
          ticks: {{ color: '#94a3b8' }},
          grid: {{ color: '#334155' }} }}
  }},
  plugins: {{ legend: {{ labels: {{ color: '#e2e8f0' }} }} }}
}};

new Chart(document.getElementById('cpuChart'), {{
  type: 'line',
  data: {{
    labels,
    datasets: [
      {{ label:'%usr', data:D.cpu_usr,
         borderColor:'#38bdf8', borderWidth:1,
         pointRadius:0, fill:false }},
      {{ label:'%sys', data:D.cpu_sys,
         borderColor:'#a78bfa', borderWidth:1,
         pointRadius:0, fill:false }},
      {{ label:'%idle', data:D.cpu_idl,
         borderColor:'#475569', borderWidth:1,
         pointRadius:0, fill:false }}
    ]
  }},
  options: {{ ...chartOpts,
    scales: {{ ...chartOpts.scales,
      y: {{ ...chartOpts.scales.y, max: 100 }} }} }}
}});

new Chart(document.getElementById('stealChart'), {{
  type: 'line',
  data: {{
    labels,
    datasets: [
      {{ label:'%steal', data:D.cpu_stl,
         borderColor:'#ef4444', borderWidth:1.5,
         pointRadius:0, fill:false }},
      {{ label:'%iowait', data:D.cpu_wai,
         borderColor:'#eab308', borderWidth:1.5,
         pointRadius:0, fill:false }}
    ]
  }},
  options: chartOpts
}});

new Chart(document.getElementById('diskChart'), {{
  type: 'line',
  data: {{
    labels,
    datasets: [
      {{ label:'Read MB/s', data:D.disk_read,
         borderColor:'#22c55e', borderWidth:1,
         pointRadius:0, fill:false }},
      {{ label:'Write MB/s', data:D.disk_write,
         borderColor:'#f97316', borderWidth:1,
         pointRadius:0, fill:false }}
    ]
  }},
  options: chartOpts
}});

new Chart(document.getElementById('netChart'), {{
  type: 'line',
  data: {{
    labels,
    datasets: [
      {{ label:'RX MB/s', data:D.net_recv,
         borderColor:'#06b6d4', borderWidth:1,
         pointRadius:0, fill:false }},
      {{ label:'TX MB/s', data:D.net_send,
         borderColor:'#f472b6', borderWidth:1,
         pointRadius:0, fill:false }}
    ]
  }},
  options: chartOpts
}});
</script>

</body>
</html>"""


def upload_to_s3(local_dir: str, s3_uri: str):
    """Upload all logs and reports to S3."""
    print(f"\n📤 Uploading logs to {s3_uri}...")
    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    tar_file = f"/tmp/monitor_logs_{timestamp}.tar.gz"

    subprocess.run(["tar", "czf", tar_file, "-C", local_dir, "."], check=True)

    s3_dest = f"{s3_uri}/monitor_logs_{timestamp}.tar.gz"
    subprocess.run(["aws", "s3", "cp", tar_file, s3_dest], check=True)
    print(f"✅ Uploaded to {s3_dest}")

    # Also upload HTML report separately
    html_files = list(Path(local_dir).glob("report_*.html"))
    for hf in html_files:
        s3_html = f"{s3_uri}/{hf.name}"
        subprocess.run(
            ["aws", "s3", "cp", str(hf), s3_html, "--content-type", "text/html"],
            check=True,
        )
        print(f"✅ Uploaded {hf.name} to {s3_html}")

    os.remove(tar_file)


def main():
    parser = argparse.ArgumentParser(description="Generate HTML monitoring report")
    parser.add_argument(
        "log_dir",
        nargs="?",
        default="/mnt/nvme/logs",
        help="Path to monitor log directory",
    )
    parser.add_argument(
        "--upload",
        metavar="S3_URI",
        help="Upload logs and report to S3 " "(e.g., s3://bucket/infra-reports)",
    )
    args = parser.parse_args()

    dstat_path = os.path.join(args.log_dir, "dstat.csv")
    if not os.path.exists(dstat_path):
        print(f"❌ dstat.csv not found in {args.log_dir}")
        print("   Run monitor.sh first to generate logs.")
        sys.exit(1)

    print("📊 Parsing dstat.csv...")
    data = parse_dstat_csv(dstat_path)

    if not data["timestamps"]:
        print("❌ No data rows found in dstat.csv")
        sys.exit(1)

    print(f"   Found {len(data['timestamps'])} samples")

    summary = compute_summary(data)
    html_content = generate_html(data, summary, args.log_dir)

    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    output_path = os.path.join(args.log_dir, f"report_{timestamp}.html")
    with open(output_path, "w") as f:
        f.write(html_content)

    print(f"✅ HTML report saved to: {output_path}")
    print(f"   Download: scp user@ec2:{output_path} ./")

    if args.upload:
        upload_to_s3(args.log_dir, args.upload)

    # Print summary to terminal
    print("\n📋 Quick Summary:")
    print(f"   CPU avg util  : {summary['cpu_avg_util']}%")
    print(f"   CPU peak steal: {summary['cpu_peak_steal']}%")
    print(f"   CPU avg iowait: {summary['cpu_avg_iowait']}%")
    print(f"   Disk peak read: {summary['disk_peak_read']} MB/s")
    print(f"   Net peak RX   : {summary['net_peak_recv']} MB/s")


if __name__ == "__main__":
    main()
