import argparse
import json
import os
import time
from datetime import datetime, timezone


def parse_simple_yaml(path):
    data = {}
    stack = [(0, data)]
    with open(path, "r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.rstrip("\n")
            if not line.strip() or line.strip().startswith("#"):
                continue
            indent = len(line) - len(line.lstrip(" "))
            key, sep, rest = line.strip().partition(":")
            if not sep:
                continue
            value = rest.strip()
            while stack and indent < stack[-1][0]:
                stack.pop()
            current = stack[-1][1]
            if value == "":
                new_obj = {}
                parsed_key = parse_key(key)
                current[parsed_key] = new_obj
                stack.append((indent + 2, new_obj))
            else:
                parsed_key = parse_key(key)
                current[parsed_key] = parse_value(value)
    return data


def parse_key(value):
    if value.isdigit():
        return int(value)
    return value


def parse_value(value):
    lowered = value.lower()
    if lowered in ("true", "false"):
        return lowered == "true"
    try:
        if "." in value:
            return float(value)
        return int(value)
    except ValueError:
        return value


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def load_metrics_line(line):
    try:
        return json.loads(line)
    except json.JSONDecodeError:
        return None


def write_incident(incident_dir, payload):
    os.makedirs(incident_dir, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    path = os.path.join(incident_dir, f"incident_{stamp}.json")
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
    return path


def write_halt(halt_file):
    with open(halt_file, "w", encoding="utf-8") as handle:
        handle.write("HALT\n")


def should_halt_for_cost(metrics_state, triggers, expected_cost):
    if expected_cost is None:
        return False, None
    if metrics_state["window_cost_count"] < 1:
        return False, None
    actual_cost = metrics_state["window_cost_sum"] / metrics_state["window_cost_count"]
    drift = ((actual_cost - expected_cost) / expected_cost) * 100
    if drift > triggers["cost_drift_percent"]:
        return True, {
            "actual_cost_per_1m": actual_cost,
            "expected_cost_per_1m": expected_cost,
            "drift_percent": drift,
        }
    return False, None


def main():
    parser = argparse.ArgumentParser(
        description="Training cost governor and HALT controller"
    )
    parser.add_argument("--metrics", required=True, help="Path to metrics JSONL")
    parser.add_argument("--budget", required=True, help="Path to budget YAML")
    parser.add_argument(
        "--poll_s", type=float, default=2.0, help="Polling interval seconds"
    )
    parser.add_argument(
        "--stall_timeout_s",
        type=float,
        default=300.0,
        help="No-metrics timeout seconds",
    )
    args = parser.parse_args()

    budget = parse_simple_yaml(args.budget)
    triggers = budget.get("triggers", {})
    stages = budget.get("stages", {})
    halt_file = budget.get("halt_file", "HALT")
    incident_dir = budget.get("incident_dir", "incidents")

    metrics_state = {
        "baseline_tokens_per_s": None,
        "baseline_samples": [],
        "throughput_below": 0,
        "data_starve_count": 0,
        "total_tokens": 0,
        "window_tokens": 0,
        "window_cost_sum": 0.0,
        "window_cost_count": 0,
        "last_metric_ts": time.time(),
        "stage": None,
    }

    last_pos = 0

    while True:
        if not os.path.exists(args.metrics):
            time.sleep(args.poll_s)
            continue

        with open(args.metrics, "r", encoding="utf-8") as handle:
            handle.seek(last_pos)
            new_lines = handle.readlines()
            last_pos = handle.tell()

        if not new_lines:
            if time.time() - metrics_state["last_metric_ts"] > args.stall_timeout_s:
                incident_path = write_incident(
                    incident_dir,
                    {
                        "ts": utc_now(),
                        "trigger": "metrics_stall",
                        "details": {"stall_timeout_s": args.stall_timeout_s},
                    },
                )
                write_halt(halt_file)
                print(f"HALT: metrics stall detected. Incident: {incident_path}")
                return
            time.sleep(args.poll_s)
            continue

        for line in new_lines:
            payload = load_metrics_line(line)
            if payload is None:
                continue
            metrics_state["last_metric_ts"] = time.time()
            stage = int(payload.get("stage", 0))
            if stage:
                metrics_state["stage"] = stage
            stage_cfg = stages.get(metrics_state["stage"], {})
            token_budget = stage_cfg.get("token_budget")
            expected_cost = stage_cfg.get("expected_cost_per_1m_tokens")

            tokens = int(payload.get("tokens", 0))
            tokens_per_s = float(payload.get("tokens_per_s", 0.0))
            data_wait_s = float(payload.get("data_wait_s", 0.0))
            step_time_s = float(payload.get("step_time_s", 0.0))
            loss = payload.get("loss")
            nan_flag = bool(payload.get("nan", False))
            if loss is not None and isinstance(loss, (float, int)):
                nan_flag = nan_flag or (loss != loss)

            if nan_flag and triggers.get("nan_detected", False):
                incident_path = write_incident(
                    incident_dir,
                    {"ts": utc_now(), "trigger": "nan_detected", "details": payload},
                )
                write_halt(halt_file)
                print(f"HALT: NaN detected. Incident: {incident_path}")
                return

            metrics_state["total_tokens"] += tokens
            metrics_state["window_tokens"] += tokens

            if token_budget and metrics_state["total_tokens"] > token_budget:
                incident_path = write_incident(
                    incident_dir,
                    {
                        "ts": utc_now(),
                        "trigger": "token_budget_exceeded",
                        "details": {
                            "total_tokens": metrics_state["total_tokens"],
                            "token_budget": token_budget,
                        },
                    },
                )
                write_halt(halt_file)
                print(f"HALT: token budget exceeded. Incident: {incident_path}")
                return

            if step_time_s > 0:
                ratio = data_wait_s / step_time_s
                if ratio > triggers.get("data_starvation_ratio", 1.0):
                    metrics_state["data_starve_count"] += 1
                else:
                    metrics_state["data_starve_count"] = 0

                if metrics_state["data_starve_count"] >= triggers.get(
                    "data_starvation_window_steps", 999999
                ):
                    incident_path = write_incident(
                        incident_dir,
                        {
                            "ts": utc_now(),
                            "trigger": "data_starvation",
                            "details": {"ratio": ratio, "payload": payload},
                        },
                    )
                    write_halt(halt_file)
                    print(f"HALT: data starvation detected. Incident: {incident_path}")
                    return

            if tokens_per_s > 0:
                if metrics_state["baseline_tokens_per_s"] is None:
                    metrics_state["baseline_samples"].append(tokens_per_s)
                    if len(metrics_state["baseline_samples"]) >= 200:
                        sorted_samples = sorted(metrics_state["baseline_samples"])
                        mid = len(sorted_samples) // 2
                        metrics_state["baseline_tokens_per_s"] = sorted_samples[mid]
                else:
                    baseline = metrics_state["baseline_tokens_per_s"]
                    if tokens_per_s < baseline * triggers.get(
                        "throughput_collapse_ratio", 0.0
                    ):
                        metrics_state["throughput_below"] += 1
                    else:
                        metrics_state["throughput_below"] = 0

                    if metrics_state["throughput_below"] >= triggers.get(
                        "throughput_collapse_window_steps", 999999
                    ):
                        incident_path = write_incident(
                            incident_dir,
                            {
                                "ts": utc_now(),
                                "trigger": "throughput_collapse",
                                "details": {"baseline": baseline, "payload": payload},
                            },
                        )
                        write_halt(halt_file)
                        print(
                            f"HALT: throughput collapse detected. Incident: {incident_path}"
                        )
                        return

            if metrics_state["window_tokens"] >= 1_000_000:
                if payload.get("cost_per_1m_tokens") is not None:
                    metrics_state["window_cost_sum"] += float(
                        payload["cost_per_1m_tokens"]
                    )
                    metrics_state["window_cost_count"] += 1

                halt_cost, details = should_halt_for_cost(
                    metrics_state, triggers, expected_cost
                )
                if halt_cost:
                    incident_path = write_incident(
                        incident_dir,
                        {"ts": utc_now(), "trigger": "cost_drift", "details": details},
                    )
                    write_halt(halt_file)
                    print(f"HALT: cost drift detected. Incident: {incident_path}")
                    return

                metrics_state["window_tokens"] = 0
                metrics_state["window_cost_sum"] = 0.0
                metrics_state["window_cost_count"] = 0

        time.sleep(args.poll_s)


if __name__ == "__main__":
    main()
