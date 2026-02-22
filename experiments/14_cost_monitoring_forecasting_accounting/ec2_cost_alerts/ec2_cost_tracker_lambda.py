"""
EC2 Cost Tracker Lambda
========================
Polls all running EC2 instances, calculates runtime hours and estimated cost,
tracks cumulative spend in S3, and sends Telegram alerts when thresholds are breached.


Uses the Telegram bot token and chat ID.

Environment Variables:
    TELEGRAM_BOT_TOKEN  - Telegram bot token 
    TELEGRAM_CHAT_ID    - Telegram chat/group ID
    STATE_BUCKET        - S3 bucket for cost state tracking
    STATE_PREFIX        - S3 key prefix (default: ec2-cost-state)
    CREDIT_LIMIT        - Per-account credit limit in USD (default: 500)
    ALERT_THRESHOLDS    - Comma-separated % thresholds (default: 60,80,90,95)
    INSTANCE_HOUR_ALERT - Alert if single instance runs > N hours (default: 4)
    SUMMARY_INTERVAL    - Hours between periodic summaries (default: 6)

Triggered by: EventBridge scheduled rule (every 15 minutes)
"""

import json
import os
import urllib.request
import urllib.parse
from datetime import datetime, timezone, timedelta
from decimal import Decimal, ROUND_HALF_UP

import boto3

# ──────────────────────────────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────────────────────────────

TELEGRAM_BOT_TOKEN = os.environ["TELEGRAM_BOT_TOKEN"]
TELEGRAM_CHAT_ID = os.environ["TELEGRAM_CHAT_ID"]
STATE_BUCKET = os.environ["STATE_BUCKET"]
STATE_PREFIX = os.environ.get("STATE_PREFIX", "ec2-cost-state")
CREDIT_LIMIT = float(os.environ.get("CREDIT_LIMIT", "500"))
ALERT_THRESHOLDS = [
    int(x) for x in os.environ.get("ALERT_THRESHOLDS", "60,80,90,95").split(",")
]
INSTANCE_HOUR_ALERT = float(os.environ.get("INSTANCE_HOUR_ALERT", "4"))
SUMMARY_INTERVAL_HOURS = float(os.environ.get("SUMMARY_INTERVAL", "6"))

# On-demand hourly pricing for us-east-1 (USD)
# Update this dict with instance types your teams actually use.
# Source: https://aws.amazon.com/ec2/pricing/on-demand/
ON_DEMAND_PRICING = {
    # General Purpose
    "t2.micro": 0.0116,
    "t2.small": 0.023,
    "t2.medium": 0.0464,
    "t2.large": 0.0928,
    "t2.xlarge": 0.1856,
    "t3.micro": 0.0104,
    "t3.small": 0.0208,
    "t3.medium": 0.0416,
    "t3.large": 0.0832,
    "t3.xlarge": 0.1664,
    "t3.2xlarge": 0.3328,
    "m5.large": 0.096,
    "m5.xlarge": 0.192,
    "m5.2xlarge": 0.384,
    "m5.4xlarge": 0.768,
    "m6i.large": 0.096,
    "m6i.xlarge": 0.192,
    "m6i.2xlarge": 0.384,
    "c5.large": 0.085,
    "c5.xlarge": 0.17,
    "c5.2xlarge": 0.34,
    "c5.4xlarge": 0.68,
    "r5.large": 0.126,
    "r5.xlarge": 0.252,
    "r5.2xlarge": 0.504,
    "g4dn.xlarge": 0.526,
    "g4dn.2xlarge": 0.752,
    "g4dn.4xlarge": 1.204,
    "g4dn.8xlarge": 2.176,
    "g4dn.12xlarge": 3.912,
    "g4dn.16xlarge": 4.352,
    "g5.xlarge": 1.006,
    "g5.2xlarge": 1.212,
    "g5.4xlarge": 1.624,
    "g5.8xlarge": 2.448,
    "g5.12xlarge": 5.672,
    "g5.16xlarge": 4.096,
    "g5.24xlarge": 8.144,
    "g5.48xlarge": 16.288,
    "p3.2xlarge": 3.06,
    "p3.8xlarge": 12.24,
    "p3.16xlarge": 24.48,
    "p4d.24xlarge": 32.77,
    "p5.48xlarge": 98.32,
    "i3.large": 0.156,
    "i3.xlarge": 0.312,
}

# Fallback: if instance type not in dict, estimate based on prefix
FALLBACK_ESTIMATES = {
    "t2": 0.05,
    "t3": 0.05,
    "m5": 0.20,
    "m6": 0.20,
    "c5": 0.20,
    "r5": 0.25,
    "g4": 1.50,
    "g5": 2.00,
    "p3": 10.00,
    "p4": 33.00,
    "p5": 98.00,
}


# ──────────────────────────────────────────────────────────────────────
# Helper functions
# ──────────────────────────────────────────────────────────────────────

def get_hourly_rate(instance_type: str) -> tuple:
    """
    Returns (hourly_rate, is_estimate).
    is_estimate=True means we used a fallback, not an exact price.
    """
    if instance_type in ON_DEMAND_PRICING:
        return ON_DEMAND_PRICING[instance_type], False

    prefix = instance_type.split(".")[0]
    if prefix in FALLBACK_ESTIMATES:
        return FALLBACK_ESTIMATES[prefix], True

    # Complete unknown - use a conservative estimate
    return 0.50, True


def get_instance_name(instance: dict) -> str:
    """Extract Name tag from EC2 instance."""
    for tag in instance.get("Tags", []):
        if tag["Key"] == "Name":
            return tag["Value"]
    return "(unnamed)"


def format_duration(seconds: float) -> str:
    """Format seconds into human-readable 'Xh Ym' string."""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    if hours > 0:
        return f"{hours}h {minutes}m"
    return f"{minutes}m"


def format_cost(amount: float) -> str:
    """Format as USD with 2 decimal places."""
    return f"${Decimal(str(amount)).quantize(Decimal('0.01'), rounding=ROUND_HALF_UP)}"


def ist_now() -> str:
    """Current time in IST formatted string."""
    ist = timezone(timedelta(hours=5, minutes=30))
    return datetime.now(ist).strftime("%d-%b-%Y %I:%M:%S %p IST")


def send_telegram(message: str):
    """Send message to Telegram chat."""
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    data = urllib.parse.urlencode({
        "chat_id": TELEGRAM_CHAT_ID,
        "text": message,
        "parse_mode": "HTML",
        "disable_web_page_preview": "true",
    }).encode("utf-8")

    req = urllib.request.Request(url, data=data, method="POST")
    req.add_header("Content-Type", "application/x-www-form-urlencoded")

    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            return json.loads(resp.read())
    except Exception as e:
        print(f"[ERROR] Telegram send failed: {e}")
        return None


# ──────────────────────────────────────────────────────────────────────
# S3 State Management
# ──────────────────────────────────────────────────────────────────────

s3 = boto3.client("s3")


def get_state_key() -> str:
    """S3 key for current month's state file."""
    now = datetime.now(timezone.utc)
    return f"{STATE_PREFIX}/{now.strftime('%Y-%m')}/state.json"


def load_state() -> dict:
    """Load cost tracking state from S3."""
    key = get_state_key()
    try:
        obj = s3.get_object(Bucket=STATE_BUCKET, Key=key)
        return json.loads(obj["Body"].read().decode("utf-8"))
    except s3.exceptions.NoSuchKey:
        return new_state()
    except Exception as e:
        print(f"[WARN] Could not load state from s3://{STATE_BUCKET}/{key}: {e}")
        return new_state()


def new_state() -> dict:
    """Create fresh state for current month."""
    now = datetime.now(timezone.utc)
    account_id = boto3.client("sts").get_caller_identity()["Account"]
    return {
        "account_id": account_id,
        "month": now.strftime("%Y-%m"),
        "cumulative_cost": 0.0,
        "instances": {},
        "alerted_thresholds": [],
        "last_summary_sent": None,
        "last_updated": now.isoformat(),
    }


def save_state(state: dict):
    """Save cost tracking state to S3."""
    key = get_state_key()
    state["last_updated"] = datetime.now(timezone.utc).isoformat()
    s3.put_object(
        Bucket=STATE_BUCKET,
        Key=key,
        Body=json.dumps(state, indent=2, default=str).encode("utf-8"),
        ContentType="application/json",
    )


# ──────────────────────────────────────────────────────────────────────
# AWS Billing (Cost Explorer) - Best effort
# ──────────────────────────────────────────────────────────────────────

def get_aws_billing() -> dict:
    """
    Fetch actual AWS account billing from Cost Explorer API.
    Returns {
        "total": float,           # Cost before credits (UnblendedCost)
        "net_cost": float,        # Cost after credits (NetUnblendedCost)
        "credits_used": float,    # Credits consumed this month
        "by_service": dict,       # Breakdown by service
        "available": bool
    }

    This is BEST EFFORT:
    - If IAM billing access isn't enabled, returns available=False
    - Never blocks the main EC2 tracking functionality
    """
    try:
        ce = boto3.client("ce")
        now = datetime.now(timezone.utc)
        start = now.strftime("%Y-%m-01")
        # End must be tomorrow (CE uses exclusive end date)
        end = (now + timedelta(days=1)).strftime("%Y-%m-%d")

        # Get total cost + credits in one call
        result = ce.get_cost_and_usage(
            TimePeriod={"Start": start, "End": end},
            Granularity="MONTHLY",
            Metrics=["UnblendedCost", "NetUnblendedCost"],
        )
        totals = result["ResultsByTime"][0]["Total"]
        total = float(totals["UnblendedCost"]["Amount"])
        net_cost = float(totals["NetUnblendedCost"]["Amount"])
        # Credits used = difference between gross and net cost
        credits_used = total - net_cost if total > net_cost else total

        # Get breakdown by service
        result_by_svc = ce.get_cost_and_usage(
            TimePeriod={"Start": start, "End": end},
            Granularity="MONTHLY",
            Metrics=["UnblendedCost"],
            GroupBy=[{"Type": "DIMENSION", "Key": "SERVICE"}],
        )
        by_service = {}
        for group in result_by_svc["ResultsByTime"][0].get("Groups", []):
            svc = group["Keys"][0]
            amt = float(group["Metrics"]["UnblendedCost"]["Amount"])
            if amt > 0.001:
                by_service[svc] = amt

        return {
            "total": total,
            "net_cost": net_cost,
            "credits_used": credits_used,
            "by_service": by_service,
            "available": True,
        }

    except Exception as e:
        print(f"[WARN] Could not fetch AWS billing: {e}")
        return {"total": 0.0, "net_cost": 0.0, "credits_used": 0.0,
                "by_service": {}, "available": False}


# ──────────────────────────────────────────────────────────────────────
# Core Logic
# ──────────────────────────────────────────────────────────────────────

def get_running_instances() -> list:
    """Get all running EC2 instances in this account/region."""
    ec2 = boto3.client("ec2")
    instances = []

    paginator = ec2.get_paginator("describe_instances")
    for page in paginator.paginate(
        Filters=[{"Name": "instance-state-name", "Values": ["running"]}]
    ):
        for reservation in page["Reservations"]:
            for instance in reservation["Instances"]:
                instances.append(instance)

    return instances


def calculate_instance_cost(instance: dict, state: dict, now: datetime) -> dict:
    """
    Calculate runtime and cost for a single instance.
    Updates state with tracking info.
    Returns dict with instance cost details.
    """
    instance_id = instance["InstanceId"]
    instance_type = instance["InstanceType"]
    instance_name = get_instance_name(instance)
    launch_time = instance["LaunchTime"]
    hourly_rate, is_estimate = get_hourly_rate(instance_type)

    # Total runtime since launch
    runtime_seconds = (now - launch_time).total_seconds()
    runtime_hours = runtime_seconds / 3600
    session_cost = runtime_hours * hourly_rate

    # Check if we've been tracking this instance
    prev = state["instances"].get(instance_id, {})
    prev_cumulative = prev.get("total_cost", 0.0)

    # Cost increment since last check
    last_updated = datetime.fromisoformat(state["last_updated"])
    delta_seconds = (now - last_updated).total_seconds()
    delta_cost = (delta_seconds / 3600) * hourly_rate

    # Update state
    state["instances"][instance_id] = {
        "name": instance_name,
        "type": instance_type,
        "hourly_rate": hourly_rate,
        "is_estimate": is_estimate,
        "launch_time": launch_time.isoformat(),
        "total_runtime_seconds": runtime_seconds,
        "total_cost": session_cost,
        "last_seen_running": now.isoformat(),
        "status": "running",
    }

    return {
        "instance_id": instance_id,
        "name": instance_name,
        "type": instance_type,
        "hourly_rate": hourly_rate,
        "is_estimate": is_estimate,
        "runtime_seconds": runtime_seconds,
        "session_cost": session_cost,
        "delta_cost": delta_cost,
        "is_new": instance_id not in state["instances"] or prev.get("status") != "running",
    }


def check_stopped_instances(state: dict, running_ids: set, now: datetime) -> list:
    """
    Detect instances that were running before but are now stopped.
    Returns list of stopped instance summaries.
    """
    stopped = []
    for iid, info in state["instances"].items():
        if iid not in running_ids and info.get("status") == "running":
            # This instance was running but is no longer
            info["status"] = "stopped"
            info["stopped_at"] = now.isoformat()
            stopped.append({
                "instance_id": iid,
                "name": info.get("name", "(unknown)"),
                "type": info.get("type", "unknown"),
                "total_cost": info.get("total_cost", 0.0),
                "total_runtime_seconds": info.get("total_runtime_seconds", 0),
            })
    return stopped


def build_summary_message(account_id: str, instance_costs: list, cumulative: float,
                          burn_rate: float, billing: dict = None) -> str:
    """Build periodic summary Telegram message."""
    # Use AWS billing for remaining credit if available, else use our EC2 estimate
    if billing and billing.get("available") and billing["total"] >= 0:
        actual_bill = billing["total"]
        credits_used = billing.get("credits_used", actual_bill)
        remaining = CREDIT_LIMIT - actual_bill
        bill_source = "AWS"
    else:
        actual_bill = None
        credits_used = None
        remaining = CREDIT_LIMIT - cumulative
        bill_source = "EC2 estimate"

    runway_hours = remaining / burn_rate if burn_rate > 0 else float("inf")

    lines = [
        f"📊 <b>EC2 Cost Report</b> — Account: <code>{account_id}</code>",
        "",
    ]

    for ic in sorted(instance_costs, key=lambda x: x["session_cost"], reverse=True):
        estimate_flag = " ~" if ic["is_estimate"] else ""
        lines.append(
            f"🖥️ <b>{ic['name']}</b> (<code>{ic['instance_id']}</code>)\n"
            f"   Type: {ic['type']} ({format_cost(ic['hourly_rate'])}/hr{estimate_flag})\n"
            f"   Running: {format_duration(ic['runtime_seconds'])}\n"
            f"   Est. Cost: {format_cost(ic['session_cost'])}"
        )
        lines.append("")

    lines.append("━━━━━━━━━━━━━━━━━━━━")
    lines.append(f"💰 Running EC2 Cost: {format_cost(sum(ic['session_cost'] for ic in instance_costs))}")

    if actual_bill is not None:
        lines.append(f"📈 AWS Account Bill: {format_cost(actual_bill)}")
        ec2_total = sum(ic['session_cost'] for ic in instance_costs)
        non_ec2 = actual_bill - ec2_total
        if non_ec2 > 0.50:
            lines.append(f"📦 Non-EC2 Costs (EBS/S3/etc): ~{format_cost(non_ec2)}")
        if credits_used is not None and credits_used > 0:
            lines.append(f"💳 Credits Used: {format_cost(credits_used)} / {format_cost(CREDIT_LIMIT)}")
    else:
        lines.append(f"📈 Month Cumulative (EC2 est.): {format_cost(cumulative)}")
        lines.append(f"⚠️ <i>AWS billing API unavailable — showing EC2 estimates only</i>")

    lines.append(f"🔴 Remaining Credit: {format_cost(remaining)} ({bill_source})")
    lines.append(f"🔥 Burn Rate: {format_cost(burn_rate)}/hr")

    if runway_hours != float("inf"):
        if runway_hours < 48:
            lines.append(f"⏰ <b>Runway: {format_duration(runway_hours * 3600)}</b>")
        else:
            days = runway_hours / 24
            lines.append(f"⏰ Runway: ~{days:.1f} days")

    if not instance_costs:
        lines = [
            f"📊 <b>EC2 Cost Report</b> — Account: <code>{account_id}</code>",
            "",
            "✅ No running instances.",
        ]
        if actual_bill is not None:
            lines.append(f"📈 AWS Account Bill: {format_cost(actual_bill)}")
            if credits_used is not None and credits_used > 0:
                lines.append(f"💳 Credits Used: {format_cost(credits_used)} / {format_cost(CREDIT_LIMIT)}")
        else:
            lines.append(f"📈 Month Cumulative (EC2 est.): {format_cost(cumulative)}")
        lines.append(f"🔴 Remaining Credit: {format_cost(remaining)}")

    lines.append(f"\n⏰ {ist_now()}")
    return "\n".join(lines)


def build_threshold_alert(account_id: str, pct: int, cumulative: float,
                          burn_rate: float, top_spenders: list, billing: dict = None) -> str:
    """Build threshold breach alert message."""
    if billing and billing.get("available") and billing["total"] >= 0:
        actual_bill = billing["total"]
        remaining = CREDIT_LIMIT - actual_bill
    else:
        actual_bill = None
        remaining = CREDIT_LIMIT - cumulative
    runway_hours = remaining / burn_rate if burn_rate > 0 else float("inf")

    lines = [
        f"🚨 <b>COST ALERT</b> — Account: <code>{account_id}</code>",
        "",
        f"⚠️ Account has crossed <b>{pct}%</b> of {format_cost(CREDIT_LIMIT)} credit limit!",
    ]

    if actual_bill is not None:
        lines.append(f"💰 AWS Account Bill: {format_cost(actual_bill)}")
        credits_used = billing.get("credits_used", actual_bill) if billing else actual_bill
        if credits_used > 0:
            lines.append(f"💳 Credits Used: {format_cost(credits_used)} / {format_cost(CREDIT_LIMIT)}")
        lines.append(f"💰 EC2 Estimate: {format_cost(cumulative)}")
    else:
        lines.append(f"💰 EC2 Estimate: {format_cost(cumulative)}")

    lines.append(f"🔴 Remaining Credit: {format_cost(remaining)}")

    lines.append("")
    lines.append("<b>Top spenders:</b>")

    for i, ic in enumerate(top_spenders[:5], 1):
        lines.append(
            f"  {i}. {ic['name']} ({ic['type']}) — "
            f"{format_cost(ic['session_cost'])} ({format_duration(ic['runtime_seconds'])})"
        )

    lines.append("")
    lines.append(f"🔥 Burn rate: {format_cost(burn_rate)}/hr")

    if runway_hours != float("inf") and runway_hours > 0:
        lines.append(f"⏰ Estimated runway: {format_duration(runway_hours * 3600)}")

    lines.append("")
    lines.append("⚡ <b>ACTION NEEDED:</b> Review and stop unused instances.")
    lines.append(f"\n⏰ {ist_now()}")

    return "\n".join(lines)


def build_long_running_alert(account_id: str, instance_cost: dict) -> str:
    """Build alert for a single instance running too long."""
    ic = instance_cost
    return (
        f"⏱️ <b>Long Running Instance</b> — Account: <code>{account_id}</code>\n\n"
        f"🖥️ <b>{ic['name']}</b> (<code>{ic['instance_id']}</code>)\n"
        f"   Type: {ic['type']} ({format_cost(ic['hourly_rate'])}/hr)\n"
        f"   Running: <b>{format_duration(ic['runtime_seconds'])}</b>\n"
        f"   Est. Cost: <b>{format_cost(ic['session_cost'])}</b>\n\n"
        f"💡 Is this instance still needed?\n\n"
        f"⏰ {ist_now()}"
    )


def build_stopped_alert(account_id: str, stopped: dict) -> str:
    """Build alert when an instance stops."""
    return (
        f"🔴 <b>Instance Stopped</b> — Account: <code>{account_id}</code>\n\n"
        f"🖥️ {stopped['name']} (<code>{stopped['instance_id']}</code>)\n"
        f"   Type: {stopped['type']}\n"
        f"   Total Runtime: {format_duration(stopped['total_runtime_seconds'])}\n"
        f"   Session Cost: {format_cost(stopped['total_cost'])}\n\n"
        f"⏰ {ist_now()}"
    )


def build_new_instance_alert(account_id: str, ic: dict) -> str:
    """Build alert when a new instance is detected running."""
    estimate_flag = " (estimated)" if ic["is_estimate"] else ""
    return (
        f"🟢 <b>Instance Running</b> — Account: <code>{account_id}</code>\n\n"
        f"🖥️ <b>{ic['name']}</b> (<code>{ic['instance_id']}</code>)\n"
        f"   Type: {ic['type']}\n"
        f"   Rate: {format_cost(ic['hourly_rate'])}/hr{estimate_flag}\n\n"
        f"💡 Cost tracking started.\n\n"
        f"⏰ {ist_now()}"
    )


# ──────────────────────────────────────────────────────────────────────
# Lambda Handler
# ──────────────────────────────────────────────────────────────────────

def lambda_handler(event, context):
    """
    Main entry point. Called every 15 minutes by EventBridge.
    """
    now = datetime.now(timezone.utc)
    state = load_state()
    account_id = state["account_id"]

    # 1. Get all running instances
    running = get_running_instances()
    running_ids = set()
    instance_costs = []

    # 2. Calculate cost for each running instance
    for instance in running:
        ic = calculate_instance_cost(instance, state, now)
        instance_costs.append(ic)
        running_ids.add(ic["instance_id"])

    # 3. Detect stopped instances
    stopped_instances = check_stopped_instances(state, running_ids, now)

    # 4. Update cumulative cost
    # Add delta cost from all tracked running instances
    total_delta = sum(ic["delta_cost"] for ic in instance_costs)
    state["cumulative_cost"] += total_delta

    cumulative = state["cumulative_cost"]
    # Also account for costs of instances that ran and stopped this month
    # (already captured in previous deltas)

    # 5. Current burn rate (sum of hourly rates of running instances)
    burn_rate = sum(ic["hourly_rate"] for ic in instance_costs)

    # 5x. Fetch actual AWS billing (best effort)
    billing = get_aws_billing()

    # For threshold checks: use AWS billing if available, else EC2 estimate
    if billing.get("available") and billing["total"] >= 0:
        cost_for_threshold = billing["total"]
    else:
        cost_for_threshold = cumulative

    # ── Alerts ──

    # 5a. New instance alerts
    for ic in instance_costs:
        if ic["is_new"]:
            send_telegram(build_new_instance_alert(account_id, ic))

    # 5b. Stopped instance alerts
    for stopped in stopped_instances:
        send_telegram(build_stopped_alert(account_id, stopped))

    # 5c. Threshold breach alerts (using best available cost data)
    pct_used = (cost_for_threshold / CREDIT_LIMIT) * 100 if CREDIT_LIMIT > 0 else 0
    top_spenders = sorted(instance_costs, key=lambda x: x["session_cost"], reverse=True)

    for threshold in sorted(ALERT_THRESHOLDS):
        if pct_used >= threshold and threshold not in state.get("alerted_thresholds", []):
            send_telegram(build_threshold_alert(account_id, threshold, cumulative, burn_rate, top_spenders, billing))
            state.setdefault("alerted_thresholds", []).append(threshold)

    # 5d. Long-running instance alerts (every INSTANCE_HOUR_ALERT interval)
    for ic in instance_costs:
        runtime_hours = ic["runtime_seconds"] / 3600
        if runtime_hours >= INSTANCE_HOUR_ALERT:
            # Alert at each multiple of INSTANCE_HOUR_ALERT
            prev_info = state["instances"].get(ic["instance_id"], {})
            prev_runtime = prev_info.get("total_runtime_seconds", 0)
            prev_intervals = int(prev_runtime / (INSTANCE_HOUR_ALERT * 3600))
            curr_intervals = int(ic["runtime_seconds"] / (INSTANCE_HOUR_ALERT * 3600))

            if curr_intervals > prev_intervals:
                send_telegram(build_long_running_alert(account_id, ic))

    # 5e. Periodic summary
    last_summary = state.get("last_summary_sent")
    should_send_summary = False

    if last_summary is None:
        should_send_summary = True
    else:
        last_summary_dt = datetime.fromisoformat(last_summary)
        hours_since = (now - last_summary_dt).total_seconds() / 3600
        if hours_since >= SUMMARY_INTERVAL_HOURS:
            should_send_summary = True

    if should_send_summary and (instance_costs or cumulative > 0):
        send_telegram(build_summary_message(account_id, instance_costs, cumulative, burn_rate, billing))
        state["last_summary_sent"] = now.isoformat()

    # 6. Save state
    save_state(state)

    return {
        "statusCode": 200,
        "body": json.dumps({
            "account_id": account_id,
            "running_instances": len(instance_costs),
            "cumulative_cost": round(cumulative, 2),
            "burn_rate": round(burn_rate, 4),
        }),
    }
