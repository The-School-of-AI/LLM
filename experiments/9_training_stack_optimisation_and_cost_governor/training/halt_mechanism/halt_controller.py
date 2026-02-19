
import boto3
import sys
import time
import json

########################################
# CONFIG
########################################

REGION = "us-east-1"
BUCKET = "YOUR_BUCKET_NAME"
# Must match the template's s3.prefix (config.yaml) so we wait for the same sentinel the trainer writes.
S3_PREFIX = "nishant/LLM"
SENTINEL_KEY = f"{S3_PREFIX.rstrip('/')}/latest/_SUCCESS" if S3_PREFIX else "latest/_SUCCESS"

METRICS_FILE = "/tmp/training_metrics.json"

HEARTBEAT_TIMEOUT = 120
THROUGHPUT_DROP = 0.5
# Lowered from 20 — with ZeRO-3 + CPU offloading, GPUs can legitimately sit at
# 15-35% during optimizer steps and gradient reductions.
GPU_MIN = 5
# Halt if GPU memory utilisation is sustained above this percentage (OOM risk).
GPU_MEMORY_MAX = 95

# How many consecutive 20-second polling cycles a sustained trigger must fire
# before the halt is issued. Prevents transient spikes (e.g. mid-checkpoint TPS
# dip, ZeRO allreduce pause) from causing false halts.
# 3 cycles = 60 seconds of sustained bad signal.
CONSECUTIVE_THRESHOLD = 3

# TPS baseline: skip the first N samples (startup / JIT warmup noise), then
# use a rolling window to compute a median baseline.
TPS_WARMUP_SAMPLES = 5
TPS_BASELINE_WINDOW = 10

# Maximum time to wait for the S3 checkpoint sentinel. A 70B ZeRO-3 checkpoint
# can be hundreds of GB; if the upload never completes (e.g. OOM during save),
# the controller should not hang indefinitely burning cost.
CHECKPOINT_TIMEOUT = 3600  # seconds (60 minutes)

ec2 = boto3.client("ec2", region_name=REGION)
ssm = boto3.client("ssm", region_name=REGION)
s3 = boto3.client("s3", region_name=REGION)

tps_samples = []       # accumulates raw TPS readings for baseline establishment
baseline_tps = None
trigger_counts = {}    # trigger_name -> consecutive-fire count

########################################
# INSTANCE DISCOVERY
########################################

def gpu_instances():

    r = ec2.describe_instances(
        Filters=[
            {"Name": "tag:Role", "Values": ["llm-gpu"]},
            {"Name": "instance-state-name", "Values": ["running"]}
        ]
    )

    ids = []

    for res in r["Reservations"]:
        for i in res["Instances"]:
            ids.append(i["InstanceId"])

    return ids

########################################
# METRICS
########################################

def read_metrics():
    try:
        with open(METRICS_FILE) as f:
            return json.load(f)
    except FileNotFoundError:
        return None
    except Exception as e:
        print(f"Warning: failed to read metrics file: {e}")
        return None

########################################
# TRIGGERS
########################################

def heartbeat_stalled(m):

    if not m:
        return False

    hb = m.get("heartbeat")

    return hb and (time.time() - hb > HEARTBEAT_TIMEOUT)


def nan_detected(m):
    return m and m.get("nan")


def divergence_detected(m):
    return m and m.get("diverged")


def throughput_collapsed(m):

    global baseline_tps, tps_samples

    if not m:
        return False

    tps = m.get("tokens_per_sec")

    if not tps:
        return False

    # Accumulate samples until we have enough to skip warmup noise and
    # compute a stable median baseline. The first TPS_WARMUP_SAMPLES readings
    # are discarded (startup / JIT / data-loader warmup). The next
    # TPS_BASELINE_WINDOW readings are used for a median — much more robust
    # than taking the single first reading as baseline.
    total_needed = TPS_WARMUP_SAMPLES + TPS_BASELINE_WINDOW
    if len(tps_samples) < total_needed:
        tps_samples.append(tps)
        if len(tps_samples) == total_needed:
            window = sorted(tps_samples[TPS_WARMUP_SAMPLES:])
            baseline_tps = window[len(window) // 2]
            print(f"Baseline TPS established (median of {TPS_BASELINE_WINDOW} post-warmup samples): {baseline_tps:.1f}")
        return False

    return tps < baseline_tps * THROUGHPUT_DROP


def gpu_idle(m):

    if not m:
        return False

    gpu = m.get("gpu_util")

    return gpu is not None and gpu < GPU_MIN


def memory_pressure(m):
    """Trigger if GPU memory utilisation is sustained above GPU_MEMORY_MAX.

    For 70B training, GPU OOM is a primary failure mode and approaching the
    limit is a pre-crash signal the trainer itself cannot report after the fact.
    """
    if not m:
        return False

    mem = m.get("gpu_memory_pct")

    return mem is not None and mem > GPU_MEMORY_MAX


########################################
# CONSECUTIVE GATE
########################################

def check_trigger(name, fired):
    """Return True only after `name` has fired CONSECUTIVE_THRESHOLD times in a row.

    Resets the counter on the first non-fired cycle, so a single transient blip
    (e.g. a mid-checkpoint TPS dip) does not trigger a halt.
    """
    if fired:
        trigger_counts[name] = trigger_counts.get(name, 0) + 1
        if trigger_counts[name] >= CONSECUTIVE_THRESHOLD:
            return True
    else:
        trigger_counts[name] = 0
    return False

########################################
# HALT FLOW
########################################

def trigger_checkpoint(ids):

    print("Triggering checkpoint via SSM...")

    response = ssm.send_command(
        InstanceIds=ids,
        DocumentName="AWS-RunShellScript",
        Parameters={"commands": ["touch /tmp/FORCE_CHECKPOINT"]}
    )
    command_id = response["Command"]["CommandId"]

    # Verify the command was actually received and executed on every instance.
    # SSM can silently drop commands against degraded instances, so polling the
    # invocation status is necessary before we trust the signal was delivered.
    time.sleep(5)
    for instance_id in ids:
        for attempt in range(12):  # up to ~60 s of polling
            try:
                inv = ssm.get_command_invocation(
                    CommandId=command_id,
                    InstanceId=instance_id,
                )
                status = inv["Status"]
                if status == "Success":
                    print(f"  SSM confirmed on {instance_id}")
                    break
                elif status in ("Failed", "Cancelled", "TimedOut"):
                    print(f"  WARNING: SSM command {status} on {instance_id} — halt file may not have been created")
                    break
                # InProgress / Pending — keep waiting
                time.sleep(5)
            except Exception as e:
                print(f"  WARNING: Could not verify SSM on {instance_id}: {e}")
                time.sleep(5)


def wait_for_checkpoint():
    """Poll S3 for the halt sentinel. Returns True if found, False if timeout hit.

    Without a timeout, a trainer that OOMs or hangs during the checkpoint save
    itself would leave the controller polling indefinitely while GPU instances
    continue to run and incur cost.
    """
    print(f"Waiting for checkpoint sentinel (timeout: {CHECKPOINT_TIMEOUT}s)...")
    deadline = time.time() + CHECKPOINT_TIMEOUT

    while time.time() < deadline:
        try:
            s3.head_object(Bucket=BUCKET, Key=SENTINEL_KEY)
            print("Checkpoint confirmed.")
            return True
        except Exception:
            time.sleep(15)

    print(f"ERROR: Checkpoint sentinel not found after {CHECKPOINT_TIMEOUT}s. "
          "Proceeding with termination to avoid runaway cost.")
    return False


def terminate(ids):
    print("Terminating instances...")
    ec2.terminate_instances(InstanceIds=ids)


def verify():

    time.sleep(60)

    if gpu_instances():
        raise RuntimeError("Orphan instances detected!")

    print("No orphan GPUs detected.")


def halt_cluster(reason):

    ids = gpu_instances()

    if not ids:
        print("No running GPU instances found.")
        return

    print("HALT TRIGGERED:", reason)

    trigger_checkpoint(ids)
    wait_for_checkpoint()
    terminate(ids)
    verify()

    print("Cluster halted safely.")
    sys.exit(0)

########################################
# LOOP
########################################

print("HALT controller running...")

while True:

    m = read_metrics()

    # NaN and divergence are unambiguous single-step signals — halt immediately.
    if nan_detected(m):
        halt_cluster("NaN detected")

    if divergence_detected(m):
        halt_cluster("Loss divergence")

    # Sustained triggers require CONSECUTIVE_THRESHOLD consecutive bad readings
    # before firing, to avoid false halts from transient events (checkpoint saves,
    # ZeRO allreduce pauses, etc.).
    if check_trigger("heartbeat", heartbeat_stalled(m)):
        halt_cluster("Heartbeat stalled")

    if check_trigger("throughput", throughput_collapsed(m)):
        halt_cluster("Throughput collapse")

    if check_trigger("gpu_idle", gpu_idle(m)):
        halt_cluster("GPU underutilization")

    if check_trigger("memory", memory_pressure(m)):
        halt_cluster("GPU memory pressure (OOM risk)")

    time.sleep(20)