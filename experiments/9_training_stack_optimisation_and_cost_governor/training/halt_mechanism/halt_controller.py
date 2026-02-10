
import boto3
import time
import json

########################################
# CONFIG
########################################

REGION = "us-east-1"
BUCKET = "YOUR_BUCKET_NAME"
METRICS_FILE = "/tmp/training_metrics.json"

HEARTBEAT_TIMEOUT = 120
THROUGHPUT_DROP = 0.5
GPU_MIN = 20

ec2 = boto3.client("ec2", region_name=REGION)
ssm = boto3.client("ssm", region_name=REGION)
s3 = boto3.client("s3", region_name=REGION)

baseline_tps = None

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
        return json.load(open(METRICS_FILE))
    except:
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

    global baseline_tps

    if not m:
        return False

    tps = m.get("tokens_per_sec")

    if not tps:
        return False

    if baseline_tps is None:
        baseline_tps = tps
        print("Baseline TPS:", baseline_tps)
        return False

    return tps < baseline_tps * THROUGHPUT_DROP


def gpu_idle(m):

    if not m:
        return False

    gpu = m.get("gpu_util")

    return gpu is not None and gpu < GPU_MIN

########################################
# HALT FLOW
########################################

def trigger_checkpoint(ids):

    print("Triggering checkpoint...")

    ssm.send_command(
        InstanceIds=ids,
        DocumentName="AWS-RunShellScript",
        Parameters={"commands": ["touch /tmp/FORCE_CHECKPOINT"]}
    )


def wait_for_checkpoint():

    print("Waiting for checkpoint upload...")

    while True:
        try:
            s3.head_object(
                Bucket=BUCKET,
                Key="latest/_SUCCESS"
            )
            print("Checkpoint confirmed.")
            return
        except:
            time.sleep(15)


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

########################################
# LOOP
########################################

print("HALT controller running...")

while True:

    m = read_metrics()

    if heartbeat_stalled(m):
        halt_cluster("Heartbeat stalled")

    elif nan_detected(m):
        halt_cluster("NaN detected")

    elif divergence_detected(m):
        halt_cluster("Loss divergence")

    elif throughput_collapsed(m):
        halt_cluster("Throughput collapse")

    elif gpu_idle(m):
        halt_cluster("GPU underutilization")

    time.sleep(20)