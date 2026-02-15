# P12 Deployment Guide

End-to-end deployment of the self-hosted observability stack for 70B LLM training.
Two EC2 instances: a **DB instance** (ClickHouse) and one or more **Training instances** (training loop + Vector sidecar).

---

## Table of Contents

1. [Architecture Overview](#1-architecture-overview)
2. [Prerequisites](#2-prerequisites)
3. [Phase 0: AWS Infrastructure Provisioning](#3-phase-0-aws-infrastructure-provisioning)
4. [Phase 1: DB Instance Deployment](#4-phase-1-db-instance-deployment)
5. [Phase 2: Training Instance Deployment](#5-phase-2-training-instance-deployment)
6. [Phase 3: Verification](#6-phase-3-verification)
7. [Health Checks & Monitoring](#7-health-checks--monitoring)
8. [Script Reference](#8-script-reference)
9. [Security Model](#9-security-model)
10. [Operational Runbook](#10-operational-runbook)

---

## 1. Architecture Overview

```
Training Instance(s)                            DB Instance
┌────────────────────────────────────┐          ┌──────────────────────────┐
│                                    │          │                          │
│  train.py                          │          │  ClickHouse (Docker)     │
│   └─ TrainingOps facade            │          │    HTTPS :8443           │
│       ├─ JSONLogger ───┐           │          │    HTTP  :8123 (local)   │
│       ├─ SystemMetrics ┤  .jsonl   │          │                          │
│       ├─ MetricsServer │           │          │  Tables:                 │
│       └─ CheckpointReg │           │          │    logs                  │
│                        │           │          │    metric_points         │
│  Vector sidecar ───────┘───────────┼─ HTTPS ─▶│    checkpoints           │
│    ├─ to_raw_logs                  │  :8443   │    events, runs          │
│    ├─ to_metric_points             │  TLS+auth│    metric_arrays         │
│    └─ to_checkpoints               │          │                          │
│                                    │          │  Users:                  │
│  /tmp/training_logs/*.jsonl        │          │    p12_writer (INSERT)   │
│                                    │          │    p12_reader (SELECT)   │
│  Healthcheck cron (1m)             │          │    default (localhost)   │
│  ┌──────────────────────┐          │          │                          │
│  │ VectorAlive          │          │          │  Healthcheck cron (1m)   │
│  │ ClickHouseReachable  │──▶ CW    │          │  ┌────────────────────┐  │
│  │ JsonlFreshness       │          │          │  │ Alive              │  │
│  └──────────────────────┘          │          │  │ DiskUsedPercent    │──▶ CW
└────────────────────────────────────┘          │  │ LogsRowCount       │  │
                                                │  └────────────────────┘  │
S3 Bucket (p12-config-XXXX)                     └──────────────────────────┘
┌────────────────────────┐
│  certs/ca.crt          │  ◀── setup-auth.sh uploads
│  vector/vector.toml    │  ◀── setup-auth.sh uploads
└────────────────────────┘
         ▲
         │  curl (public read)
         │
  userdata-vector.sh (at EC2 boot)

SSM Parameter Store
┌────────────────────────────────────┐
│  /p12/training/clickhouse-password │  (SecureString)
│  /p12/training/clickhouse-endpoint │  (String)
│  /p12/dashboard/clickhouse-password│  (SecureString)
└────────────────────────────────────┘
```

**Data flow**: Training code writes JSONL to local NVMe. Vector tails those files, transforms each line, and pushes to ClickHouse over HTTPS. No data leaves the VPC unencrypted.

**Credential flow**: `setup-auth.sh` generates passwords + TLS certs, uploads non-sensitive config (CA cert, vector.toml) to a public S3 bucket, and stores secrets in SSM Parameter Store. Training instances pull both at boot via `userdata-vector.sh`.

---

## 2. Prerequisites

### Pre-existing Resources (must exist before deployment)

| Resource | Purpose | Notes |
|----------|---------|-------|
| VPC with private subnets | Network isolation | Training + DB in same VPC |
| VPC endpoints for SSM **or** NAT gateway | SSM Session Manager connectivity | Private-subnet instances need to reach SSM service |
| SSM Session Manager plugin | Operator shell access | Install on operator workstation: `session-manager-plugin` |
| AWS CLI v2 | Infrastructure commands | On operator workstation |
| `openssl` >= 1.1.1 | TLS cert generation | On operator workstation |

### Created by This Guide (not pre-existing)

| Resource | Created in | Purpose |
|----------|-----------|---------|
| IAM role: `p12-clickhouse-db-role` | Phase 0, Step 1 | DB instance: SSM access + CloudWatch push |
| IAM role: `p12-training-instance-role` | Phase 0, Step 2 | Training instance: SSM param read + CloudWatch push |
| Security group: `p12-clickhouse-sg` | Phase 0, Step 3 | Restricts port 8443 to training + dashboard CIDRs |
| DB EC2 instance (t3.medium) | Phase 0, Step 4 | ClickHouse host |
| EBS gp3 volume | Phase 0, Step 5 | ClickHouse data at `/data/clickhouse` |
| S3 bucket: `p12-config-{ACCOUNT}` | Phase 1 (setup-auth.sh) | Public CA cert + vector.toml distribution |
| SSM parameters: `/p12/*` | Phase 1 (setup-auth.sh) | Encrypted credential storage |
| SNS topic: `p12-alerts` | Phase 3, Health Checks | Alarm notification delivery |
| CloudWatch alarms (6x) | Phase 3, Health Checks | DB + training health monitoring |

### Operator IAM Permissions

The IAM role/user running the deployment commands (your operator workstation or CI runner) needs these permissions:

```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Sid": "EC2InstanceAndEBS",
      "Effect": "Allow",
      "Action": [
        "ec2:RunInstances",
        "ec2:DescribeInstances",
        "ec2:DescribeInstanceStatus",
        "ec2:CreateVolume",
        "ec2:AttachVolume",
        "ec2:DetachVolume",
        "ec2:DescribeVolumes",
        "ec2:ModifyVolume",
        "ec2:CreateSnapshot",
        "ec2:DescribeSnapshots",
        "ec2:CreateTags",
        "ec2:DescribeSubnets",
        "ec2:DescribeImages"
      ],
      "Resource": "*"
    },
    {
      "Sid": "SecurityGroupManagement",
      "Effect": "Allow",
      "Action": [
        "ec2:AuthorizeSecurityGroupIngress",
        "ec2:RevokeSecurityGroupIngress",
        "ec2:DescribeSecurityGroups",
        "ec2:DescribeSecurityGroupRules",
        "ec2:CreateSecurityGroup",
        "ec2:DeleteSecurityGroup"
      ],
      "Resource": "*"
    },
    {
      "Sid": "IAMRolesAndProfiles",
      "Effect": "Allow",
      "Action": [
        "iam:CreateRole",
        "iam:PutRolePolicy",
        "iam:AttachRolePolicy",
        "iam:CreateInstanceProfile",
        "iam:AddRoleToInstanceProfile",
        "iam:PassRole",
        "iam:GetRole",
        "iam:GetInstanceProfile"
      ],
      "Resource": [
        "arn:aws:iam::*:role/p12-*",
        "arn:aws:iam::*:instance-profile/p12-*"
      ]
    },
    {
      "Sid": "CloudWatchAlarms",
      "Effect": "Allow",
      "Action": [
        "cloudwatch:PutMetricData",
        "cloudwatch:PutMetricAlarm",
        "cloudwatch:DescribeAlarms",
        "cloudwatch:DeleteAlarms",
        "cloudwatch:ListMetrics"
      ],
      "Resource": "*"
    },
    {
      "Sid": "SNSForAlarms",
      "Effect": "Allow",
      "Action": [
        "sns:CreateTopic",
        "sns:Subscribe",
        "sns:Publish",
        "sns:ListTopics"
      ],
      "Resource": "*"
    },
    {
      "Sid": "SSMParameterStore",
      "Effect": "Allow",
      "Action": [
        "ssm:GetParameter",
        "ssm:GetParameters",
        "ssm:PutParameter",
        "ssm:DescribeParameters",
        "ssm:StartSession"
      ],
      "Resource": [
        "arn:aws:ssm:*:*:parameter/p12/*",
        "arn:aws:ssm:*:*:session/*"
      ]
    },
    {
      "Sid": "S3ForConfigDistribution",
      "Effect": "Allow",
      "Action": [
        "s3:CreateBucket",
        "s3:GetObject",
        "s3:PutObject",
        "s3:ListBucket",
        "s3:PutBucketPolicy",
        "s3:PutBucketPublicAccessBlock"
      ],
      "Resource": [
        "arn:aws:s3:::p12-config-*",
        "arn:aws:s3:::p12-config-*/*"
      ]
    },
    {
      "Sid": "DLMForSnapshots",
      "Effect": "Allow",
      "Action": [
        "dlm:CreateLifecyclePolicy",
        "dlm:GetLifecyclePolicy",
        "dlm:UpdateLifecyclePolicy"
      ],
      "Resource": "*"
    }
  ]
}
```

### Software on Training Instance

- Python 3.8+ with `psutil`, `pyyaml`, `numpy`, `pynvml`
- `awscli`, `jq`, `curl`, `bc` (installed by userdata script)
- Vector >= 0.30 (installed by userdata script)

### DB Instance Sizing

| Scenario | Instance | vCPUs | RAM | EBS | Cost/mo |
|----------|----------|-------|-----|-----|---------|
| 1-2 concurrent runs | `t3.small` | 2 | 2 GB | 20 GB gp3 | ~$15 |
| 3-5 concurrent runs | `t3.medium` | 2 | 4 GB | 50 GB gp3 | ~$30 |
| 10+ runs, long retention | `t3.large` | 2 | 8 GB | 100 GB gp3 | ~$60 |

ClickHouse compresses MergeTree data 5-10x. A month of 70B training metrics at 10-step logging interval is typically under 1 GB compressed.

---

## 3. Phase 0: AWS Infrastructure Provisioning

Run these steps from the operator workstation. They create all AWS resources needed before deploying software.

**Set environment variables** used throughout this phase:

```bash
REGION="us-east-1"
VPC_ID="vpc-xxxxxxxxxx"
SUBNET_ID="subnet-xxxxxxxxxx"           # private subnet for the DB instance
TRAINING_SUBNET_CIDR="10.0.1.0/24"     # CIDR of the training instance subnet
DASHBOARD_SUBNET_CIDR="10.0.2.0/24"    # CIDR of the dashboard subnet
```

### Step 1: Create IAM Role for the DB Instance

The DB instance needs SSM Session Manager (shell access — no SSH) and CloudWatch metrics push (health checks).

```bash
# Create the role with EC2 trust policy
aws iam create-role \
  --role-name p12-clickhouse-db-role \
  --assume-role-policy-document '{
    "Version": "2012-10-17",
    "Statement": [{
      "Effect": "Allow",
      "Principal": {"Service": "ec2.amazonaws.com"},
      "Action": "sts:AssumeRole"
    }]
  }'

# Attach SSM managed policy (required for SSM Session Manager access)
aws iam attach-role-policy \
  --role-name p12-clickhouse-db-role \
  --policy-arn arn:aws:iam::aws:policy/AmazonSSMManagedInstanceCore

# Inline policy: CloudWatch metrics push (for healthcheck script)
aws iam put-role-policy \
  --role-name p12-clickhouse-db-role \
  --policy-name p12-db-cloudwatch \
  --policy-document '{
    "Version": "2012-10-17",
    "Statement": [{
      "Effect": "Allow",
      "Action": ["cloudwatch:PutMetricData"],
      "Resource": "*"
    }]
  }'

# Inline policy: S3 config bucket (setup-auth.sh creates bucket, uploads ca.crt + vector.toml)
aws iam put-role-policy \
  --role-name p12-clickhouse-db-role \
  --policy-name p12-db-s3-config \
  --policy-document '{
    "Version": "2012-10-17",
    "Statement": [{
      "Effect": "Allow",
      "Action": [
        "s3:CreateBucket",
        "s3:ListBucket",
        "s3:PutObject",
        "s3:GetObject",
        "s3:PutBucketPolicy",
        "s3:PutBucketPublicAccessBlock",
        "s3:GetBucketLocation"
      ],
      "Resource": [
        "arn:aws:s3:::p12-config-*",
        "arn:aws:s3:::p12-config-*/*"
      ]
    }]
  }'

# Create instance profile and attach role
aws iam create-instance-profile \
  --instance-profile-name p12-clickhouse-db-profile

aws iam add-role-to-instance-profile \
  --instance-profile-name p12-clickhouse-db-profile \
  --role-name p12-clickhouse-db-role

# Wait for IAM propagation (eventually consistent)
echo "Waiting 10s for IAM propagation..."
sleep 10
```

### Step 2: Create IAM Role for Training Instances

Training instances need SSM Parameter Store read (for passwords), CloudWatch push (for health metrics), and SSM Session Manager (for operator access). S3 read is **not** needed — `ca.crt` and `vector.toml` are public S3 objects fetched via `curl`.

```bash
# Create the role
aws iam create-role \
  --role-name p12-training-instance-role \
  --assume-role-policy-document '{
    "Version": "2012-10-17",
    "Statement": [{
      "Effect": "Allow",
      "Principal": {"Service": "ec2.amazonaws.com"},
      "Action": "sts:AssumeRole"
    }]
  }'

# Attach SSM managed policy (required for SSM Session Manager access)
aws iam attach-role-policy \
  --role-name p12-training-instance-role \
  --policy-arn arn:aws:iam::aws:policy/AmazonSSMManagedInstanceCore

# Inline policy: SSM Parameter Store read (passwords) + CloudWatch push (health metrics)
ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
aws iam put-role-policy \
  --role-name p12-training-instance-role \
  --policy-name p12-training-access \
  --policy-document "{
    \"Version\": \"2012-10-17\",
    \"Statement\": [
      {
        \"Sid\": \"SSMReadCreds\",
        \"Effect\": \"Allow\",
        \"Action\": [\"ssm:GetParameter\", \"ssm:GetParameters\"],
        \"Resource\": \"arn:aws:ssm:*:${ACCOUNT_ID}:parameter/p12/training/*\"
      },
      {
        \"Sid\": \"CloudWatchPush\",
        \"Effect\": \"Allow\",
        \"Action\": [\"cloudwatch:PutMetricData\"],
        \"Resource\": \"*\"
      }
    ]
  }"

# Create instance profile
aws iam create-instance-profile \
  --instance-profile-name p12-training-instance-profile

aws iam add-role-to-instance-profile \
  --instance-profile-name p12-training-instance-profile \
  --role-name p12-training-instance-role

sleep 10
```

### Step 3: Create Security Group for ClickHouse

Restricts port 8443 (HTTPS) to training and dashboard subnets only. No SSH port — all access is via SSM Session Manager.

```bash
# Create the security group
DB_SG_ID=$(aws ec2 create-security-group \
  --group-name p12-clickhouse-sg \
  --description "P12 ClickHouse DB - restricted access" \
  --vpc-id "$VPC_ID" \
  --query 'GroupId' --output text)

echo "Created security group: $DB_SG_ID"

# Rule 1: ClickHouse HTTPS (8443) from training subnet only
aws ec2 authorize-security-group-ingress \
  --group-id "$DB_SG_ID" \
  --protocol tcp --port 8443 \
  --cidr "$TRAINING_SUBNET_CIDR" \
  --tag-specifications "ResourceType=security-group-rule,Tags=[{Key=Name,Value=p12-training-to-clickhouse}]"

# Rule 2: ClickHouse HTTPS (8443) from dashboard subnet only
aws ec2 authorize-security-group-ingress \
  --group-id "$DB_SG_ID" \
  --protocol tcp --port 8443 \
  --cidr "$DASHBOARD_SUBNET_CIDR" \
  --tag-specifications "ResourceType=security-group-rule,Tags=[{Key=Name,Value=p12-dashboard-to-clickhouse}]"

# NO rule for port 8123 — it's localhost-only inside the container
# NO SSH rule — all access is via SSM Session Manager

# Verify
aws ec2 describe-security-group-rules \
  --filters "Name=group-id,Values=${DB_SG_ID}" \
  --query 'SecurityGroupRules[].[SecurityGroupRuleId,FromPort,ToPort,CidrIpv4]' \
  --output table
```

### Step 4: Launch the DB EC2 Instance

```bash
# Find latest Ubuntu 22.04 AMI
AMI_ID=$(aws ec2 describe-images \
  --owners 099720109477 \
  --filters "Name=name,Values=ubuntu/images/hvm-ssd/ubuntu-jammy-22.04-amd64-server-*" \
             "Name=state,Values=available" \
  --query 'sort_by(Images, &CreationDate)[-1].ImageId' \
  --output text --region "$REGION")

echo "Using AMI: $AMI_ID"

# User data script: installs Docker + AWS CLI on the DB instance
DB_USERDATA=$(cat <<'USERDATA'
#!/bin/bash
set -euo pipefail
exec > >(tee /var/log/p12-db-userdata.log) 2>&1

echo "P12 DB instance bootstrap started at $(date -u)"

export DEBIAN_FRONTEND=noninteractive
apt-get update -qq
apt-get install -y -qq docker.io docker-compose-v2 awscli jq git

systemctl enable docker
systemctl start docker

# Add ubuntu user to docker group
usermod -aG docker ubuntu

echo "P12 DB instance bootstrap completed at $(date -u)"
USERDATA
)

# Launch the instance
DB_INSTANCE_ID=$(aws ec2 run-instances \
  --image-id "$AMI_ID" \
  --instance-type t3.medium \
  --subnet-id "$SUBNET_ID" \
  --security-group-ids "$DB_SG_ID" \
  --iam-instance-profile Name=p12-clickhouse-db-profile \
  --user-data "$DB_USERDATA" \
  --block-device-mappings '[{"DeviceName":"/dev/sda1","Ebs":{"VolumeSize":20,"VolumeType":"gp3"}}]' \
  --tag-specifications "ResourceType=instance,Tags=[{Key=Name,Value=p12-clickhouse-db},{Key=Project,Value=p12}]" \
  --query 'Instances[0].InstanceId' --output text \
  --region "$REGION")

echo "Launched DB instance: $DB_INSTANCE_ID"

# Wait for it to be running
aws ec2 wait instance-running --instance-ids "$DB_INSTANCE_ID" --region "$REGION"

# Get the private IP (save this — needed for setup-auth.sh and training config)
DB_PRIVATE_IP=$(aws ec2 describe-instances \
  --instance-ids "$DB_INSTANCE_ID" \
  --query 'Reservations[0].Instances[0].PrivateIpAddress' \
  --output text --region "$REGION")

echo "DB instance private IP: $DB_PRIVATE_IP"
```

### Step 5: Create and Attach the EBS gp3 Data Volume

ClickHouse data lives on a dedicated EBS volume, separate from the root volume. This allows independent snapshots, resizing, and migration.

```bash
# Get the AZ of the DB instance (EBS must be in the same AZ)
AZ=$(aws ec2 describe-instances \
  --instance-ids "$DB_INSTANCE_ID" \
  --query 'Reservations[0].Instances[0].Placement.AvailabilityZone' \
  --output text --region "$REGION")

VOLUME_SIZE_GB=100

# Create the gp3 volume
VOLUME_ID=$(aws ec2 create-volume \
  --volume-type gp3 \
  --size "$VOLUME_SIZE_GB" \
  --iops 3000 \
  --throughput 125 \
  --availability-zone "$AZ" \
  --tag-specifications "ResourceType=volume,Tags=[{Key=Name,Value=p12-clickhouse-data},{Key=Project,Value=p12}]" \
  --query 'VolumeId' --output text \
  --region "$REGION")

echo "Created volume: $VOLUME_ID"

# Wait for it to become available
aws ec2 wait volume-available --volume-ids "$VOLUME_ID" --region "$REGION"

# Attach to the DB instance
aws ec2 attach-volume \
  --volume-id "$VOLUME_ID" \
  --instance-id "$DB_INSTANCE_ID" \
  --device /dev/xvdf \
  --region "$REGION"

aws ec2 wait volume-in-use --volume-ids "$VOLUME_ID" --region "$REGION"
echo "Volume $VOLUME_ID attached to $DB_INSTANCE_ID as /dev/xvdf"
```

**Why gp3?** gp3 provides consistent 3000 IOPS baseline regardless of volume size (unlike gp2 where IOPS scales with size). For a 100 GB volume, gp3 is both cheaper and faster than gp2.

### Step 6: Connect via SSM and Deploy ClickHouse

Wait ~1-2 minutes for user data to finish, then connect via SSM Session Manager (no SSH key needed):
Follow below steps to install SSM on top of aws cli

``` bash
curl "https://s3.amazonaws.com/session-manager-downloads/plugin/latest/mac_arm64/session-manager-plugin.pkg" -o "session-manager-plugin.pkg"
sudo installer -pkg session-manager-plugin.pkg -target /
sudo ln -s /usr/local/sessionmanagerplugin/bin/session-manager-plugin /usr/local/bin/session-manager-plugin
```

```bash
# From operator workstation
aws ssm start-session --target "$DB_INSTANCE_ID" --region "$REGION"
```

Once connected, run the all-in-one setup script:

```bash
sudo -iu ubuntu

# Clone the repo and run setup
git clone https://github.com/<org>/<repo>.git /tmp/p12-repo
bash /tmp/p12-repo/experiments/12_training_operations/components/clickhouse/setup-db-instance.sh
```

The script prompts for passwords and subnet CIDRs (or read from env vars for non-interactive mode):

```bash
# Non-interactive mode:
P12_WRITER_PASSWORD="$(openssl rand -base64 18)" \
P12_READER_PASSWORD="$(openssl rand -base64 18)" \
TRAINING_SUBNET_CIDR=10.0.1.0/24 \
DASHBOARD_SUBNET_CIDR=10.0.2.0/24 \
P12_REGION=us-east-1 \
bash /tmp/p12-repo/experiments/12_training_operations/components/clickhouse/setup-db-instance.sh
```

`setup-db-instance.sh` handles: EBS format/mount → git clone → auth setup (TLS + S3 + SSM) → docker compose up → verify.

---

## 4. Phase 1: DB Instance Deployment

Phase 0 handles the AWS infrastructure. This phase covers the software configuration on the DB instance.

### Option A: Automated (recommended for fresh instances)

If you followed Phase 0 Step 6, the setup is already complete. `setup-db-instance.sh` runs all 5 steps below automatically.

**What `setup-db-instance.sh` does** (5 steps):

| Step | Action | Rationale |
|------|--------|-----------|
| 1. EBS setup | Formats `/dev/xvdf` as ext4, mounts at `/data/clickhouse`, adds to fstab | ClickHouse data persists across container restarts. Nitro NVMe naming is auto-detected. |
| 2. Git clone | Pulls `clickhouse/` directory from the repo | Gets all config, schema, and scripts onto the instance. |
| 3. Auth setup | Runs `setup-auth.sh` (see below) | Generates TLS certs, user configs, uploads to S3/SSM. |
| 4. Docker start | `docker compose up -d` | Starts ClickHouse. Schema applied automatically on first boot via `initdb.d/`. |
| 5. Verify | Waits for health, shows tables | Confirms ClickHouse is serving and schema exists. |

### Option B: Step-by-step (existing instances or debugging)

#### Step 1: Generate auth credentials + TLS certs

```bash
cd ~/clickhouse
bash setup-auth.sh
```

**What `setup-auth.sh` does** (7 steps):

| Step | Action | Output |
|------|--------|--------|
| 1. Collect inputs | Prompts for passwords (min 12 chars) + subnet CIDRs + DB private IP | — |
| 2. Hash passwords | SHA-256 hashes for ClickHouse `password_sha256_hex` | — |
| 3. Generate users XML | Substitutes hashes + CIDRs into `p12-users.xml.template` | `users.d/p12-users.xml` |
| 4. Generate TLS certs | CA key+cert (10yr), then server cert signed by CA (825 days) | `tls/ca/ca.key`, `tls/ca/ca.crt`, `tls/server.crt`, `tls/server.key` |
| 5. Upload to S3 | Creates `p12-config-{ACCOUNT_ID}` bucket, uploads `ca.crt` + `vector.toml` | Public-read S3 bucket |
| 6. Store in SSM | Writer password (SecureString), reader password (SecureString), endpoint (String) | `/p12/training/*`, `/p12/dashboard/*` |
| 7. Write .env files | Local reference files (not committed to git) | `training-instance.env`, `dashboard.env` |

**Why a public S3 bucket for CA cert + vector.toml?** These are non-sensitive configuration files. Making them public avoids requiring IAM credentials just to pull the Vector config at boot time. Passwords never touch S3 — they go to SSM Parameter Store with encryption.

**Why SSM for passwords?** SecureString parameters are encrypted at rest with KMS. The training instance IAM role grants read-only access to the specific parameter paths. No `.env` files need to be SCPed between instances.

#### Step 2: Generate TLS certificates (run by setup-auth.sh, or manually)

```bash
# CA (run once, guard the key)
bash tls/generate-ca.sh

# Server cert (needs DB private IP for SAN)
bash tls/generate-server-cert.sh 10.0.1.5
```

**TLS design decisions:**
- **One-way TLS** (server cert only). Clients verify the server using the CA cert. No client certificates — authentication is password-based.
- **SAN includes the DB private IP**, so TLS hostname verification works without DNS.
- **CA validity: 10 years.** Server cert: 825 days (Apple/browser limit, good practice even for internal certs).
- **4096-bit RSA keys.** Overkill for internal traffic but costs nothing and future-proofs.

#### Step 3: Start ClickHouse

```bash
sudo docker compose up -d
```

Docker Compose mounts:
- `/data/clickhouse/data` → ClickHouse data directory (EBS-backed, survives container recreation)
- `/data/clickhouse/logs` → ClickHouse server logs
- `initdb.d/` → Schema SQL files (executed on first start only)
- `users.d/` → User config with password hashes + CIDR restrictions
- `config.d/` → HTTPS/TLS configuration
- `tls/` → Certificate files

**Port mapping:**
- `127.0.0.1:8123` → HTTP (localhost only — admin, healthchecks)
- `0.0.0.0:8443` → HTTPS (network-accessible, requires auth + TLS)

#### Step 4: Install healthcheck cron

```bash
sudo cp healthcheck/clickhouse-healthcheck.sh /usr/local/bin/
sudo chmod +x /usr/local/bin/clickhouse-healthcheck.sh

echo "* * * * * root /usr/local/bin/clickhouse-healthcheck.sh >> /var/log/p12-clickhouse-healthcheck.log 2>&1" \
  | sudo tee /etc/cron.d/p12-clickhouse-healthcheck
```

This pushes 6 CloudWatch metrics every minute (see [Health Checks](#7-health-checks--monitoring)).

#### Step 5: Verify

```bash
# ClickHouse responding?
sudo docker exec p12-clickhouse clickhouse-client --query "SELECT 1"

# Schema exists?
sudo docker exec p12-clickhouse clickhouse-client --query "SHOW TABLES FROM training_observability"
# Expected: checkpoints, events, logs, metric_arrays, metric_points, runs

# HTTPS works?
curl -sk https://localhost:8443/ping
# Expected: Ok.

# Authenticated query?
curl -sk "https://localhost:8443/?user=p12_reader&password=<password>&query=SELECT+1"
# Expected: 1
```

---

## 5. Phase 2: Training Instance Deployment

### Option A: EC2 User Data (recommended)

Paste `userdata-vector.sh` into the EC2 launch template's User Data field. Edit the two config variables at the top:

```bash
P12_CONFIG_BUCKET="p12-config-XXXXXXXXXXXX"  # from setup-auth.sh output
P12_REGION="us-east-1"
```

Ensure the instance has the `p12-training-instance-profile` IAM role attached (created in Phase 0, Step 2).

**What `userdata-vector.sh` does** (8 steps):

| Step | Action | Rationale |
|------|--------|-----------|
| 1. System packages | Installs `awscli`, `jq`, `curl`, `bc` | Dependencies for credential retrieval and healthcheck. |
| 2. Install Vector | Vector >= 0.30 from official installer | Log shipper. Idempotent — skips if already installed. |
| 3. Create directories | `/etc/p12`, `/tmp/training_logs`, `/var/lib/vector` | Config home, log landing zone, Vector buffer directory. |
| 4. Pull config from S3 | Downloads `ca.crt` and `vector.toml` via public HTTPS | Non-sensitive config. No IAM needed for this step. |
| 5. Pull credentials from SSM | Reads password + endpoint from Parameter Store | Secrets never stored on disk in plaintext except in the restricted `/etc/p12/vector.env` (mode 600). |
| 6. Create systemd service | `p12-vector.service` with hardening (`NoNewPrivileges`, `ProtectSystem=strict`) | Vector runs as a system service, auto-restarts on crash. |
| 7. Install healthcheck | Cron job pushing 5 CloudWatch metrics every minute | Operational visibility into sidecar health. |
| 8. Verify | Checks Vector is running | Fails loudly in `/var/log/p12-userdata.log` if something goes wrong. |

**Why User Data?** Training instances are often ephemeral (spot, preemptible). User Data runs at every boot, making each instance self-configuring. No manual SSH required.

**Why systemd for Vector?** Systemd provides process supervision (auto-restart), resource limits, and security hardening. The `ProtectSystem=strict` + `ReadWritePaths` combination prevents Vector from writing anywhere except its buffer dir and the log directory.

#### Example: Launch a training instance with user data

```bash
TRAINING_AMI="ami-xxxxxxxxxx"       # your GPU AMI
TRAINING_SUBNET="subnet-xxxxxxxxxx"
TRAINING_SG="sg-xxxxxxxxxx"         # your training instance SG

aws ec2 run-instances \
  --image-id "$TRAINING_AMI" \
  --instance-type p4d.24xlarge \
  --subnet-id "$TRAINING_SUBNET" \
  --security-group-ids "$TRAINING_SG" \
  --iam-instance-profile Name=p12-training-instance-profile \
  --user-data file://sidecar_agent/userdata-vector.sh \
  --tag-specifications "ResourceType=instance,Tags=[{Key=Name,Value=p12-training-node},{Key=Project,Value=p12}]" \
  --region "$REGION"
```

### Option B: Manual setup

```bash
# 1. Install Vector
curl --proto '=https' --tlsv1.2 -sSfL https://sh.vector.dev | bash -s -- -y --prefix /usr/local

# 2. Create directories
sudo mkdir -p /etc/p12 /tmp/training_logs /var/lib/vector
sudo chown ubuntu:ubuntu /tmp/training_logs /var/lib/vector

# 3. Copy config
scp db-instance:~/clickhouse/tls/ca/ca.crt /etc/p12/ca.crt
scp db-instance:~/clickhouse/training-instance.env /etc/p12/vector.env
cp sidecar_agent/vector.toml /etc/p12/vector.toml

# 4. Source credentials
export $(cat /etc/p12/vector.env | grep -v '^#' | xargs)

# 5. Run Vector
vector --config /etc/p12/vector.toml --data-dir /var/lib/vector
```

### Python dependencies (on training instances)

```bash
pip install psutil pyyaml numpy pynvml
```

### Train.py integration (minimal)

```python
from components import TrainingOps

ops = TrainingOps(
    run_id="run_2026_02_15_70b_v5",
    rank=int(os.environ.get("RANK", 0)),
    default_context={"model": "70B_v5", "cluster": "us-east-1-p4d"},
)

for step, batch in enumerate(dataloader):
    loss = train_step(batch)
    if step % 10 == 0:
        ops.log_step(step=step, metrics={"loss": loss.item(), "lr": lr})

ops.shutdown()
```

---

## 6. Phase 3: Verification

Run these checks after both instances are deployed.

### From the DB instance

```bash
# Row counts (should increase as training runs)
sudo docker exec p12-clickhouse clickhouse-client --query \
  "SELECT count() FROM training_observability.logs"

sudo docker exec p12-clickhouse clickhouse-client --query \
  "SELECT metric, count(), min(value), max(value)
   FROM training_observability.metric_points GROUP BY metric"

sudo docker exec p12-clickhouse clickhouse-client --query \
  "SELECT run_id, step, s3_key, tag, is_protected, status
   FROM training_observability.checkpoints FINAL"
```

### From the training instance

```bash
# Vector running?
systemctl is-active p12-vector

# Vector API healthy?
curl -s http://localhost:8686/health
# Expected: {"ok":true}

# ClickHouse reachable over TLS?
source /etc/p12/vector.env
curl -sk --cacert /etc/p12/ca.crt \
  "${CLICKHOUSE_HTTPS_ENDPOINT}/?user=${CLICKHOUSE_USER}&password=${CLICKHOUSE_PASSWORD}&query=SELECT+1"
# Expected: 1

# JSONL files being written?
ls -la /tmp/training_logs/
```

---

## 7. Health Checks & Monitoring

Both instances push CloudWatch custom metrics via cron (every minute).

### DB Instance Metrics (namespace: `P12/ClickHouse`)

| Metric | Description | Alarm threshold |
|--------|-------------|-----------------|
| `Alive` | ClickHouse responds to `SELECT 1` | 0 for 3 consecutive minutes |
| `UptimeSeconds` | Server uptime | — (informational) |
| `DiskUsedPercent` | `/data/clickhouse` EBS usage | > 80% for 10 minutes |
| `LogsRowCount` | Total rows in `logs` table | — (informational) |
| `MetricPointsRowCount` | Total rows in `metric_points` | — (informational) |
| `LastInsertAgeSeconds` | Seconds since last insert into `logs` | > 600 during active training |

Script: `clickhouse/healthcheck/clickhouse-healthcheck.sh`

### Training Instance Metrics (namespace: `P12/Training`)

| Metric | Description | Alarm threshold |
|--------|-------------|-----------------|
| `VectorAlive` | Vector process running (`pgrep`) | 0 for 3 consecutive minutes |
| `VectorServiceActive` | systemd service active | 0 for 3 consecutive minutes |
| `ClickHouseReachable` | HTTPS query returns 200 | 0 for 3 consecutive minutes |
| `JsonlFreshnessSeconds` | Age of newest `.jsonl` file | > 300 during active training |
| `VectorApiHealthy` | Vector health API returns 200 | 0 for 3 consecutive minutes |

Script: embedded in `sidecar_agent/userdata-vector.sh` (step 7), installed at `/usr/local/bin/p12-training-healthcheck.sh`.

### Step 1: Create SNS Topic for Alarm Notifications

Two options — use one or both. All 6 CloudWatch alarms fire to whichever topic ARN(s) you configure.

**Option A: Create a new P12-specific SNS topic (email alerts)**

```bash
REGION="us-east-1"

# Create the topic
P12_TOPIC_ARN=$(aws sns create-topic --name p12-alerts --query 'TopicArn' --output text --region "$REGION")

# Subscribe (email — confirm in inbox)
aws sns subscribe \
  --topic-arn "$P12_TOPIC_ARN" \
  --protocol email \
  --notification-endpoint "your-team@company.com" \
  --region "$REGION"

echo "P12 SNS topic: $P12_TOPIC_ARN"
echo "Confirm the subscription in your email inbox before alarms will work."
```

**Option B: Reuse existing Telegram SNS topic**

If you already have an existing SNS topic (e.g. `T15-IdleCPUMonitor-410-Telegram-Alert-Topic`), look up its ARN:

```bash
TELEGRAM_TOPIC_ARN=$(aws sns list-topics --region "$REGION" \
  --query "Topics[?ends_with(TopicArn, ':T15-IdleCPUMonitor-410-Telegram-Alert-Topic')].TopicArn" \
  --output text)

echo "Existing Telegram topic: $TELEGRAM_TOPIC_ARN"
```

**Using both topics:** Pass multiple `--alarm-actions` to each alarm so both topics receive alerts:

```bash
--alarm-actions "$P12_TOPIC_ARN" "$TELEGRAM_TOPIC_ARN" \
--ok-actions "$P12_TOPIC_ARN" "$TELEGRAM_TOPIC_ARN" \
```

### Step 2: Create All 6 CloudWatch Alarms

Set the topic ARN(s) from Step 1:

```bash
REGION="us-east-1"
ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
TOPIC_ARN="arn:aws:sns:${REGION}:${ACCOUNT_ID}:p12-alerts"
# Uncomment to also use existing Telegram topic:
# TELEGRAM_TOPIC_ARN="arn:aws:sns:${REGION}:${ACCOUNT_ID}:T15-IdleCPUMonitor-410-Telegram-Alert-Topic"
```

**Alarm 1: ClickHouse Down** — fires if ClickHouse fails to respond for 3 consecutive minutes.

```bash
aws cloudwatch put-metric-alarm \
  --alarm-name "P12-ClickHouse-Down" \
  --alarm-description "ClickHouse is not responding to queries for 3+ minutes" \
  --namespace "P12/ClickHouse" \
  --metric-name "Alive" \
  --statistic Minimum \
  --period 60 \
  --evaluation-periods 3 \
  --threshold 1 \
  --comparison-operator LessThanThreshold \
  --treat-missing-data breaching \
  --alarm-actions "$TOPIC_ARN" \
  --ok-actions "$TOPIC_ARN" \
  --region "$REGION"
```

**Alarm 2: ClickHouse EBS Disk > 80%** — fires if EBS volume usage exceeds 80% for 10 minutes.

```bash
aws cloudwatch put-metric-alarm \
  --alarm-name "P12-ClickHouse-Disk-High" \
  --alarm-description "ClickHouse EBS data volume usage > 80% for 10+ minutes" \
  --namespace "P12/ClickHouse" \
  --metric-name "DiskUsedPercent" \
  --statistic Maximum \
  --period 300 \
  --evaluation-periods 2 \
  --threshold 80 \
  --comparison-operator GreaterThanThreshold \
  --alarm-actions "$TOPIC_ARN" \
  --region "$REGION"
```

**Alarm 3: Ingestion Stale** — fires if no new rows appear in the logs table for 10+ minutes during active training.

```bash
aws cloudwatch put-metric-alarm \
  --alarm-name "P12-Ingestion-Stale" \
  --alarm-description "No new data ingested into ClickHouse for >10 minutes" \
  --namespace "P12/ClickHouse" \
  --metric-name "LastInsertAgeSeconds" \
  --statistic Maximum \
  --period 60 \
  --evaluation-periods 5 \
  --threshold 600 \
  --comparison-operator GreaterThanThreshold \
  --treat-missing-data notBreaching \
  --alarm-actions "$TOPIC_ARN" \
  --region "$REGION"
```

**Alarm 4: Vector Sidecar Down** — fires if Vector process is not running on a training instance for 3+ minutes.

```bash
aws cloudwatch put-metric-alarm \
  --alarm-name "P12-Vector-Sidecar-Down" \
  --alarm-description "Vector sidecar not running on training instance for 3+ minutes" \
  --namespace "P12/Training" \
  --metric-name "VectorAlive" \
  --statistic Minimum \
  --period 60 \
  --evaluation-periods 3 \
  --threshold 1 \
  --comparison-operator LessThanThreshold \
  --treat-missing-data breaching \
  --alarm-actions "$TOPIC_ARN" \
  --ok-actions "$TOPIC_ARN" \
  --region "$REGION"
```

**Alarm 5: Training → ClickHouse Connectivity Lost** — fires if training instance can't reach ClickHouse for 3+ minutes.

```bash
aws cloudwatch put-metric-alarm \
  --alarm-name "P12-Training-ClickHouse-Unreachable" \
  --alarm-description "Training instance cannot reach ClickHouse for 3+ minutes" \
  --namespace "P12/Training" \
  --metric-name "ClickHouseReachable" \
  --statistic Minimum \
  --period 60 \
  --evaluation-periods 3 \
  --threshold 1 \
  --comparison-operator LessThanThreshold \
  --treat-missing-data breaching \
  --alarm-actions "$TOPIC_ARN" \
  --ok-actions "$TOPIC_ARN" \
  --region "$REGION"
```

**Alarm 6: JSONL Files Stale** — fires if no JSONL writes on training instance for 5+ minutes (training may have crashed).

```bash
aws cloudwatch put-metric-alarm \
  --alarm-name "P12-JSONL-Stale" \
  --alarm-description "No JSONL log writes on training instance for >5 minutes" \
  --namespace "P12/Training" \
  --metric-name "JsonlFreshnessSeconds" \
  --statistic Maximum \
  --period 60 \
  --evaluation-periods 5 \
  --threshold 300 \
  --comparison-operator GreaterThanThreshold \
  --treat-missing-data notBreaching \
  --alarm-actions "$TOPIC_ARN" \
  --region "$REGION"
```

### Step 3: Verify Alarms

```bash
# List all P12 alarms
aws cloudwatch describe-alarms \
  --alarm-name-prefix "P12-" \
  --query 'MetricAlarms[].{Name:AlarmName,State:StateValue,Metric:MetricName}' \
  --output table --region "$REGION"

# Force-test: push a bad metric to trigger an alarm
aws cloudwatch put-metric-data \
  --namespace "P12/ClickHouse" \
  --metric-name "Alive" \
  --value 0 \
  --dimensions "InstanceId=test-manual" \
  --region "$REGION"

# Wait ~5 min, then check alarm state
aws cloudwatch describe-alarms \
  --alarm-names "P12-ClickHouse-Down" \
  --query 'MetricAlarms[0].StateValue' \
  --region "$REGION"
```

### Alarm Summary

| # | Alarm | What it detects | Fires when | Runbook action |
|---|-------|----------------|-----------|----------------|
| 1 | P12-ClickHouse-Down | Server unresponsive | 3 min of SELECT 1 failures | SSM to DB, `sudo docker compose restart` |
| 2 | P12-ClickHouse-Disk-High | EBS filling up | >80% for 10 min | Resize EBS (see Operational Runbook) |
| 3 | P12-Ingestion-Stale | No new data in CH | >10 min gap in logs table | Check Vector, SG rules, certs |
| 4 | P12-Vector-Sidecar-Down | Vector process died | 3 min down | `systemctl restart p12-vector` |
| 5 | P12-Training-CH-Unreachable | Network/auth broken | 3 min unreachable | Check SG, certs, password, IP |
| 6 | P12-JSONL-Stale | Training not writing | >5 min no writes | Check training process, disk space |

---

## 8. Script Reference

### DB Instance Scripts

| Script | Location | When to run | Idempotent? |
|--------|----------|-------------|-------------|
| `setup-db-instance.sh` | `clickhouse/` | Once per fresh DB instance | Yes |
| `setup-auth.sh` | `clickhouse/` | Once per deployment (or when rotating creds) | Partially (overwrites existing certs/params) |
| `generate-ca.sh` | `clickhouse/tls/` | Once (CA should be long-lived) | Yes (skips if `ca.key` exists) |
| `generate-server-cert.sh` | `clickhouse/tls/` | Once per DB instance IP | No (always regenerates) |
| `apply_schema.sh` | `clickhouse/` | After schema changes on a running instance | Yes (uses `IF NOT EXISTS`) |
| `clickhouse-healthcheck.sh` | `clickhouse/healthcheck/` | Runs via cron every minute | Yes |

### Training Instance Scripts

| Script | Location | When to run | Idempotent? |
|--------|----------|-------------|-------------|
| `userdata-vector.sh` | `sidecar_agent/` | EC2 User Data (every boot) | Yes |
| `p12-training-healthcheck.sh` | Installed by userdata at `/usr/local/bin/` | Runs via cron every minute | Yes |

### Schema Files (applied automatically on first ClickHouse boot)

| File | Creates |
|------|---------|
| `initdb.d/001_database.sql` | `training_observability` database |
| `initdb.d/002_logs_table.sql` | `logs` table (MergeTree, partitioned by month) |
| `initdb.d/003_typed_tables.sql` | `runs`, `metric_points`, `metric_arrays`, `events`, `checkpoints` tables |

### Config Files

| File | Purpose |
|------|---------|
| `config.d/https.xml` | Enables HTTPS on port 8443, references TLS cert paths |
| `users.d/p12-users.xml.template` | Template for ClickHouse users with placeholder hashes and CIDRs |
| `sidecar_agent/vector.toml` | Vector pipeline: sources (file), transforms (parse, fan-out, checkpoint filter), sinks (ClickHouse) |
| `docker-compose.yml` | ClickHouse container config with EBS mounts, port mapping, healthcheck |

---

## 9. Security Model

### Network

- **HTTPS only** for cross-instance traffic (port 8443, TLS 1.2+).
- **HTTP (8123)** bound to `127.0.0.1` on the DB instance — admin/healthcheck access only.
- ClickHouse security group allows TCP 8443 inbound only from the training and dashboard subnet CIDRs.
- No SSH ports open — all operator access is via SSM Session Manager.

### Authentication

- **p12_writer**: INSERT + SELECT on `training_observability`. Restricted to training subnet CIDR.
- **p12_reader**: SELECT only on `training_observability`. Restricted to dashboard subnet CIDR.
- **default**: Localhost only (`127.0.0.1`, `::1`). Used by `docker exec` healthchecks.

### Credential Storage

| Secret | Storage | Access |
|--------|---------|--------|
| Writer password | SSM `/p12/training/clickhouse-password` (SecureString) | Training instance IAM role |
| Reader password | SSM `/p12/dashboard/clickhouse-password` (SecureString) | Dashboard instance IAM role |
| ClickHouse endpoint | SSM `/p12/training/clickhouse-endpoint` (String) | Training instance IAM role |
| CA private key | `tls/ca/ca.key` on DB instance only | Never leaves the DB instance |
| Server private key | `tls/server.key` on DB instance only | Mounted read-only into Docker |

### IAM Roles

| Role | Attached to | Permissions |
|------|-------------|-------------|
| `p12-clickhouse-db-role` | DB EC2 instance | SSM Session Manager, CloudWatch PutMetricData, S3 config bucket (create/upload) |
| `p12-training-instance-role` | Training EC2 instances | SSM Session Manager, SSM GetParameter (`/p12/training/*`), CloudWatch PutMetricData |

### Files that must NOT be committed to git

- `tls/ca/` (CA private key)
- `tls/server.crt`, `tls/server.key`
- `users.d/p12-users.xml` (contains password hashes)
- `training-instance.env`, `dashboard.env`
- Any `.env` files

---

## 10. Operational Runbook

### Rotate credentials

```bash
# On the DB instance:
cd ~/clickhouse

# Set new passwords
export P12_WRITER_PASSWORD="$(openssl rand -base64 18)"
export P12_READER_PASSWORD="$(openssl rand -base64 18)"
export TRAINING_SUBNET_CIDR=10.0.1.0/24
export DASHBOARD_SUBNET_CIDR=10.0.2.0/24
export DB_PRIVATE_IP=$(curl -s http://169.254.169.254/latest/meta-data/local-ipv4)

# Regenerate users XML + update SSM
bash setup-auth.sh

# Restart ClickHouse to pick up new user config
sudo docker compose restart

# Training instances: reboot to re-run userdata (pulls new password from SSM)
# Or manually: aws ssm get-parameter ... and update /etc/p12/vector.env
```

### Automated EBS snapshots (DLM)

Set up a Data Lifecycle Manager policy to take daily snapshots of the ClickHouse data volume. Only targets volumes tagged `Name=p12-clickhouse-data` (set during Phase 0, Step 5).

```bash
REGION="us-east-1"
ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)

aws dlm create-lifecycle-policy \
  --description "Daily snapshot of P12 ClickHouse data volume" \
  --state ENABLED \
  --execution-role-arn "arn:aws:iam::${ACCOUNT_ID}:role/AWSDataLifecycleManagerDefaultRole" \
  --policy-details '{
    "PolicyType": "EBS_SNAPSHOT_MANAGEMENT",
    "ResourceTypes": ["VOLUME"],
    "TargetTags": [{"Key": "Name", "Value": "p12-clickhouse-data"}],
    "Schedules": [{
      "Name": "DailySnapshot",
      "CreateRule": {"Interval": 24, "IntervalUnit": "HOURS", "Times": ["03:00"]},
      "RetainRule": {"Count": 7},
      "CopyTags": true
    }]
  }' --region "$REGION"
```

This retains 7 daily snapshots (1 week rolling window). Snapshots are taken at 03:00 UTC.

### Resize EBS volume (online, no downtime)

When the P12-ClickHouse-Disk-High alarm fires (>80% usage), resize the EBS volume online:

```bash
REGION="us-east-1"
VOLUME_ID="vol-xxxxxxxxxx"   # the p12-clickhouse-data volume

# Increase from 100GB to 200GB (no downtime)
aws ec2 modify-volume --volume-id "$VOLUME_ID" --size 200 --region "$REGION"

# Wait for modification to complete
aws ec2 describe-volumes-modifications \
  --volume-ids "$VOLUME_ID" \
  --query 'VolumesModifications[0].ModificationState' --region "$REGION"
# Wait until it shows "optimizing" or "completed"

# Resize the filesystem on the DB instance (also online, no downtime)
# Connect via SSM first:
aws ssm start-session --target "$DB_INSTANCE_ID" --region "$REGION"

# Then on the DB instance:
sudo resize2fs /dev/xvdf    # or the nvme device name
df -h /data/clickhouse
```

### Migrate to a new DB instance

```bash
# 1. On old instance: export data
sudo docker exec p12-clickhouse clickhouse-client \
  --query "SELECT * FROM training_observability.logs FORMAT Native" > logs.native
sudo docker exec p12-clickhouse clickhouse-client \
  --query "SELECT * FROM training_observability.metric_points FORMAT Native" > metric_points.native
sudo docker exec p12-clickhouse clickhouse-client \
  --query "SELECT * FROM training_observability.checkpoints FORMAT Native" > checkpoints.native

# 2. Deploy new instance (Phase 0 + Phase 1 above)

# 3. Import data on new instance
cat logs.native | sudo docker exec -i p12-clickhouse clickhouse-client \
  --query "INSERT INTO training_observability.logs FORMAT Native"
cat metric_points.native | sudo docker exec -i p12-clickhouse clickhouse-client \
  --query "INSERT INTO training_observability.metric_points FORMAT Native"
cat checkpoints.native | sudo docker exec -i p12-clickhouse clickhouse-client \
  --query "INSERT INTO training_observability.checkpoints FORMAT Native"

# 4. Update SSM endpoint parameter
aws ssm put-parameter \
  --name "/p12/training/clickhouse-endpoint" \
  --value "https://<NEW_IP>:8443" \
  --type String --overwrite --region us-east-1

# 5. Reboot training instances (re-runs userdata, picks up new endpoint)
```

### Apply schema changes to a running instance

```bash
# Edit the SQL files in initdb.d/, then:
bash apply_schema.sh

# Or run directly:
sudo docker exec p12-clickhouse clickhouse-client \
  --multiquery --queries-file /docker-entrypoint-initdb.d/003_typed_tables.sql
```

Note: `initdb.d/` SQL files use `CREATE TABLE IF NOT EXISTS`, so re-running them is safe. For schema migrations (ALTER TABLE), write new SQL files and apply them manually.

### Wipe and restart (dev/test only)

```bash
# On DB instance:
sudo docker compose down -v   # -v removes Docker volumes — ALL DATA IS LOST
sudo docker compose up -d     # fresh start, schema re-applied from initdb.d/
```

### Check logs

```bash
# DB instance — userdata/setup log
cat /var/log/p12-db-setup.log

# DB instance — healthcheck log
tail -f /var/log/p12-clickhouse-healthcheck.log

# DB instance — ClickHouse server logs
sudo docker logs p12-clickhouse --tail 100

# Training instance — userdata log
cat /var/log/p12-userdata.log

# Training instance — Vector logs
journalctl -u p12-vector -f

# Training instance — healthcheck log
tail -f /var/log/p12-training-healthcheck.log
```

### Deployment order checklist

```
Phase 0: AWS Infrastructure (from operator workstation)
[ ] 1. Create IAM role for DB instance (p12-clickhouse-db-role)
[ ] 2. Create IAM role for training instances (p12-training-instance-role)
[ ] 3. Create security group for ClickHouse (p12-clickhouse-sg)
[ ] 4. Launch DB EC2 instance (t3.medium)
[ ] 5. Create and attach EBS gp3 volume (100 GB)
[ ] 6. Connect via SSM to DB instance

Phase 1: DB Instance (on the DB instance via SSM)
[ ] 7. Run setup-db-instance.sh (EBS format, auth, certs, S3/SSM, docker compose)
[ ] 8. Verify: SHOW TABLES, curl ping, authenticated query
[ ] 9. Install DB healthcheck cron
[ ] 10. Note the S3 bucket name from setup-auth.sh output

Phase 2: Training Instance
[ ] 11. Update training EC2 launch template with userdata-vector.sh (set bucket name + IAM role)
[ ] 12. Launch training instance(s)
[ ] 13. Verify: Vector running, ClickHouse reachable, JSONL flowing
[ ] 14. Start training — ops.log_step() should produce rows in ClickHouse

Phase 3: Monitoring
[ ] 15. Create SNS topic (or reuse existing)
[ ] 16. Create all 6 CloudWatch alarms
[ ] 17. Set up automated EBS snapshots (DLM)
[ ] 18. Verify alarms (manual trigger test)
```
