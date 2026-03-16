# Post-ClickHouse Installation Setup

Scripts for configuring the ClickHouse DB environment after the initial install. Covers security group creation, EBS volume provisioning, credential storage in Secrets Manager, cross-account IAM for the Vector sidecar, automated snapshots, and a healthcheck cron.

**Prerequisites:**
- AWS CLI installed and configured with appropriate permissions
- Access to the target VPC and subnet CIDRs
- The ClickHouse DB EC2 instance is already running

---

## Setup Checklist

Run these steps in order. Steps 1–4 are required for the Vector sidecar to function; steps 5–6 are operational extras.

| # | Step | Run in | Required? |
|---|------|--------|-----------|
| 1 | [Create the ClickHouse Security Group](#1-create-the-clickhouse-security-group) | Account B (infra) | **Required** — unless 8443 access is already configured |
| 2 | [Create and Attach an EBS Data Volume](#2-create-and-attach-an-ebs-data-volume) | Account B (infra) | **Recommended** — skip only if ClickHouse data already lives on a separate volume |
| 3 | [Store Credentials in Secrets Manager](#3-store-credentials-in-secrets-manager) | Account B (infra) | **Required** — `userdata_vector.sh` will fail without this secret |
| 4 | [Set Up Cross-Account IAM for the Vector Sidecar](#4-set-up-cross-account-iam-for-the-vector-sidecar) | Accounts A and B | **Required** — training instances need this role to read the secret |
| 5 | [Set Up Automated EBS Snapshots (DLM)](#5-set-up-automated-ebs-snapshots-dlm) | Account B (infra) | **Optional** — recommended for production; not needed for core functionality |
| 6 | [Install the Healthcheck Cron](#6-install-the-healthcheck-cron) | ClickHouse instance | **Optional** — adds CloudWatch visibility into ClickHouse availability |

---

## 1. Create the ClickHouse Security Group

> **Required** — Without this, nothing can reach ClickHouse on port 8443. Skip if the instance already has a security group that permits TCP 8443 from the training and dashboard subnets.

Creates a dedicated security group for the ClickHouse instance and opens port **8443** (ClickHouse HTTPS) to the training and dashboard subnets only.

**IAM policy required** — provide this to your AWS admin:

```json
{
  "Version": "2012-10-17",
  "Statement": [
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
    }
  ]
}
```

**Script:**

```bash
VPC_ID="vpc-067afb94fe77053c4"
TRAINING_SUBNET_CIDR=1.2.3.4/32 # REPLACE with your training subnet CIDR
DASHBOARD_SUBNET_CIDR=5.6.7.8/32 # REPLACE with your dashboard subnet CIDR
PREFIX="T12-TrainingOperations-239" # REPLACE with your unique prefix for resource naming
AWS_REGION="${AWS_REGION:-us-east-1}"

# Tags
TAG_TEAM="Team12"
TAG_TASK_ID="Issue239"
TAG_WORKLOAD_TYPE="TrainingOperations"

# Create the security group
DB_SG_ID=$(aws ec2 create-security-group \
  --group-name t12-clickhouse-sg \
  --description "T12 ClickHouse DB - restricted access" \
  --vpc-id "$VPC_ID" \
  --query 'GroupId' --output text)

echo "Created security group: $DB_SG_ID"

# Rule 1: Allow ClickHouse HTTPS (8443) from the training subnet
aws ec2 authorize-security-group-ingress \
  --group-id "$DB_SG_ID" \
  --protocol tcp --port 8443 \
  --cidr "$TRAINING_SUBNET_CIDR" \
  --tag-specifications "ResourceType=security-group-rule,Tags=[{Key=Team,Value=${TAG_TEAM}},{Key=TaskId,Value=${TAG_TASK_ID}},{Key=WorkloadType,Value=${TAG_WORKLOAD_TYPE}}]"

# Rule 2: Allow ClickHouse HTTPS (8443) from the dashboard subnet
aws ec2 authorize-security-group-ingress \
  --group-id "$DB_SG_ID" \
  --protocol tcp --port 8443 \
  --cidr "$DASHBOARD_SUBNET_CIDR" \
  --tag-specifications "ResourceType=security-group-rule,Tags=[{Key=Team,Value=${TAG_TEAM}},{Key=TaskId,Value=${TAG_TASK_ID}},{Key=WorkloadType,Value=${TAG_WORKLOAD_TYPE}}]"
```

---

## 2. Create and Attach an EBS Data Volume

> **Recommended** — Keeps ClickHouse data on a volume independent of the root disk, so the instance can be replaced without data loss. Skip only if ClickHouse is already configured to write to a separately managed volume. The DLM snapshot policy in [Section 5](#5-set-up-automated-ebs-snapshots-dlm) targets this volume by tag.

Provisions a **gp3** EBS volume and attaches it to the ClickHouse DB instance. The volume is tagged `Name=t12-clickhouse-data`.

> The EBS volume must be in the same Availability Zone as the instance. The script looks up the instance's AZ automatically.

**IAM policy required:**

```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Sid": "EC2InstanceAndEBS",
      "Effect": "Allow",
      "Action": [
        "ec2:DescribeInstances",
        "ec2:DescribeVolumes",
        "ec2:CreateVolume",
        "ec2:AttachVolume",
        "ec2:CreateTags"
      ],
      "Resource": "*"
    }
  ]
}
```

**Script:**

```bash
PREFIX="T12-TrainingOperations-239" # REPLACE with your unique prefix for resource naming
AWS_REGION="${AWS_REGION:-us-east-1}"

# Tags
TAG_TEAM="Team12"
TAG_TASK_ID="Issue239"
TAG_WORKLOAD_TYPE="TrainingOperations"

DB_INSTANCE_ID="i-0b1c2d3e4f5g6h7i8" # REPLACE with your DB instance ID

# Look up the instance's Availability Zone (EBS must be in the same AZ)
AZ=$(aws ec2 describe-instances \
  --instance-ids "$DB_INSTANCE_ID" \
  --query 'Reservations[0].Instances[0].Placement.AvailabilityZone' \
  --output text --region "$AWS_REGION")

VOLUME_SIZE_GB=100

# Create the gp3 volume
VOLUME_ID=$(aws ec2 create-volume \
  --volume-type gp3 \
  --size "$VOLUME_SIZE_GB" \
  --iops 3000 \
  --throughput 125 \
  --availability-zone "$AZ" \
  --tag-specifications "ResourceType=volume,Tags=[{Key=Name,Value=t12-clickhouse-data},{Key=Project,Value=p12},{Key=Team,Value=${TAG_TEAM}},{Key=TaskId,Value=${TAG_TASK_ID}},{Key=WorkloadType,Value=${TAG_WORKLOAD_TYPE}}]" \
  --query 'VolumeId' --output text \
  --region "$AWS_REGION")

echo "Created volume: $VOLUME_ID"

# Wait for the volume to become available, then attach it
aws ec2 wait volume-available --volume-ids "$VOLUME_ID" --region "$AWS_REGION"

aws ec2 attach-volume \
  --volume-id "$VOLUME_ID" \
  --instance-id "$DB_INSTANCE_ID" \
  --device /dev/xvdf \
  --region "$AWS_REGION"

aws ec2 wait volume-in-use --volume-ids "$VOLUME_ID" --region "$AWS_REGION"
echo "Volume $VOLUME_ID attached to $DB_INSTANCE_ID as /dev/xvdf"
```

---

## 3. Store Credentials in Secrets Manager

> **Required** — `userdata_vector.sh` reads this secret at boot to configure the Vector sidecar. The bootstrap will fail at step 6 if the secret does not exist. Training instances access it via cross-account assume-role (see [Section 4](#4-set-up-cross-account-iam-for-the-vector-sidecar)).

Writes the ClickHouse writer password and HTTPS endpoint into AWS Secrets Manager as a single JSON secret. Run in the **infra account (Account B)**. The secret is encrypted with the default KMS key.

**IAM policy required:**

```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Sid": "SecretsManagerClickHouse",
      "Effect": "Allow",
      "Action": [
        "secretsmanager:CreateSecret",
        "secretsmanager:PutSecretValue",
        "secretsmanager:GetSecretValue",
        "secretsmanager:DescribeSecret"
      ],
      "Resource": "arn:aws:secretsmanager:us-east-1:*:secret:t12/clickhouse*"
    }
  ]
}
```

**Script:**

```bash
AWS_REGION="${AWS_REGION:-us-east-1}"

# Tags
TAG_TEAM="Team12"
TAG_TASK_ID="Issue239"
TAG_WORKLOAD_TYPE="TrainingOperations"

P12_WRITER_PASSWORD="password"

DB_PUBLIC_IP=54.174.194.76

aws secretsmanager create-secret \
  --name "t12/clickhouse" \
  --description "ClickHouse credentials for T12 Vector sidecar" \
  --secret-string "{\"writer-password\":\"${P12_WRITER_PASSWORD}\",\"endpoint\":\"https://${DB_PUBLIC_IP}:8443\"}" \
  --tags "[{\"Key\":\"Team\",\"Value\":\"${TAG_TEAM}\"},{\"Key\":\"TaskId\",\"Value\":\"${TAG_TASK_ID}\"},{\"Key\":\"WorkloadType\",\"Value\":\"${TAG_WORKLOAD_TYPE}\"}]" \
  --region "$AWS_REGION"

echo "✓ Credentials stored in Secrets Manager"
```

To update the secret later (password or endpoint changed):

```bash
aws secretsmanager put-secret-value \
  --secret-id "t12/clickhouse" \
  --secret-string "{\"writer-password\":\"${P12_WRITER_PASSWORD}\",\"endpoint\":\"https://${DB_PUBLIC_IP}:8443\"}" \
  --region "$AWS_REGION"
```

---

## 4. Set Up Cross-Account IAM for the Vector Sidecar

> **Required** — Training instances must assume a role in Account B to read the `t12/clickhouse` secret. Without this, the Vector sidecar bootstrap fails with an access-denied error at step 5.

The Vector sidecar (`userdata_vector.sh`) runs on training instances in **Account A** and reads the `t12/clickhouse` secret from **Account B** (this infra account) via cross-account assume-role. Two setup scripts handle this — one per account.

```
Account A (training)                    Account B (infra)
──────────────────────────────────      ──────────────────────────────────────
EC2 instance profile role               t12-secrets-reader role
  └─ sts:AssumeRole ──────────────────▶   └─ secretsmanager:GetSecretValue
  └─ cloudwatch:PutMetricData                └─ kms:Decrypt (via Secrets Manager)
```

### Step 1 — Run in Account A (training): create the instance role

Creates the EC2 instance role and instance profile that training instances use. **Run this first** — the infra script needs the training account ID (though not the role itself, with the current trust policy design).

The script is idempotent — if the role already exists, it only updates the inline policy.

```bash
bash sidecar_agent/setup-training-account.sh \
  --infra-account-id <INFRA_AWS_ACCOUNT_ID> \
  --role-name        t12-traininginstance-239-role \
  --region us-east-1
```

**IAM permissions required in Account A to run this script:**

```json
{
  "Version": "2012-10-17",
  "Statement": [{
    "Effect": "Allow",
    "Action": [
      "iam:CreateRole", "iam:GetRole", "iam:PutRolePolicy",
      "iam:CreateInstanceProfile", "iam:GetInstanceProfile", "iam:AddRoleToInstanceProfile"
    ],
    "Resource": "*"
  }]
}
```

### Step 2 — Run in Account B (infra): create the reader role

Creates the `t12-secrets-reader` role that training instances will assume to read the secret.

```bash
bash sidecar_agent/setup-infra-account.sh \
  --training-account-id <TRAINING_AWS_ACCOUNT_ID> \
  --training-role-name  t12-traininginstance-239-role \
  --region us-east-1
```

**IAM permissions required in Account B to run this script:**

```json
{
  "Version": "2012-10-17",
  "Statement": [{
    "Effect": "Allow",
    "Action": ["iam:CreateRole", "iam:PutRolePolicy", "iam:GetRole"],
    "Resource": "arn:aws:iam::*:role/t12-secrets-reader"
  }]
}
```

### Step 3 — Attach the instance profile to training EC2 instances

Every training instance that runs `userdata_vector.sh` must have this instance profile attached.

```bash
# At launch time:
aws ec2 run-instances \
  --iam-instance-profile Name=t12-traininginstance-239-role \
  ...

# Or on a running instance:
aws ec2 associate-iam-instance-profile \
  --instance-id <INSTANCE_ID> \
  --iam-instance-profile Name=t12-traininginstance-239-role
```

### Verify cross-account access (optional smoke test)

**⚠️ Important:** This test must be run from a training EC2 instance with the `t12-traininginstance-239-role` instance profile attached. The `t12-secrets-reader` role's trust policy only allows the training instance role to assume it, not individual IAM users.

**Steps:**
1. Launch or connect to a training instance with the instance profile attached
2. Connect via SSM: `aws ssm start-session --target <instance-id>`
3. Run the test commands below

```bash
# Assume the infra role (this works because the instance has t12-traininginstance-239-role)
CREDS=$(aws sts assume-role \
  --role-arn arn:aws:iam::205991465724:role/t12-secrets-reader \
  --role-session-name test-session --output json)

export AWS_ACCESS_KEY_ID=$(echo "$CREDS" | jq -r '.Credentials.AccessKeyId')
export AWS_SECRET_ACCESS_KEY=$(echo "$CREDS" | jq -r '.Credentials.SecretAccessKey')
export AWS_SESSION_TOKEN=$(echo "$CREDS" | jq -r '.Credentials.SessionToken')

# Read the secret
aws secretsmanager get-secret-value --secret-id t12/clickhouse --region us-east-1

unset AWS_ACCESS_KEY_ID AWS_SECRET_ACCESS_KEY AWS_SESSION_TOKEN
```

**Note:** Testing from your local workstation with IAM user credentials will fail with `AccessDenied` because the trust policy is restricted to the training instance role only.

---

## 5. Set Up Automated EBS Snapshots (DLM)

> **Optional** — Provides automatic daily backups of the ClickHouse data volume. Recommended for any environment where data loss would be disruptive. Requires [Section 2](#2-create-and-attach-an-ebs-data-volume) to have been completed (the DLM policy targets the `Name=p12-clickhouse-data` tag set there). The `AWSDataLifecycleManagerDefaultRole` is created automatically by AWS the first time you create a DLM policy via the console; if running this script in a fresh account, create that role first via the console or create it manually.

Creates a Data Lifecycle Manager policy that takes **daily snapshots** of the ClickHouse data volume and retains the last **7 snapshots**. Run in Account B.

**IAM policy required:**

```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Sid": "DLMForSnapshots",
      "Effect": "Allow",
      "Action": [
        "dlm:CreateLifecyclePolicy",
        "dlm:GetLifecyclePolicy",
        "dlm:UpdateLifecyclePolicy",
        "iam:PassRole"
      ],
      "Resource": "*"
    }
  ]
}
```

**Script:**

```bash
AWS_REGION="${AWS_REGION:-us-east-1}"

# Tags
TAG_TEAM="Team12"
TAG_TASK_ID="Issue239"
TAG_WORKLOAD_TYPE="TrainingOperations"

ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)

aws dlm create-lifecycle-policy \
  --description "Daily snapshot of T12 ClickHouse data volume" \
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
      "TagsToAdd": [
        {"Key": "Team", "Value": "'"${TAG_TEAM}"'"},
        {"Key": "TaskId", "Value": "'"${TAG_TASK_ID}"'"},
        {"Key": "WorkloadType", "Value": "'"${TAG_WORKLOAD_TYPE}"'"}
      ],
      "CopyTags": true
    }]
  }' --region "$AWS_REGION"
```

### Expanding the EBS volume later

If you need to increase the data volume size after initial setup:

```bash
aws ec2 modify-volume --volume-id "$VOLUME_ID" --size 200 --region "$AWS_REGION"

# Poll until the modification state shows "optimizing" or "completed"
aws ec2 describe-volumes-modifications \
  --volume-ids "$VOLUME_ID" \
  --query 'VolumesModifications[0].ModificationState' --region "$AWS_REGION"
```

---

## 6. Install the Healthcheck Cron

> **Optional** — Adds a CloudWatch monitoring heartbeat for the ClickHouse DB instance itself (separate from the Vector sidecar healthcheck embedded in `userdata_vector.sh`). Run directly on the ClickHouse EC2 instance. Useful for alerting on DB availability; not required for the Vector sidecar to function.

Installs a cron job that runs the ClickHouse healthcheck script every minute and pushes metrics to CloudWatch under the `T12/ClickHouse` namespace.

```bash
sudo cp healthcheck/clickhouse-healthcheck.sh /usr/local/bin/
sudo chmod +x /usr/local/bin/clickhouse-healthcheck.sh

echo "* * * * * root /usr/local/bin/clickhouse-healthcheck.sh >> /var/log/t12-clickhouse-healthcheck.log 2>&1" \
  | sudo tee /etc/cron.d/p12-clickhouse-healthcheck
```
