# Post-ClickHouse Installation Setup

Scripts for configuring the ClickHouse DB environment after the initial install. Covers security group creation, EBS volume provisioning, credential storage in SSM Parameter Store, automated snapshots, and a healthcheck cron.

**Prerequisites:**
- AWS CLI installed and configured with appropriate permissions
- Access to the target VPC and subnet CIDRs
- The ClickHouse DB EC2 instance is already running

---

## 1. Create the ClickHouse Security Group

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
  --group-name p12-clickhouse-sg \
  --description "P12 ClickHouse DB - restricted access" \
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

Provisions a **gp3** EBS volume and attaches it to the ClickHouse DB instance. The volume is tagged `Name=p12-clickhouse-data`, which the DLM snapshot policy in [Section 4](#4-set-up-automated-ebs-snapshots-dlm) uses to identify it.

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
        "ec2:Waiter",
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
  --tag-specifications "ResourceType=volume,Tags=[{Key=Name,Value=p12-clickhouse-data},{Key=Project,Value=p12},{Key=Team,Value=${TAG_TEAM}},{Key=TaskId,Value=${TAG_TASK_ID}},{Key=WorkloadType,Value=${TAG_WORKLOAD_TYPE}}]" \
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

## 3. Store Credentials in SSM Parameter Store

Writes the ClickHouse writer password, reader password, and HTTPS endpoint into AWS Systems Manager Parameter Store. Passwords are stored as `SecureString` (encrypted with the default KMS key).

**IAM policy required:**

```json
{
  "Version": "2012-10-17",
  "Statement": [
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
        "arn:aws:ssm:*:*:parameter/T12-TrainingOperations-239/*",
        "arn:aws:ssm:*:*:session/*"
      ]
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

P12_WRITER_PASSWORD="password"
P12_READER_PASSWORD="password"

DB_PUBLIC_IP=54.174.194.76

SKIP_CREDENTIALS=false
if [ "$SKIP_CREDENTIALS" = "false" ]; then
  aws ssm put-parameter \
    --name "/$PREFIX/clickhouse/writer-password" \
    --value "$P12_WRITER_PASSWORD" \
    --type SecureString \
    --overwrite \
    --region "$AWS_REGION" >/dev/null

  aws ssm put-parameter \
    --name "/$PREFIX/clickhouse/reader-password" \
    --value "$P12_READER_PASSWORD" \
    --type SecureString \
    --overwrite \
    --region "$AWS_REGION" >/dev/null

  aws ssm put-parameter --region "$AWS_REGION" --cli-input-json "{
    \"Name\": \"/$PREFIX/clickhouse/endpoint\",
    \"Value\": \"https://${DB_PUBLIC_IP}:8443\",
    \"Type\": \"String\",
    \"Overwrite\": true
  }" >/dev/null

  echo "✓ Credentials stored in SSM Parameter Store"
fi
```

### Standalone commands for the Vector sidecar parameters

The Vector sidecar on training instances (Account A) reads two of these parameters via cross-account assume-role. If you need to write or update them individually, run these in the **SSM/infra account (Account B)**:

**Writer password** (SecureString — encrypted at rest with default KMS key):

```bash
aws ssm put-parameter \
  --name "/T12-TrainingOperations-239/clickhouse/writer-password" \
  --value "YOUR_WRITER_PASSWORD" \
  --type SecureString \
  --overwrite \
  --region us-east-1
```

**ClickHouse endpoint** (String — the HTTPS URL including port):

```bash
aws configure set cli_follow_urlparam false
aws ssm put-parameter \
  --name "/T12-TrainingOperations-239/clickhouse/endpoint" \
  --value "https://CLICKHOUSE_IP:8443" \
  --type String \
  --overwrite \
  --region us-east-1
```

> **Cross-account access:** Training instances assume the `t12-ssm-reader` role in Account B to read these parameters. See `sidecar_agent/ssm-reader-cross-account-role.json` for the role definition and `sidecar_agent/userdata_vector.sh` step [5/9] for the assume-role flow.

---

## 4. Set Up Automated EBS Snapshots (DLM)

Creates a Data Lifecycle Manager policy that takes **daily snapshots** of the ClickHouse data volume and retains the last **7 snapshots**. The policy targets volumes tagged `Name=p12-clickhouse-data` (applied in [Section 2](#2-create-and-attach-an-ebs-data-volume)).

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

## 5. Install the Healthcheck Cron

Installs a cron job that runs the ClickHouse healthcheck script every minute. The script pushes 6 CloudWatch metrics per run (see [Health Checks](#7-health-checks--monitoring)).

```bash
sudo cp healthcheck/clickhouse-healthcheck.sh /usr/local/bin/
sudo chmod +x /usr/local/bin/clickhouse-healthcheck.sh

echo "* * * * * root /usr/local/bin/clickhouse-healthcheck.sh >> /var/log/p12-clickhouse-healthcheck.log 2>&1" \
  | sudo tee /etc/cron.d/p12-clickhouse-healthcheck
```
