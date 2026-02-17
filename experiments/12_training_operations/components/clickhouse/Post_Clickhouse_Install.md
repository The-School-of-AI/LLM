# Contains scripts to set up the ClickHouse DB including security group and EBS volume creation, and storing credentials in SSM Parameter Store.

## Script for setting up ClickHouse DB security group

### Note: This script assumes you have AWS CLI configured with appropriate permissions and that you have the necessary environment variables set for the training and dashboard subnet CIDRs.

##### TO-DO: Need IAM permissions needed: ec2:CreateSecurityGroup, ec2:AuthorizeSecurityGroupIngress, ec2:DescribeVpcs

```
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
```

```bash
VPC_ID="vpc-067afb94fe77053c4"
TRAINING_SUBNET_CIDR=1.2.3.4/32 # REPLACE with your training subnet CIDR
# Default values
PREFIX="T12-TrainingOperations-239" # REPLACE with your unique prefix for resource naming
AWS_REGION="${AWS_REGION:-us-east-1}"

# Tags
TAG_TEAM="Team12"
TAG_TASK_ID="Issue440"
TAG_WORKLOAD_TYPE="TrainingOperations"

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
  #--tag-specifications "ResourceType=security-group-rule,Tags=[{Key=Name,Value=p12-training-to-clickhouse}]"
  --tags Team=${TAG_TEAM},TaskId=${TAG_TASK_ID},WorkloadType=${TAG_WORKLOAD_TYPE}

# Rule 2: ClickHouse HTTPS (8443) from dashboard subnet only
aws ec2 authorize-security-group-ingress \
  --group-id "$DB_SG_ID" \
  --protocol tcp --port 8443 \
  --cidr "$DASHBOARD_SUBNET_CIDR" \
  #--tag-specifications "ResourceType=security-group-rule,Tags=[{Key=Name,Value=p12-dashboard-to-clickhouse}]"
  --tags Team=${TAG_TEAM},TaskId=${TAG_TASK_ID},WorkloadType=${TAG_WORKLOAD_TYPE}
```


## Script for setting up Creating and attaching EBS volume to ClickHouse DB instance

#### TO-DO: Need IAM permissions needed: ec2:CreateVolume, ec2:DescribeInstances, ec2:AttachVolume, ec2:Waiter
```
   {
      "Sid": "EC2InstanceAndEBS",
      "Effect": "Allow",
      "Action": [
        "ec2:DescribeInstances",
        "ec2:CreateVolume",
        "ec2:AttachVolume",
        "ec2:Waiter"
      ],
      "Resource": "*"
    },

```


```bash
# Default values
PREFIX="T12-TrainingOperations-239" # REPLACE with your unique prefix for resource naming
AWS_REGION="${AWS_REGION:-us-east-1}"

# Tags
TAG_TEAM="Team12"
TAG_TASK_ID="Issue440"
TAG_WORKLOAD_TYPE="TrainingOperations"

DB_INSTANCE_ID="i-0b1c2d3e4f5g6h7i8" # REPLACE with your DB instance ID
# Get the AZ of the DB instance (EBS must be in the same AZ)
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
  #--tag-specifications "ResourceType=volume,Tags=[{Key=Name,Value=p12-clickhouse-data},{Key=Project,Value=p12}]" \
  --tags Team=${TAG_TEAM},TaskId=${TAG_TASK_ID},WorkloadType=${TAG_WORKLOAD_TYPE} \
  --query 'VolumeId' --output text \
  --region "$AWS_REGION")

echo "Created volume: $VOLUME_ID"

# Wait for it to become available
aws ec2 wait volume-available --volume-ids "$VOLUME_ID" --region "$AWS_REGION"

# Attach to the DB instance
aws ec2 attach-volume \
  --volume-id "$VOLUME_ID" \
  --instance-id "$DB_INSTANCE_ID" \
  --device /dev/xvdf \
  --region "$AWS_REGION"

aws ec2 wait volume-in-use --volume-ids "$VOLUME_ID" --region "$AWS_REGION"
echo "Volume $VOLUME_ID attached to $DB_INSTANCE_ID as /dev/xvdf"
```


## Script for writing parameters to AWS SSM
#### TO-DO Need IAM permissions for ssm:PutParameter and ssm:GetParameter to run this script
```
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
```


``` bash
PREFIX="T12-TrainingOperations-239" # REPLACE with your unique prefix for resource naming
AWS_REGION="${AWS_REGION:-us-east-1}"

# Tags
TAG_TEAM="Team12"
TAG_TASK_ID="Issue440"
TAG_WORKLOAD_TYPE="TrainingOperations"

P12_WRITER_PASSWORD="trtraining_ops_writer_pass"
P12_READER_PASSWORD="trtraining_ops_reader_pass"

DB_PUBLIC_IP=54.174.194.76
# ---- 6. Store credentials in SSM Parameter Store ----
SKIP_CREDENTIALS=false
if [ "$SKIP_CREDENTIALS" = "false" ]; then
  aws ssm put-parameter \
    --name "/$PREFIX/clickhouse/writer-password" \
    --value "$P12_WRITER_PASSWORD" \
    --type SecureString \
    --overwrite \
    --region "$AWS_REGION" >/dev/null

  aws ssm put-parameter \
    --name "/$PREFIX$/clickhouse/reader-password" \
    --value "$P12_READER_PASSWORD" \
    --type SecureString \
    --overwrite \
    --region "$P12_REGION" >/dev/null

  aws ssm put-parameter --region "$AWS_REGION" --cli-input-json "{
    \"Name\": \"/$PREFIX/clickhouse/endpoint\",
    \"Value\": \"https://${DB_PUBLIC_IP}:8443\",
    \"Type\": \"String\",
    \"Overwrite\": true
  }" >/dev/null

  echo "✓ Credentials stored in SSM Parameter Store"
fi

```

## Automated EBS snapshots (DLM)
####  IAM permissions needed: dlm:CreateLifecyclePolicy, iam:PassRole (for the DLM execution role)

```
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
```

Set up a Data Lifecycle Manager policy to take daily snapshots of the ClickHouse data volume. Only targets volumes tagged `Name=p12-clickhouse-data` (set during Phase 0, Step 5).

```bash
AWS_REGION="${AWS_REGION:-us-east-1}"

# Tags
TAG_TEAM="Team12"
TAG_TASK_ID="Issue440"
TAG_WORKLOAD_TYPE="TrainingOperations"

ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)

aws dlm create-lifecycle-policy \
  --description "Daily snapshot of P12 ClickHouse data volume" \
  --state ENABLED \
  --execution-role-arn "arn:aws:iam::${ACCOUNT_ID}:role/AWSDataLifecycleManagerDefaultRole" \
  --policy-details '{
    "PolicyType": "EBS_SNAPSHOT_MANAGEMENT",
    "ResourceTypes": ["VOLUME"],
    --tags Team=${TAG_TEAM},TaskId=${TAG_TASK_ID},WorkloadType=${TAG_WORKLOAD_TYPE} \
    "Schedules": [{
      "Name": "DailySnapshot",
      "CreateRule": {"Interval": 24, "IntervalUnit": "HOURS", "Times": ["03:00"]},
      "RetainRule": {"Count": 7},
      "CopyTags": true
    }]
  }' --region "$REGION"

```

If the need is to increase the EBS volume size
```
aws ec2 modify-volume --volume-id "$VOLUME_ID" --size 200 --region "$AWS_REGION"

# Wait for modification to complete
aws ec2 describe-volumes-modifications \
  --volume-ids "$VOLUME_ID" \
  --query 'VolumesModifications[0].ModificationState' --region "$REGION"
# Wait until it shows "optimizing" or "completed"
```

## Step 4: Install healthcheck cron

```bash
sudo cp healthcheck/clickhouse-healthcheck.sh /usr/local/bin/
sudo chmod +x /usr/local/bin/clickhouse-healthcheck.sh

echo "* * * * * root /usr/local/bin/clickhouse-healthcheck.sh >> /var/log/p12-clickhouse-healthcheck.log 2>&1" \
  | sudo tee /etc/cron.d/p12-clickhouse-healthcheck
```

This pushes 6 CloudWatch metrics every minute (see [Health Checks](#7-health-checks--monitoring)).
