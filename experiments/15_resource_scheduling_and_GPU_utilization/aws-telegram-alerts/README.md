# AWS CloudWatch CPU Idle Alerts to Telegram

Monitors EC2 idle CPU and sends alerts to Telegram. Automatically creates alarms for new EC2 instances via EventBridge.

## Architecture

```
                                    ┌─────────────────────────────────────────┐
                                    │           On EC2 Launch                 │
EC2 Instance Starts ──► EventBridge ──► Lambda ──► Creates CloudWatch Alarm   │
                                    │              + Notifies Telegram        │
                                    └─────────────────────────────────────────┘

                                    ┌─────────────────────────────────────────┐
                                    │           On CPU Idle                   │
CloudWatch Alarm ──► SNS ──► Lambda ──► Telegram Alert                        │
       │                            │                                         │
       └──► EC2 Stop Action         │                                         │
                                    └─────────────────────────────────────────┘
```

## What Gets Created

| Resource | Name | Purpose |
|----------|------|---------|
| SSM Parameter | `/{PREFIX}/telegram-bot-token` | Encrypted Telegram token (SecureString) |
| IAM Role | `{PREFIX}-Telegram-alert-lambda-execution-role` | Lambda execution |
| Lambda | `{PREFIX}-Telegram-alert-forwarder` | Forwards alarms to Telegram |
| Lambda | `{PREFIX}-ec2-launch-alarm-creator` | Auto-creates alarms on EC2 launch |
| SNS Topic | `{PREFIX}-Telegram-alert-topic` | Bridges CloudWatch to Lambda |
| EventBridge Rule | `{PREFIX}-ec2-launch-cpu-alarm-rule` | Triggers on EC2 start |
| CloudWatch Alarms | `{instance-name}-cpu-idle` | One per EC2 instance |

## Security

The Telegram bot token is stored as an SSM Parameter Store `SecureString` at deploy time. It is **never** written to Lambda environment variables in plaintext.

- `setup.sh` writes the token to `/{PREFIX}/telegram-bot-token` using `ssm:PutParameter`
- Both Lambda functions fetch the token at runtime using `ssm:GetParameter` with `WithDecryption=True`
- The Lambda execution role is granted `ssm:GetParameter` on that parameter path only, via an inline IAM policy
- The token is cached in memory for the lifetime of the Lambda container to avoid repeated SSM calls

To rotate the token: run `setup.sh` again with the new token — it will overwrite the SSM parameter (`--overwrite`) and update the Lambda code. No redeployment of the Lambda zip is needed.

## Files

```
setup.sh          # Deploy to single account
deploy-all.sh     # Deploy to multiple accounts
teardown.sh       # Remove all resources (supports --all)
accounts.txt      # List of AWS profiles
iam-policy.json   # IAM policy for setup user
```

## Telegram Setup

1. Message `@BotFather` → `/newbot` → Save the **token**
2. Add bot to your group
3. Get chat ID:
   ```bash
   curl -s "https://api.telegram.org/bot<TOKEN>/getUpdates" | grep -o '"id":-[0-9]*' | head -1
   ```

## IAM Setup (Prerequisite)

Attach `iam-policy.json` to IAM users/roles that will run the setup scripts.

The managed policy `T15-IdleCPUMonitor-410` must exist in each account before running `setup.sh`. It is attached to the Lambda execution role and grants the permissions the Lambdas need at runtime. Create it with:

```bash
aws iam create-policy \
  --policy-name "T15-IdleCPUMonitor-410" \
  --policy-document '{
    "Version": "2012-10-17",
    "Statement": [
      {
        "Effect": "Allow",
        "Action": [
          "cloudwatch:PutMetricAlarm",
          "cloudwatch:DescribeAlarms",
          "cloudwatch:TagResource"
        ],
        "Resource": "*"
      },
      {
        "Effect": "Allow",
        "Action": [
          "ec2:DescribeInstances",
          "ec2:DescribeTags"
        ],
        "Resource": "*"
      }
    ]
  }'
```

> The `ssm:GetParameter` permission for the Lambda role is handled automatically by `setup.sh` via an inline policy — it does **not** need to be part of the managed policy above.
## Single Account

```bash
chmod +x setup.sh

./setup.sh \
  --telegram-token 123456789:ABCdef... \
  --telegram-chat-id -1001234567890 \
  --region us-east-1
```

After running once, new EC2 instances automatically get alarms via EventBridge.

## Multiple Accounts

1. Edit `accounts.txt` with AWS profile names (one per line):
   ```
   production
   staging
   development
   ```

2. Run:
   ```bash
   chmod +x deploy-all.sh

   export TELEGRAM_BOT_TOKEN=123456789:ABCdef...
   export TELEGRAM_CHAT_ID=-1001234567890
   export AWS_REGION=ap-south-1

   ./deploy-all.sh
   ```

## Options

| Flag | Default | Description |
|------|---------|-------------|
| `--telegram-token` | required | Bot API token |
| `--telegram-chat-id` | required | Group chat ID |
| `--region` | us-east-1 | AWS region |
| `--cpu-threshold` | 10 | Alert when CPU < this % |

## Alert Messages

**When alarm triggers (CPU idle):**
```
🚨 CPU Idle Alert

Account: 123456789012
Alarm: my-server-cpu-idle
Status: ALARM
Region: us-east-1

Metric: CPUUtilization
Dimensions: InstanceId=i-0abc123
Threshold: 10

Reason: Threshold crossed...

Time: 03-Feb-2026 09:30:45 PM IST
```

**When new EC2 launches:**
```
🆕 Alarm Auto-Created

Account: 123456789012
Instance: my-server (i-0abc123)
Alarm: my-server-cpu-idle
Region: us-east-1
Threshold: CPU < 10%

Time: 03-Feb-2026 09:30:45 PM IST
```

## Teardown

```bash
# Single account
./teardown.sh

# All accounts
./teardown.sh --all
```

## Tags

All resources are tagged with:
```
Team = Team15
TaskId = Issue339
WorkloadType = Monitoring
```
