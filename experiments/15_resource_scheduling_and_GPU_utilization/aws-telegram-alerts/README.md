# AWS CloudWatch CPU Alerts to Telegram

Monitors EC2 idle CPU and sends alerts to Telegram. Idempotent—run anytime to add alarms for new instances only.

## Files

```
setup.sh          # Deploy to single account
deploy-all.sh     # Deploy to multiple accounts
teardown.sh       # Remove all resources
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

## IAM Setup

Attach `iam-policy.json` to IAM users/roles that will run the setup scripts.

## Single Account

```bash
chmod +x setup.sh

./setup.sh \
  --telegram-token "123456789:ABCdef..." \
  --telegram-chat-id "-1001234567890" \
  --region "ap-south-1"
```

Run again after launching new instances—it skips existing alarms.

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

   export TELEGRAM_BOT_TOKEN="123456789:ABCdef..."
   export TELEGRAM_CHAT_ID="-1001234567890"
   export AWS_REGION="ap-south-1"

   ./deploy-all.sh
   ```

## Options

| Flag | Default | Description |
|------|---------|-------------|
| `--telegram-token` | required | Bot API token |
| `--telegram-chat-id` | required | Group chat ID |
| `--region` | us-east-1 | AWS region |
| `--cpu-threshold` | 10 | Alert when CPU < this % |

## Alert Format

```
🚨 CloudWatch Alarm

Account: 123456789012
Alarm: cpu-idle-i-0abc123
Status: ALARM
Region: ap-south-1

Metric: CPUUtilization
Dimensions: InstanceId=i-0abc123
Threshold: 10

Reason: Threshold crossed...

Time: 03-Feb-2026 09:30:45 PM IST
```

## Teardown

```bash
# Single account
./teardown.sh

# Specific account
AWS_PROFILE=production ./teardown.sh
```
## References
[How to get Telegram Bot Chat ID](https://gist.github.com/nafiesl/4ad622f344cd1dc3bb1ecbe468ff9f8a)
