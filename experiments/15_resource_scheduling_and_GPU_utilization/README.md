Here are the list of things from the original assignment. Please check below for updates against each 
- [x] expensive GPUs are never idle without explanation - This is taken care by this PR
- [ ] ~approved training windows are used fully and continuously - The infra is managed manually and being run on few account. So this was not addressed. Some early exploration and design is captured [here]~(https://github.com/The-School-of-AI/LLM/issues/230)
- [ ] validation, LoRA locking, growth checks, and synthetic injections are scheduled deliberately, not opportunistically - Not addressed by this PR. This did not come up in our group discussions and we did not address it
- [x] no compute is wasted due to poor orchestration - This is taken care by this PR


## AWS CloudWatch CPU Idle Alerts to Telegram

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


### Summary
This application allows one to setup alerts on low cpu usage (configurable) on any AWS accounts (please refer [IAM Setuo needed](https://raw.githubusercontent.com/The-School-of-AI/LLM/refs/heads/p15/idle_cpu_alert_telegram/experiments/15_resource_scheduling_and_GPU_utilization/aws-telegram-alerts/README.md) in the project README.md) 

It handles this in 2 ways
- Existing EC2 instances are scanned and necessary alerts are setup. **Lambda** --> **Telegram-alert-forwarder**
- An eventbridge is also configured that is triggered whenever any new EC2 instances is launched. This takes of future needs or new instances created after the script is run. **Lambda** --> **ec2-launch-alarm-creator**

You may also overrride these alerts and instance wont be stopped if `IdleCPUAutoStop` is set to false as a tag on an EC2 instance

Following resources for more clarity
| Resource | Name | Purpose |
|----------|------|---------|
| SSM Parameter | `/{PREFIX}/telegram-bot-token` | Encrypted Telegram token (SecureString) |
| IAM Role | `{PREFIX}-Telegram-alert-lambda-execution-role` | Lambda execution |
| Lambda | `{PREFIX}-Telegram-alert-forwarder` | Forwards alarms to Telegram |
| Lambda | `{PREFIX}-ec2-launch-alarm-creator` | Auto-creates alarms on EC2 launch |
| SNS Topic | `{PREFIX}-Telegram-alert-topic` | Bridges CloudWatch to Lambda |
| EventBridge Rule | `{PREFIX}-ec2-launch-cpu-alarm-rule` | Triggers on EC2 start |
| CloudWatch Alarms | `{instance-name}-cpu-idle` | One per EC2 instance |

