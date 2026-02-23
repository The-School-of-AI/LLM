#!/bin/sh
set -eu

#######################################
# AWS CloudWatch CPU Alerts to Telegram
# Idempotent - safe to run multiple times
# Only creates alarms for new instances
# Auto-creates alarms on new EC2 launches via EventBridge
#######################################

log_info() { echo "[INFO] $1"; }
log_warn() { echo "[WARN] $1"; }
log_error() { echo "[ERROR] $1"; }

# Default values
PREFIX="T15-IdleCPUMonitor-410"
AWS_REGION="${AWS_REGION:-us-east-1}"
CPU_THRESHOLD="${CPU_THRESHOLD:-10}"
EVALUATION_PERIODS="${EVALUATION_PERIODS:-3}"
PERIOD_SECONDS="${PERIOD_SECONDS:-300}"
SNS_TOPIC_NAME="${SNS_TOPIC_NAME:-${PREFIX}-Telegram-alert-topic}"
LAMBDA_FUNCTION_NAME="${LAMBDA_FUNCTION_NAME:-${PREFIX}-Telegram-alert-forwarder}"
LAMBDA_ROLE_NAME="${LAMBDA_ROLE_NAME:-${PREFIX}-Telegram-alert-lambda-execution-role}"
EVENTBRIDGE_LAMBDA_NAME="${EVENTBRIDGE_LAMBDA_NAME:-${PREFIX}-ec2-launch-alarm-creator}"
EVENTBRIDGE_RULE_NAME="${EVENTBRIDGE_RULE_NAME:-${PREFIX}-ec2-launch-cpu-alarm-rule}"

# Tags
TAG_TEAM="Team15"
TAG_TASK_ID="Issue410"
TAG_WORKLOAD_TYPE="Monitoring"

# Parse arguments
while [ $# -gt 0 ]; do
  case $1 in
    --telegram-token) TELEGRAM_BOT_TOKEN="$2"; shift 2 ;;
    --telegram-chat-id) TELEGRAM_CHAT_ID="$2"; shift 2 ;;
    --region) AWS_REGION="$2"; shift 2 ;;
    --cpu-threshold) CPU_THRESHOLD="$2"; shift 2 ;;
    --env-file) . "$2"; shift 2 ;;
    --help)
      echo "Usage: $0 --telegram-token TOKEN --telegram-chat-id CHAT_ID [options]"
      echo ""
      echo "Required:"
      echo "  --telegram-token    Telegram Bot API token"
      echo "  --telegram-chat-id  Telegram group chat ID"
      echo ""
      echo "Optional:"
      echo "  --region            AWS region (default: us-east-1)"
      echo "  --cpu-threshold     CPU % threshold (default: 10)"
      echo "  --env-file          Path to .env file"
      exit 0
      ;;
    *) log_error "Unknown parameter: $1"; exit 1 ;;
  esac
done

# Validate required parameters
if [ -z "${TELEGRAM_BOT_TOKEN:-}" ]; then
  log_error "TELEGRAM_BOT_TOKEN is required"
  exit 1
fi

if [ -z "${TELEGRAM_CHAT_ID:-}" ]; then
  log_error "TELEGRAM_CHAT_ID is required"
  exit 1
fi

AWS_ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
log_info "Account: ${AWS_ACCOUNT_ID} | Region: ${AWS_REGION}"

#######################################
# Step 1: Create IAM Role (if not exists)
#######################################
log_info "Checking IAM role..."

# hardcoding policy arn as its going to same across all accounts
POLICY_ARN="arn:aws:iam::${AWS_ACCOUNT_ID}:policy/${PREFIX}"

log_info "Checking if ${LAMBDA_ROLE_NAME} role exists..."
if ! aws iam get-role --role-name "${LAMBDA_ROLE_NAME}" >/dev/null 2>&1; then
  log_info "Inside IF - ${LAMBDA_ROLE_NAME} Role does not exist"
  log_info "Creating IAM role...${LAMBDA_ROLE_NAME}"
  aws iam create-role \
    --role-name "${LAMBDA_ROLE_NAME}" \
    --assume-role-policy-document '{
      "Version": "2012-10-17",
      "Statement": [{
        "Effect": "Allow",
        "Principal": {"Service": "lambda.amazonaws.com"},
        "Action": "sts:AssumeRole"
      }]
    }' \
    --tags Key=Team,Value=${TAG_TEAM} Key=TaskId,Value=${TAG_TASK_ID} Key=WorkloadType,Value=${TAG_WORKLOAD_TYPE} \
    > /dev/null

  log_info "Attaching AWSLambdaBasicExecutionRole policy to role...${LAMBDA_ROLE_NAME}"
  aws iam attach-role-policy \
    --role-name "${LAMBDA_ROLE_NAME}" \
    --policy-arn "arn:aws:iam::aws:policy/service-role/AWSLambdaBasicExecutionRole"

  # Add inline policy for CloudWatch alarms and EC2 describe
  log_info "Attaching ${POLICY_ARN} policy to role...${LAMBDA_ROLE_NAME}"
  aws iam attach-role-policy \
    --role-name "${LAMBDA_ROLE_NAME}" \
    --policy-arn ${POLICY_ARN}

  log_info "Waiting for role to propagate..."
  sleep 10
else
  log_info "Inside ELSE - ${LAMBDA_ROLE_NAME} Role exists"
  # Ensure inline policy exists for existing role
  log_info "Attaching ${POLICY_ARN} policy to role...${LAMBDA_ROLE_NAME}"
  aws iam attach-role-policy \
    --role-name "${LAMBDA_ROLE_NAME}" \
    --policy-arn ${POLICY_ARN} 2>/dev/null || true
fi

LAMBDA_ROLE_ARN="arn:aws:iam::${AWS_ACCOUNT_ID}:role/${LAMBDA_ROLE_NAME}"

# Attach inline policy so Lambda can read the SSM token parameter
log_info "Attaching SSM read inline policy to role...${LAMBDA_ROLE_NAME}"
aws iam put-role-policy \
  --role-name "${LAMBDA_ROLE_NAME}" \
  --policy-name "${PREFIX}-ssm-read" \
  --policy-document "{
    \"Version\": \"2012-10-17\",
    \"Statement\": [{
      \"Sid\": \"SSMTokenRead\",
      \"Effect\": \"Allow\",
      \"Action\": \"ssm:GetParameter\",
      \"Resource\": \"arn:aws:ssm:${AWS_REGION}:${AWS_ACCOUNT_ID}:parameter/${PREFIX}/*\"
    }]
  }"

#######################################
# Step 1b: Store Telegram token in SSM Parameter Store
#######################################
SSM_PARAM_NAME="/${PREFIX}/telegram-bot-token"
log_info "Storing Telegram bot token in SSM Parameter Store (${SSM_PARAM_NAME})..."
aws ssm put-parameter \
  --name "${SSM_PARAM_NAME}" \
  --value "${TELEGRAM_BOT_TOKEN}" \
  --type "SecureString" \
  --overwrite \
  --region "${AWS_REGION}" > /dev/null
log_info "Token stored in SSM."

#######################################
# Step 2: Create/Update Telegram Forwarder Lambda
#######################################
log_info "Checking Telegram forwarder Lambda..."

TEMP_DIR=$(mktemp -d)
cat > "${TEMP_DIR}/lambda_function.py" << 'PYTHON_EOF'
import json
import os
import boto3
import urllib.request
import urllib.parse
from datetime import datetime, timedelta

_token_cache = None

def get_telegram_token():
    global _token_cache
    if _token_cache:
        return _token_cache
    ssm = boto3.client('ssm')
    response = ssm.get_parameter(
        Name=os.environ['TELEGRAM_TOKEN_PARAM'],
        WithDecryption=True
    )
    _token_cache = response['Parameter']['Value']
    return _token_cache

def lambda_handler(event, context):
    bot_token = get_telegram_token()
    chat_id = os.environ['TELEGRAM_CHAT_ID']
    account_id = os.environ.get('AWS_ACCOUNT_ID', 'Unknown')

    try:
        if 'Records' in event:
            message = event['Records'][0]['Sns']['Message']
            try:
                alarm = json.loads(message)
                # Extract instance name using EC2 API
                instance_id = None
                region = alarm.get('Region', os.environ.get('AWS_REGION', 'us-east-1'))
                for dim in alarm.get('Trigger', {}).get('Dimensions', []):
                  if dim.get('name') == 'InstanceId':
                    instance_id = dim.get('value')
                    break

                instance_name = get_instance_name(instance_id)

                text = format_alarm(alarm, account_id, instance_name)
            except json.JSONDecodeError:
                text = "📢 *AWS Alert* ({})\n\n{}".format(account_id, message)
        else:
            text = "📢 *AWS Alert* ({})\n\n```\n{}\n```".format(account_id, json.dumps(event, indent=2))

        send_telegram(bot_token, chat_id, text)
        return {'statusCode': 200}

    except Exception as e:
        send_telegram(bot_token, chat_id, "❌ *Error* ({}): {}".format(account_id, str(e)))
        raise

def get_instance_name(instance_id):
    """Return the Name tag value, or instance_id if no Name tag exists."""
    if not instance_id:
        return "unknown"
    try:
        import boto3
        ec2 = boto3.client('ec2')
        response = ec2.describe_instances(InstanceIds=[instance_id])
        tags = response['Reservations'][0]['Instances'][0].get('Tags', [])
        for tag in tags:
            if tag.get('Key') == 'Name':
                name = tag.get('Value', '').strip()
                if name:
                    return name
    except Exception as e:
        print("Warning: could not get name for {}: {}".format(instance_id, e))
    return instance_id

def format_alarm(alarm, account_id, instance_name=None):
    name = alarm.get('AlarmName', 'Unknown')
    state = alarm.get('NewStateValue', 'Unknown')
    reason = alarm.get('NewStateReason', 'N/A')
    region = alarm.get('Region', 'Unknown')

    ts = alarm.get('StateChangeTime', '')
    try:
        utc_time = datetime.fromisoformat(ts.replace('Z', '+00:00'))
        ist_time = utc_time + timedelta(hours=5, minutes=30)
        time_str = ist_time.strftime('%d-%b-%Y %I:%M:%S %p IST')
    except Exception:
        time_str = ts

    if state == 'ALARM':
        emoji = '🚨'
    elif state == 'OK':
        emoji = '✅'
    else:
        emoji = '⚠️'

    trigger = alarm.get('Trigger', {})
    metric = trigger.get('MetricName', 'N/A')
    threshold = trigger.get('Threshold', 'N/A')
    dims = trigger.get('Dimensions', [])
    if dims:
        dim_str = ', '.join(["{}={}".format(d['name'], d['value']) for d in dims])
    else:
        dim_str = 'N/A'

    lines = [
      "{} *CPU Idle: {}*".format(emoji, instance_name if instance_name else name),
      "",
      "*Account:* {}".format(account_id),
      "*Alarm:* {}".format(name),
      "*Status:* {}".format(state),
      "*Region:* {}".format(region),
      "*Instance Name:* {}".format(instance_name if instance_name else "N/A"),
      "",
      "*Metric:* {}".format(metric),
      "*Dimensions:* {}".format(dim_str),
      "*Threshold:* {}".format(threshold),
      "",
      "*Reason:* {}".format(reason),
      "",
      "*Time:* {}".format(time_str)
    ]
    return "\n".join(lines)

def send_telegram(bot_token, chat_id, text):
    url = "https://api.telegram.org/bot{}/sendMessage".format(bot_token)
    if len(text) > 4000:
        text = text[:4000] + "\n...(truncated)"
    data = urllib.parse.urlencode({
        'chat_id': chat_id,
        'text': text,
        'parse_mode': 'Markdown'
    }).encode()
    req = urllib.request.Request(url, data=data)
    with urllib.request.urlopen(req, timeout=10) as r:
        return r.read()
PYTHON_EOF

cd "${TEMP_DIR}" && zip -q lambda.zip lambda_function.py

log_info "Checking Lambda function ${LAMBDA_FUNCTION_NAME} exists"
if aws lambda get-function --function-name "${LAMBDA_FUNCTION_NAME}" --region "${AWS_REGION}" >/dev/null 2>&1; then
  log_info "Inside IF Lambda function exists ${LAMBDA_FUNCTION_NAME}"
  log_info "Updating Telegram forwarder Lambda...${LAMBDA_FUNCTION_NAME}"
  aws lambda update-function-code \
    --function-name "${LAMBDA_FUNCTION_NAME}" \
    --zip-file "fileb://lambda.zip" \
    --region "${AWS_REGION}" > /dev/null

  sleep 5
  aws lambda update-function-configuration \
    --function-name "${LAMBDA_FUNCTION_NAME}" \
    --environment "Variables={TELEGRAM_TOKEN_PARAM=${SSM_PARAM_NAME},TELEGRAM_CHAT_ID=${TELEGRAM_CHAT_ID},AWS_ACCOUNT_ID=${AWS_ACCOUNT_ID}}" \
    --region "${AWS_REGION}" > /dev/null
else
  log_info "Inside ELSE, ${LAMBDA_FUNCTION_NAME} does not exist, Creating it..."
  aws lambda create-function \
    --function-name "${LAMBDA_FUNCTION_NAME}" \
    --runtime "python3.12" \
    --role "${LAMBDA_ROLE_ARN}" \
    --handler "lambda_function.lambda_handler" \
    --zip-file "fileb://lambda.zip" \
    --timeout 30 \
    --memory-size 128 \
    --environment "Variables={TELEGRAM_TOKEN_PARAM=${SSM_PARAM_NAME},TELEGRAM_CHAT_ID=${TELEGRAM_CHAT_ID},AWS_ACCOUNT_ID=${AWS_ACCOUNT_ID}}" \
    --tags Team=${TAG_TEAM},TaskId=${TAG_TASK_ID},WorkloadType=${TAG_WORKLOAD_TYPE} \
    --region "${AWS_REGION}" > /dev/null

  aws lambda wait function-active --function-name "${LAMBDA_FUNCTION_NAME}" --region "${AWS_REGION}"
fi

cd - > /dev/null
LAMBDA_ARN="arn:aws:lambda:${AWS_REGION}:${AWS_ACCOUNT_ID}:function:${LAMBDA_FUNCTION_NAME}"

#######################################
# Step 3: Create SNS Topic
#####################################
log_info "Checking SNS topic...${SNS_TOPIC_NAME}"

SNS_TOPIC_ARN=$(aws sns create-topic \
  --name "${SNS_TOPIC_NAME}" \
  --tags Key=Team,Value=${TAG_TEAM} Key=TaskId,Value=${TAG_TASK_ID} Key=WorkloadType,Value=${TAG_WORKLOAD_TYPE} \
  --region "${AWS_REGION}" \
  --query 'TopicArn' \
  --output text)

log_info "Received SNS topic ARN: ${SNS_TOPIC_ARN} from create-topic for ${SNS_TOPIC_NAME}"

# Add Lambda permission for SNS
log_info "Adding Lambda permission for SNS to invoke...${LAMBDA_FUNCTION_NAME}"
aws lambda add-permission \
  --function-name "${LAMBDA_FUNCTION_NAME}" \
  --statement-id "sns-invoke" \
  --action "lambda:InvokeFunction" \
  --principal "sns.amazonaws.com" \
  --source-arn "${SNS_TOPIC_ARN}" \
  --region "${AWS_REGION}" 2>/dev/null || true

# Subscribe Lambda to SNS
log_info "Subscribing Lambda to SNS topic...${SNS_TOPIC_NAME}"
aws sns subscribe \
  --topic-arn "${SNS_TOPIC_ARN}" \
  --protocol "lambda" \
  --notification-endpoint "${LAMBDA_ARN}" \
  --region "${AWS_REGION}" > /dev/null

#######################################
# Step 4: Create EventBridge Lambda (auto-alarm on EC2 launch)
#######################################
log_info "Checking EventBridge Lambda...${EVENTBRIDGE_LAMBDA_NAME}"

cat > "${TEMP_DIR}/eventbridge_lambda.py" << 'PYTHON_EOF'
import json
import os
import boto3
import urllib.request
import urllib.parse
from datetime import datetime, timedelta

cloudwatch = boto3.client('cloudwatch')
ec2 = boto3.client('ec2')

# Configuration from environment
CPU_THRESHOLD      = int(os.environ.get('CPU_THRESHOLD', '10'))
EVALUATION_PERIODS = int(os.environ.get('EVALUATION_PERIODS', '3'))
PERIOD_SECONDS     = int(os.environ.get('PERIOD_SECONDS', '300'))
SNS_TOPIC_ARN      = os.environ['SNS_TOPIC_ARN']
AWS_ACCOUNT_ID     = os.environ.get('AWS_ACCOUNT_ID', 'Unknown')
TELEGRAM_CHAT_ID   = os.environ['TELEGRAM_CHAT_ID']
TAG_TEAM           = os.environ.get('TAG_TEAM', '')
TAG_TASK_ID        = os.environ.get('TAG_TASK_ID', '')
TAG_WORKLOAD_TYPE  = os.environ.get('TAG_WORKLOAD_TYPE', '')
PREFIX             = os.environ.get('PREFIX', '')

_token_cache = None

def get_telegram_token():
    global _token_cache
    if _token_cache:
        return _token_cache
    ssm = boto3.client('ssm')
    response = ssm.get_parameter(
        Name=os.environ['TELEGRAM_TOKEN_PARAM'],
        WithDecryption=True
    )
    _token_cache = response['Parameter']['Value']
    return _token_cache

def lambda_handler(event, context):
    instance_id = event['detail']['instance-id']
    region = event['region']

    # Check opt-out tag: instances tagged IdleCPUAutoStop=false are skipped
    try:
        tag_response = ec2.describe_tags(
            Filters=[
                {'Name': 'resource-id', 'Values': [instance_id]},
                {'Name': 'key', 'Values': ['IdleCPUAutoStop']}
            ]
        )
        if tag_response['Tags'] and tag_response['Tags'][0]['Value'] == 'false':
            print("Instance {} opted out via IdleCPUAutoStop=false".format(instance_id))
            return {'statusCode': 200, 'body': 'Opted out — no alarm created'}
    except Exception as e:
        print("Warning: could not check IdleCPUAutoStop tag for {}: {}".format(instance_id, e))
        # Fail open: proceed to create alarm if tag check fails

    # Get instance name
    instance_name = get_instance_name(instance_id)
    alarm_name = "{}-{}-cpu-idle".format(PREFIX, instance_name)
    
    # Check if alarm already exists
    existing = cloudwatch.describe_alarms(AlarmNames=[alarm_name])
    if existing['MetricAlarms']:
        return {'statusCode': 200, 'body': 'Alarm already exists'}
    
    # Create alarm
    cloudwatch.put_metric_alarm(
        AlarmName=alarm_name,
        AlarmDescription="CPU below {}% for {}".format(CPU_THRESHOLD, instance_name),
        MetricName='CPUUtilization',
        Namespace='AWS/EC2',
        Statistic='Average',
        Period=PERIOD_SECONDS,
        Threshold=CPU_THRESHOLD,
        ComparisonOperator='LessThanThreshold',
        EvaluationPeriods=EVALUATION_PERIODS,
        Dimensions=[{'Name': 'InstanceId', 'Value': instance_id}],
        AlarmActions=[SNS_TOPIC_ARN, "arn:aws:automate:{}:ec2:stop".format(region)],
        OKActions=[SNS_TOPIC_ARN],
        TreatMissingData='notBreaching',
        Tags=[
            {'Key': 'Team', 'Value': TAG_TEAM},
            {'Key': 'TaskId', 'Value': TAG_TASK_ID},
            {'Key': 'WorkloadType', 'Value': TAG_WORKLOAD_TYPE}
        ]
    )
    
    # Notify Telegram
    notify_telegram(instance_id, instance_name, alarm_name, region)
    
    return {'statusCode': 200, 'body': 'Alarm created: {}'.format(alarm_name)}

def get_instance_name(instance_id):
    """Return the Name tag value, or instance_id if no Name tag exists."""
    if not instance_id:
        return "unknown"
    try:
        response = ec2.describe_instances(InstanceIds=[instance_id])
        tags = response['Reservations'][0]['Instances'][0].get('Tags', [])
        for tag in tags:
            if tag.get('Key') == 'Name':
                name = tag.get('Value', '').strip()
                if name:
                    return name
    except Exception as e:
        print("Warning: could not get name for {}: {}".format(instance_id, e))
    return instance_id



def notify_telegram(instance_id, instance_name, alarm_name, region):
    ist_time = datetime.utcnow() + timedelta(hours=5, minutes=30)
    time_str = ist_time.strftime('%d-%b-%Y %I:%M:%S %p IST')
    
    lines = [
        "🆕 *Alarm Auto-Created*",
        "",
        "*Account:* {}".format(AWS_ACCOUNT_ID),
        "*Instance:* {} ({})".format(instance_name, instance_id),
        "*Alarm:* {}".format(alarm_name),
        "*Region:* {}".format(region),
        "*Threshold:* CPU < {}%".format(CPU_THRESHOLD),
        "",
        "*Time:* {}".format(time_str)
    ]
    text = "\n".join(lines)
    
    url = "https://api.telegram.org/bot{}/sendMessage".format(get_telegram_token())
    data = urllib.parse.urlencode({
        'chat_id': TELEGRAM_CHAT_ID,
        'text': text,
        'parse_mode': 'Markdown'
    }).encode()
    req = urllib.request.Request(url, data=data)
    try:
        with urllib.request.urlopen(req, timeout=10) as r:
            r.read()
    except Exception:
        pass
PYTHON_EOF

cd "${TEMP_DIR}" && zip -q eventbridge_lambda.zip eventbridge_lambda.py

log_info "Checking EventBridge Lambda function ${EVENTBRIDGE_LAMBDA_NAME} exists"
if aws lambda get-function --function-name "${EVENTBRIDGE_LAMBDA_NAME}" --region "${AWS_REGION}" >/dev/null 2>&1; then
  log_info "Inside IF, Updating EventBridge Lambda...${EVENTBRIDGE_LAMBDA_NAME}"
  aws lambda update-function-code \
    --function-name "${EVENTBRIDGE_LAMBDA_NAME}" \
    --zip-file "fileb://eventbridge_lambda.zip" \
    --region "${AWS_REGION}" > /dev/null

  sleep 5
  log_info "Updating EventBridge Lambda configuration...${EVENTBRIDGE_LAMBDA_NAME}"
  aws lambda update-function-configuration \
    --function-name "${EVENTBRIDGE_LAMBDA_NAME}" \
    --environment "Variables={CPU_THRESHOLD=${CPU_THRESHOLD},EVALUATION_PERIODS=${EVALUATION_PERIODS},PERIOD_SECONDS=${PERIOD_SECONDS},SNS_TOPIC_ARN=${SNS_TOPIC_ARN},AWS_ACCOUNT_ID=${AWS_ACCOUNT_ID},TELEGRAM_TOKEN_PARAM=${SSM_PARAM_NAME},TELEGRAM_CHAT_ID=${TELEGRAM_CHAT_ID},TAG_TEAM=${TAG_TEAM},TAG_TASK_ID=${TAG_TASK_ID},TAG_WORKLOAD_TYPE=${TAG_WORKLOAD_TYPE},PREFIX=${PREFIX}}" \
    --region "${AWS_REGION}" > /dev/null
else
  log_info "Inside ELSE, Creating EventBridge Lambda...${EVENTBRIDGE_LAMBDA_NAME}"
  aws lambda create-function \
    --function-name "${EVENTBRIDGE_LAMBDA_NAME}" \
    --runtime "python3.12" \
    --role "${LAMBDA_ROLE_ARN}" \
    --handler "eventbridge_lambda.lambda_handler" \
    --zip-file "fileb://eventbridge_lambda.zip" \
    --timeout 30 \
    --memory-size 128 \
    --environment "Variables={CPU_THRESHOLD=${CPU_THRESHOLD},EVALUATION_PERIODS=${EVALUATION_PERIODS},PERIOD_SECONDS=${PERIOD_SECONDS},SNS_TOPIC_ARN=${SNS_TOPIC_ARN},AWS_ACCOUNT_ID=${AWS_ACCOUNT_ID},TELEGRAM_TOKEN_PARAM=${SSM_PARAM_NAME},TELEGRAM_CHAT_ID=${TELEGRAM_CHAT_ID},TAG_TEAM=${TAG_TEAM},TAG_TASK_ID=${TAG_TASK_ID},TAG_WORKLOAD_TYPE=${TAG_WORKLOAD_TYPE},PREFIX=${PREFIX}}" \
    --tags Team=${TAG_TEAM},TaskId=${TAG_TASK_ID},WorkloadType=${TAG_WORKLOAD_TYPE} \
    --region "${AWS_REGION}" > /dev/null

  aws lambda wait function-active --function-name "${EVENTBRIDGE_LAMBDA_NAME}" --region "${AWS_REGION}"
fi

EVENTBRIDGE_LAMBDA_ARN="arn:aws:lambda:${AWS_REGION}:${AWS_ACCOUNT_ID}:function:${EVENTBRIDGE_LAMBDA_NAME}"

#######################################
# Step 5: Create EventBridge Rule
#######################################
log_info "Checking EventBridge rule...${EVENTBRIDGE_RULE_NAME}"

# Create rule for EC2 running state
aws events put-rule \
  --name "${EVENTBRIDGE_RULE_NAME}" \
  --event-pattern '{
    "source": ["aws.ec2"],
    "detail-type": ["EC2 Instance State-change Notification"],
    "detail": {
      "state": ["running"]
    }
  }' \
  --state ENABLED \
  --description "Triggers alarm creation when EC2 instance starts" \
  --region "${AWS_REGION}" > /dev/null

# Tag the rule
log_info "Tagging EventBridge rule...${EVENTBRIDGE_RULE_NAME}"
aws events tag-resource \
  --resource-arn "arn:aws:events:${AWS_REGION}:${AWS_ACCOUNT_ID}:rule/${EVENTBRIDGE_RULE_NAME}" \
  --tags Key=Team,Value=${TAG_TEAM} Key=TaskId,Value=${TAG_TASK_ID} Key=WorkloadType,Value=${TAG_WORKLOAD_TYPE} \
  --region "${AWS_REGION}" 2>/dev/null || true

# Add Lambda permission for EventBridge
log_info "Adding Lambda permission for EventBridge to invoke...${EVENTBRIDGE_LAMBDA_NAME}"
aws lambda add-permission \
  --function-name "${EVENTBRIDGE_LAMBDA_NAME}" \
  --statement-id "eventbridge-invoke" \
  --action "lambda:InvokeFunction" \
  --principal "events.amazonaws.com" \
  --source-arn "arn:aws:events:${AWS_REGION}:${AWS_ACCOUNT_ID}:rule/${EVENTBRIDGE_RULE_NAME}" \
  --region "${AWS_REGION}" 2>/dev/null || true

# Add Lambda as target
log_info "Adding Lambda as target to EventBridge rule...${EVENTBRIDGE_RULE_NAME}"
aws events put-targets \
  --rule "${EVENTBRIDGE_RULE_NAME}" \
  --targets "Id=1,Arn=${EVENTBRIDGE_LAMBDA_ARN}" \
  --region "${AWS_REGION}" > /dev/null

cd - > /dev/null && rm -rf "${TEMP_DIR}"

#######################################
# Step 6: Create alarms for existing instances
#######################################

EXISTING_ALARMS=$(aws cloudwatch describe-alarms \
  --query "MetricAlarms[?ends_with(AlarmName, '-cpu-idle')].Dimensions[?Name=='InstanceId'].Value | []" \
  --output text \
  --region "${AWS_REGION}" 2>/dev/null | tr '\t' '\n' | sort -u)

log_info "Got alarms for existing instances...${EXISTING_ALARMS}"

RUNNING_INSTANCES=$(aws ec2 describe-instances \
  --filters "Name=instance-state-name,Values=running" \
  --query 'Reservations[].Instances[].[InstanceId,Tags[?Key==`Name`].Value | [0]]' \
  --output text \
  --region "${AWS_REGION}")
log_info "Got running instances...${RUNNING_INSTANCES}"

NEW_COUNT=0
SKIP_COUNT=0

if [ -z "${RUNNING_INSTANCES}" ]; then
  log_warn "No running instances found"
else
  echo "${RUNNING_INSTANCES}" | while IFS='	' read -r instance_id instance_name; do
    # Check opt-out tag: instances tagged IdleCPUAutoStop=false are skipped
    AUTO_STOP_TAG=$(aws ec2 describe-tags \
      --filters "Name=resource-id,Values=${instance_id}" "Name=key,Values=IdleCPUAutoStop" \
      --query 'Tags[0].Value' \
      --output text \
      --region "${AWS_REGION}" 2>/dev/null || echo "")
    if [ "${AUTO_STOP_TAG}" = "false" ]; then
      log_info "Skipping ${instance_id} — IdleCPUAutoStop=false"
      continue
    fi

    if [ -z "${instance_name}" ] || [ "${instance_name}" = "None" ]; then
      instance_name="${instance_id}"
    fi

    if echo "${EXISTING_ALARMS}" | grep -q "^${instance_id}$"; then
      SKIP_COUNT=$((SKIP_COUNT + 1))
      continue
    fi

    alarm_name="${PREFIX}-${instance_name}-cpu-idle"
    log_info "Creating alarm for ${instance_id} (${instance_name})..."

    aws cloudwatch put-metric-alarm \
      --alarm-name "${alarm_name}" \
      --alarm-description "CPU below ${CPU_THRESHOLD}% for ${instance_name}" \
      --metric-name "CPUUtilization" \
      --namespace "AWS/EC2" \
      --statistic "Average" \
      --period "${PERIOD_SECONDS}" \
      --threshold "${CPU_THRESHOLD}" \
      --comparison-operator "LessThanThreshold" \
      --evaluation-periods "${EVALUATION_PERIODS}" \
      --dimensions "Name=InstanceId,Value=${instance_id}" \
      --alarm-actions "${SNS_TOPIC_ARN}" "arn:aws:automate:${AWS_REGION}:ec2:stop" \
      --ok-actions "${SNS_TOPIC_ARN}" \
      --treat-missing-data "notBreaching" \
      --tags Key=Team,Value=${TAG_TEAM} Key=TaskId,Value=${TAG_TASK_ID} Key=WorkloadType,Value=${TAG_WORKLOAD_TYPE} \
      --region "${AWS_REGION}"

    NEW_COUNT=$((NEW_COUNT + 1))
  done
fi

#######################################
# Summary
#######################################
echo ""
echo "=========================================="
echo "Complete"
echo "=========================================="
echo "Account:            ${AWS_ACCOUNT_ID}"
echo "Region:             ${AWS_REGION}"
echo ""
echo "EventBridge rule:   ${EVENTBRIDGE_RULE_NAME}"
echo "  → New EC2 instances will automatically get CPU idle alarms"
echo ""
