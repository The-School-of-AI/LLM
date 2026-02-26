import boto3

ACCOUNT_IDS = ["111111111111", "222222222222"]  # Replace with actual account IDs

ROLE_NAME = "RoleName"  # Replace with the actual role name to assume
REGION = "us-east-1"

QUOTAS = [
    ("ec2", "L-417A185B"),
    ("ec2", "L-C4BD4855"),
    ("ec2", "L-7212CCBC"),
    ("ec2", "L-DB2E81BA"),
    ("ec2", "L-3819A6DF"),
    ("ec2", "L-1216C47A"),
    ("ec2", "L-34B43A08"),
]

sts = boto3.client("sts")


def assume(account):
    creds = sts.assume_role(
        RoleArn=f"arn:aws:iam::{account}:role/{ROLE_NAME}",
        RoleSessionName="quota-check",
    )["Credentials"]

    return boto3.client(
        "service-quotas",
        region_name=REGION,
        aws_access_key_id=creds["AccessKeyId"],
        aws_secret_access_key=creds["SecretAccessKey"],
        aws_session_token=creds["SessionToken"],
    )


for acc in ACCOUNT_IDS:

    print(f"\n========== {acc} ==========")

    sq = assume(acc)

    # build status lookup
    history = sq.list_requested_service_quota_change_history(ServiceCode="ec2")[
        "RequestedQuotas"
    ]
    status_map = {r["QuotaCode"]: r["Status"] for r in history}

    for svc, code in QUOTAS:
        current = sq.get_service_quota(ServiceCode=svc, QuotaCode=code)["Quota"][
            "Value"
        ]
        status = status_map.get(code, "NO_REQUEST")

        print(f"{code} | status={status:12} | current={current}")
