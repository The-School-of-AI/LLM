import time

import boto3
from botocore.exceptions import ClientError

ACCOUNT_IDS = ["111111111111", "222222222222"]  # Replace with actual account IDs

ROLE_NAME = "RoleName"  # Replace with the actual role name to assume
REGION = "us-east-1"

QUOTAS = [
    ("ec2", "L-417A185B", 192),
    ("ec2", "L-C4BD4855", 192),
    ("ec2", "L-7212CCBC", 96),
    ("ec2", "L-DB2E81BA", 192),
    ("ec2", "L-3819A6DF", 192),
    ("ec2", "L-1216C47A", 128),
    ("ec2", "L-34B43A08", 128),
]

sts = boto3.client("sts")


def assume(account_id):
    creds = sts.assume_role(
        RoleArn=f"arn:aws:iam::{account_id}:role/{ROLE_NAME}",
        RoleSessionName="quota-test",
    )["Credentials"]

    return boto3.client(
        "service-quotas",
        region_name=REGION,
        aws_access_key_id=creds["AccessKeyId"],
        aws_secret_access_key=creds["SecretAccessKey"],
        aws_session_token=creds["SessionToken"],
    )


for ACCOUNT_ID in ACCOUNT_IDS:

    print(f"\n========== {ACCOUNT_ID} ==========")

    try:
        sq = assume(ACCOUNT_ID)
        print("Assumed role successfully\n")

        for svc, code, target in QUOTAS:
            current = sq.get_service_quota(ServiceCode=svc, QuotaCode=code)["Quota"][
                "Value"
            ]

            print(f"{code} current={current} target={target}")

            if current >= target:
                print(" -> already sufficient")
                time.sleep(0.5)
                continue

            try:
                resp = sq.request_service_quota_increase(
                    ServiceCode=svc, QuotaCode=code, DesiredValue=target
                )

                req_id = resp["RequestedQuota"]["Id"]
                print(f" -> request submitted | request_id={req_id}")

            except ClientError as e:
                if e.response["Error"]["Code"] == "ResourceAlreadyExistsException":
                    print(" -> already pending (skipped)")
                else:
                    raise

            time.sleep(0.5)

    except Exception as e:
        print("ERROR:", str(e))
