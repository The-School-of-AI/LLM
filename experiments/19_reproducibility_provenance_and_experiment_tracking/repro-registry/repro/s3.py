from pathlib import Path

import boto3


class ImmutableS3Writer:
    def __init__(self, bucket: str, prefix: str):
        self.bucket = bucket
        self.prefix = prefix
        self.s3 = boto3.client("s3")

    def upload_file(self, local: Path, key: str):
        full_key = f"{self.prefix}/{key}"
        self.s3.upload_file(
            str(local),
            self.bucket,
            full_key,
            ExtraArgs={"ACL": "bucket-owner-full-control"},
        )
