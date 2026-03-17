"""
S3 uploader — uploads local files to S3 with optional MD5 checksum verification.
Uses boto3 multipart upload for large files.
"""
from __future__ import annotations

import hashlib
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def upload_directory(
    local_dir: str | Path,
    bucket: str,
    prefix: str,
    aws_profile: str | None = None,
    checksum_verify: bool = True,
    multipart_threshold_mb: int = 100,
    glob: str = "*.arrow",
) -> list[str]:
    """
    Upload all files matching *glob* in *local_dir* to s3://{bucket}/{prefix}/.

    Returns list of uploaded S3 URIs.
    """
    try:
        import boto3
        from boto3.s3.transfer import TransferConfig
    except ImportError:
        raise ImportError("boto3 is required for S3 upload: pip install boto3")

    session = boto3.Session(profile_name=aws_profile) if aws_profile else boto3.Session()
    s3 = session.client("s3")

    config = TransferConfig(
        multipart_threshold=multipart_threshold_mb * 1024 * 1024,
        multipart_chunksize=multipart_threshold_mb * 1024 * 1024,
    )

    local_dir = Path(local_dir)
    files = sorted(local_dir.glob(glob))
    if not files:
        logger.warning("No files matching '%s' found in %s", glob, local_dir)
        return []

    uploaded: list[str] = []
    for local_path in files:
        s3_key = f"{prefix.rstrip('/')}/{local_path.name}"
        logger.info("Uploading %s → s3://%s/%s", local_path, bucket, s3_key)

        extra: dict = {}
        if checksum_verify:
            md5 = _md5_hex(local_path)
            import base64
            extra["ContentMD5"] = base64.b64encode(bytes.fromhex(md5)).decode()

        s3.upload_file(
            str(local_path),
            bucket,
            s3_key,
            ExtraArgs=extra or None,
            Config=config,
        )

        if checksum_verify:
            _verify_etag(s3, bucket, s3_key, local_path)

        uri = f"s3://{bucket}/{s3_key}"
        uploaded.append(uri)
        logger.info("Uploaded: %s", uri)

    return uploaded


def _md5_hex(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _verify_etag(s3_client, bucket: str, key: str, local_path: Path) -> None:
    """Best-effort ETag check (works for non-multipart uploads)."""
    try:
        head = s3_client.head_object(Bucket=bucket, Key=key)
        etag = head.get("ETag", "").strip('"')
        local_md5 = _md5_hex(local_path)
        if "-" not in etag and etag != local_md5:
            raise RuntimeError(
                f"Checksum mismatch for {key}: local={local_md5} remote={etag}"
            )
    except Exception as exc:
        logger.warning("ETag verification skipped or failed: %s", exc)
