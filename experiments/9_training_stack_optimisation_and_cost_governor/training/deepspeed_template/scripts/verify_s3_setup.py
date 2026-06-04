#!/usr/bin/env python3
"""
Verify S3 Setup for Checkpoint Manager.

This script verifies that:
1. AWS credentials are configured
2. S3 bucket exists and is accessible
3. Required permissions are granted
4. boto3 is installed correctly

Usage:
    python scripts/verify_s3_setup.py --bucket my-bucket --region us-east-1

    # Or with environment variables
    export S3_BUCKET_NAME=my-bucket
    export S3_REGION=us-east-1
    python scripts/verify_s3_setup.py
"""

import argparse
import os
import sys
import tempfile
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def check_boto3_installation():
    """Check if boto3 is installed."""
    print("Checking boto3 installation...")
    try:
        import boto3
        import botocore

        print(f"  ✓ boto3 version: {boto3.__version__}")
        print(f"  ✓ botocore version: {botocore.__version__}")
        return True
    except ImportError as e:
        print(f"  ✗ boto3 not installed: {e}")
        print("\n  Install with: pip install boto3 botocore")
        return False


def check_aws_credentials():
    """Check if AWS credentials are configured."""
    print("\nChecking AWS credentials...")

    try:
        import boto3

        # Try to get credentials
        session = boto3.Session()
        credentials = session.get_credentials()

        if credentials is None:
            print("  ✗ No AWS credentials found")
            print("\n  Configure credentials using one of:")
            print("    1. AWS CLI: aws configure")
            print("    2. Environment variables:")
            print("       export AWS_ACCESS_KEY_ID=your-key")
            print("       export AWS_SECRET_ACCESS_KEY=your-secret")
            print("    3. IAM role (for EC2/SageMaker)")
            return False

        # Get caller identity to verify credentials work
        sts = boto3.client("sts")
        identity = sts.get_caller_identity()

        print(f"  ✓ AWS Account: {identity['Account']}")
        print(f"  ✓ User/Role: {identity['Arn']}")
        return True

    except Exception as e:
        print(f"  ✗ Error checking credentials: {e}")
        return False


def check_bucket_access(bucket_name, region):
    """Check if S3 bucket exists and is accessible."""
    print(f"\nChecking S3 bucket: s3://{bucket_name}")

    try:
        import boto3
        from botocore.exceptions import ClientError

        s3_client = boto3.client("s3", region_name=region)

        # Check if bucket exists
        try:
            s3_client.head_bucket(Bucket=bucket_name)
            print("  ✓ Bucket exists and is accessible")
        except ClientError as e:
            error_code = e.response["Error"]["Code"]
            if error_code == "404":
                print("  ✗ Bucket does not exist")
                print(f"\n  Create with: aws s3 mb s3://{bucket_name}")
                return False
            elif error_code == "403":
                print("  ✗ Access denied to bucket")
                print("\n  Check IAM permissions for S3 access")
                return False
            else:
                print(f"  ✗ Error accessing bucket: {e}")
                return False

        # Check bucket location
        try:
            location = s3_client.get_bucket_location(Bucket=bucket_name)
            bucket_region = location["LocationConstraint"] or "us-east-1"
            print(f"  ✓ Bucket region: {bucket_region}")

            if bucket_region != region:
                print(
                    f"  ⚠️  Warning: Bucket is in {bucket_region}, "
                    f"but you specified {region}"
                )
        except ClientError as e:
            print(f"  ⚠️  Could not get bucket location: {e}")

        return True

    except Exception as e:
        print(f"  ✗ Error: {e}")
        return False


def check_bucket_permissions(bucket_name, region):
    """Check S3 bucket permissions."""
    print("\nChecking S3 permissions...")

    try:
        import boto3
        from botocore.exceptions import ClientError

        s3_client = boto3.client("s3", region_name=region)
        test_key = "checkpoint-test/test.txt"
        test_content = b"test checkpoint data"

        # Test PutObject
        print("  Testing PutObject permission...")
        try:
            s3_client.put_object(Bucket=bucket_name, Key=test_key, Body=test_content)
            print("    ✓ PutObject: OK")
        except ClientError as e:
            print(f"    ✗ PutObject: FAILED - {e}")
            return False

        # Test GetObject
        print("  Testing GetObject permission...")
        try:
            response = s3_client.get_object(Bucket=bucket_name, Key=test_key)
            data = response["Body"].read()
            if data == test_content:
                print("    ✓ GetObject: OK")
            else:
                print("    ✗ GetObject: Data mismatch")
                return False
        except ClientError as e:
            print(f"    ✗ GetObject: FAILED - {e}")
            return False

        # Test ListObjects
        print("  Testing ListObjects permission...")
        try:
            response = s3_client.list_objects_v2(
                Bucket=bucket_name, Prefix="checkpoint-test/"
            )
            print("    ✓ ListObjects: OK")
        except ClientError as e:
            print(f"    ✗ ListObjects: FAILED - {e}")
            return False

        # Test DeleteObject
        print("  Testing DeleteObject permission...")
        try:
            s3_client.delete_object(Bucket=bucket_name, Key=test_key)
            print("    ✓ DeleteObject: OK")
        except ClientError as e:
            print(f"    ✗ DeleteObject: FAILED - {e}")
            return False

        print("\n  ✓ All required permissions verified")
        return True

    except Exception as e:
        print(f"  ✗ Error testing permissions: {e}")
        return False


def test_checkpoint_manager(bucket_name, region):
    """Test S3CheckpointManager initialization."""
    print("\nTesting S3CheckpointManager...")

    try:
        from aws.config import S3Config
        from src.checkpoint import S3CheckpointManager

        # Create config
        with tempfile.TemporaryDirectory() as tmpdir:
            config = S3Config(
                bucket_name=bucket_name,
                s3_prefix="checkpoint-test",
                region=region,
                local_checkpoint_dir=tmpdir,
                verbose=False,
            )

            # Validate config
            print("  Validating configuration...")
            config.validate()
            print("    ✓ Configuration valid")

            # Initialize manager
            print("  Initializing checkpoint manager...")

            # Mock distributed environment
            os.environ["LOCAL_RANK"] = "0"
            os.environ["RANK"] = "0"
            os.environ["WORLD_SIZE"] = "1"

            manager = S3CheckpointManager(config)
            print("    ✓ Manager initialized")
            print(f"    ✓ World size: {manager.world_size}")
            print(f"    ✓ Global rank: {manager.global_rank}")
            print(f"    ✓ Local rank: {manager.local_rank}")

            return True

    except Exception as e:
        print(f"  ✗ Error: {e}")
        import traceback

        traceback.print_exc()
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Verify S3 setup for checkpoint manager"
    )
    parser.add_argument(
        "--bucket",
        type=str,
        default=os.getenv("S3_BUCKET_NAME"),
        help="S3 bucket name (or set S3_BUCKET_NAME env var)",
    )
    parser.add_argument(
        "--region",
        type=str,
        default=os.getenv("S3_REGION", "us-east-1"),
        help="AWS region (default: us-east-1)",
    )
    parser.add_argument(
        "--skip-permissions",
        action="store_true",
        help="Skip permission tests (read-only check)",
    )

    args = parser.parse_args()

    print("=" * 70)
    print("S3 Checkpoint Setup Verification")
    print("=" * 70)

    # Check bucket name
    if not args.bucket:
        print("\n✗ No bucket specified")
        print("  Provide via --bucket or S3_BUCKET_NAME environment variable")
        sys.exit(1)

    # Run checks
    checks = []

    checks.append(("boto3 installation", check_boto3_installation()))

    if checks[-1][1]:  # Only continue if boto3 is installed
        checks.append(("AWS credentials", check_aws_credentials()))

        if checks[-1][1]:  # Only continue if credentials are valid
            checks.append(
                ("S3 bucket access", check_bucket_access(args.bucket, args.region))
            )

            if checks[-1][1] and not args.skip_permissions:
                checks.append(
                    (
                        "S3 permissions",
                        check_bucket_permissions(args.bucket, args.region),
                    )
                )

            if checks[-1][1]:
                checks.append(
                    (
                        "Checkpoint manager",
                        test_checkpoint_manager(args.bucket, args.region),
                    )
                )

    # Summary
    print("\n" + "=" * 70)
    print("Summary")
    print("=" * 70)

    passed = sum(1 for _, result in checks if result)
    total = len(checks)

    for check_name, result in checks:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"  {status}: {check_name}")

    print(f"\nResults: {passed}/{total} checks passed")

    if passed == total:
        print("\n🎉 All checks passed! S3 checkpoint system is ready to use.")
        print("\nYou can now run training with:")
        print("  deepspeed main.py \\")
        print("    --deepspeed_config deepspeed/zero-2-moe.json \\")
        print(f"    --s3_bucket {args.bucket} \\")
        print("    --s3_prefix experiments/my-model \\")
        print("    --checkpoint_interval 100")
        sys.exit(0)
    else:
        print("\n⚠️  Some checks failed. Please fix the issues above.")
        sys.exit(1)


if __name__ == "__main__":
    main()
