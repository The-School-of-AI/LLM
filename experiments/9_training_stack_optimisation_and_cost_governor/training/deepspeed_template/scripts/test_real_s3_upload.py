#!/usr/bin/env python3
"""
Simple script to test REAL S3 upload from local files.

This script actually uploads files to your S3 bucket to verify the async upload works.

Usage:
    # Set your bucket name
    export S3_BUCKET_NAME="your-bucket-name"
    export S3_PREFIX="test-uploads/checkpoints"
    
    # Run the test
    python scripts/test_real_s3_upload.py
"""

import os
import sys
import tempfile
import time
from pathlib import Path

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from aws.config import S3Config
from src.checkpoint import S3CheckpointManager


def test_real_upload():
    """Test uploading actual files to S3."""
    
    print("\n" + "="*70)
    print("🧪 REAL S3 UPLOAD TEST".center(70))
    print("="*70 + "\n")
    
    # Check for required environment variables
    bucket_name = os.environ.get('S3_BUCKET_NAME')
    if not bucket_name:
        print("❌ Error: S3_BUCKET_NAME environment variable not set!")
        print("\nSet it with:")
        print("  export S3_BUCKET_NAME='your-bucket-name'")
        print("  export S3_PREFIX='test-uploads/checkpoints'  # Optional")
        return False
    
    s3_prefix = os.environ.get('S3_PREFIX', 'test-uploads/checkpoints')
    
    print(f"📦 Bucket: {bucket_name}")
    print(f"📂 Prefix: {s3_prefix}")
    print()
    
    # Verify AWS credentials
    print("🔐 Checking AWS credentials...")
    try:
        import boto3
        s3_client = boto3.client('s3')
        s3_client.head_bucket(Bucket=bucket_name)
        print("✅ AWS credentials valid and bucket accessible\n")
    except Exception as e:
        print(f"❌ Error: {e}")
        print("\nMake sure:")
        print("  1. AWS credentials are configured")
        print("  2. Bucket exists")
        print("  3. You have write permissions")
        return False
    
    # Create temporary directory for test files
    with tempfile.TemporaryDirectory() as tmpdir:
        print(f"📁 Created temp directory: {tmpdir}\n")
        
        # Configure checkpoint manager
        with patch_distributed_env():
            config = S3Config(
                bucket_name=bucket_name,
                s3_prefix=s3_prefix,
                local_checkpoint_dir=tmpdir,
                verbose=True
            )
            
            print("🚀 Initializing S3 Checkpoint Manager...\n")
            manager = S3CheckpointManager(config)
            
            # Create test checkpoint
            checkpoint_tag = f'test_upload_{int(time.time())}'
            checkpoint_dir = os.path.join(tmpdir, checkpoint_tag)
            os.makedirs(checkpoint_dir)
            
            print(f"📝 Creating test files in: {checkpoint_dir}")
            
            # Create test files
            test_files = {
                'model.pt': 'Model weights data...' * 100,
                'optimizer.pt': 'Optimizer state...' * 100,
                'config.json': '{"learning_rate": 0.001}',
                'metadata.txt': f'Checkpoint created at {time.time()}'
            }
            
            for filename, content in test_files.items():
                file_path = os.path.join(checkpoint_dir, filename)
                with open(file_path, 'w') as f:
                    f.write(content)
                file_size = len(content) / 1024
                print(f"  ✅ {filename} ({file_size:.2f} KB)")
            
            print()
            
            # Queue for upload
            step = 100
            print(f"📤 Queueing checkpoint for async upload...")
            manager.upload_queue.put((checkpoint_dir, checkpoint_tag, step))
            with manager._upload_lock:
                manager.active_uploads.append(step)
            
            print(f"⏳ Waiting for upload to complete...\n")
            
            # Wait for upload
            start_time = time.time()
            manager.wait_for_uploads()
            upload_time = time.time() - start_time
            
            print(f"\n✅ Upload completed in {upload_time:.2f}s!\n")
            
            # Verify files in S3
            print("🔍 Verifying files in S3...")
            s3_keys = []
            for filename in test_files.keys():
                s3_key = f'{s3_prefix}/{checkpoint_tag}/{filename}'
                s3_keys.append(s3_key)
                
                try:
                    response = s3_client.head_object(
                        Bucket=bucket_name,
                        Key=s3_key
                    )
                    size_kb = response['ContentLength'] / 1024
                    print(f"  ✅ {filename} - {size_kb:.2f} KB")
                except Exception as e:
                    print(f"  ❌ {filename} - Not found: {e}")
                    return False
            
            print()
            print("="*70)
            print("✅ SUCCESS! Files uploaded to S3".center(70))
            print("="*70)
            print()
            print(f"📍 Location: s3://{bucket_name}/{s3_prefix}/{checkpoint_tag}/")
            print()
            print("🔗 View files:")
            for s3_key in s3_keys:
                print(f"  • s3://{bucket_name}/{s3_key}")
            print()
            
            # Ask if user wants to cleanup
            cleanup = input("🗑️  Delete test files from S3? [y/N]: ").lower().strip()
            if cleanup == 'y':
                print("\n🗑️  Cleaning up...")
                for s3_key in s3_keys:
                    try:
                        s3_client.delete_object(
                            Bucket=bucket_name,
                            Key=s3_key
                        )
                        print(f"  ✅ Deleted: {os.path.basename(s3_key)}")
                    except Exception as e:
                        print(f"  ⚠️  Could not delete {s3_key}: {e}")
                print("✅ Cleanup complete!")
            else:
                print("\n📌 Test files left in S3 for inspection")
            
            return True


def patch_distributed_env():
    """Context manager to patch distributed training environment."""
    class DistEnvPatch:
        def __enter__(self):
            self.old_values = {}
            env_vars = {
                'LOCAL_RANK': '0',
                'RANK': '0',
                'WORLD_SIZE': '1'
            }
            for key, value in env_vars.items():
                self.old_values[key] = os.environ.get(key)
                os.environ[key] = value
            return self
        
        def __exit__(self, *args):
            for key, value in self.old_values.items():
                if value is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = value
    
    return DistEnvPatch()


def list_s3_uploads():
    """List all files uploaded by this test script."""
    bucket_name = os.environ.get('S3_BUCKET_NAME')
    if not bucket_name:
        print("❌ Set S3_BUCKET_NAME environment variable")
        return
    
    s3_prefix = os.environ.get('S3_PREFIX', 'test-uploads/checkpoints')
    
    print(f"\n📋 Listing files in s3://{bucket_name}/{s3_prefix}\n")
    
    try:
        import boto3
        s3_client = boto3.client('s3')
        
        response = s3_client.list_objects_v2(
            Bucket=bucket_name,
            Prefix=s3_prefix
        )
        
        if 'Contents' not in response:
            print("📭 No files found")
            return
        
        # Group by checkpoint
        checkpoints = {}
        for obj in response['Contents']:
            key = obj['Key']
            parts = key.replace(s3_prefix + '/', '').split('/')
            if len(parts) >= 2:
                checkpoint = parts[0]
                if checkpoint not in checkpoints:
                    checkpoints[checkpoint] = []
                checkpoints[checkpoint].append(obj)
        
        for checkpoint, files in checkpoints.items():
            total_size = sum(f['Size'] for f in files)
            print(f"📦 {checkpoint}")
            for obj in files:
                filename = os.path.basename(obj['Key'])
                size_kb = obj['Size'] / 1024
                print(f"   • {filename} ({size_kb:.2f} KB)")
            print(f"   Total: {len(files)} files, {total_size / 1024:.2f} KB\n")
        
    except Exception as e:
        print(f"❌ Error: {e}")


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Test real S3 upload')
    parser.add_argument('--list', action='store_true', help='List uploaded files')
    args = parser.parse_args()
    
    if args.list:
        list_s3_uploads()
    else:
        success = test_real_upload()
        sys.exit(0 if success else 1)
