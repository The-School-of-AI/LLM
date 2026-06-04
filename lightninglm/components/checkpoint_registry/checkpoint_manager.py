"""Checkpoint manager for P12 POC"""

import time
from datetime import datetime
from pathlib import Path

import boto3
import torch
import yaml

from lightninglm.components.metrics_server import get_metrics_server


class CheckpointManager:
    def __init__(self, config_path="config.yaml"):
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)

        self.s3_client = boto3.client("s3", region_name=self.config["aws"]["region"])
        self.bucket = self.config["aws"]["s3_bucket"]
        self.prefix = self.config["aws"]["checkpoint_prefix"]
        self.run_name = self.config["training"]["run_name"]

        self.local_dir = Path(f"checkpoints_local/{self.run_name}")
        self.local_dir.mkdir(parents=True, exist_ok=True)

        self.metadata = {"checkpoints": [], "latest": None}

        print("✓ CheckpointManager initialized")
        print(f"  S3: s3://{self.bucket}/{self.prefix}{self.run_name}/")
        print(f"  Local: {self.local_dir}")

    def save_checkpoint(self, model, optimizer, step, epoch, loss, **kwargs):
        start_time = time.time()

        checkpoint_name = f"checkpoint_step{step}_epoch{epoch}.pt"
        local_path = self.local_dir / checkpoint_name

        print(f"\n[CHECKPOINT] Saving step {step}...")

        checkpoint_state = {
            "step": step,
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "loss": loss,
            "timestamp": datetime.now().isoformat(),
        }
        checkpoint_state.update(kwargs)

        try:
            torch.save(checkpoint_state, local_path)
            print(f"  ✓ Saved locally: {checkpoint_name}")

            s3_key = f"{self.prefix}{self.run_name}/{checkpoint_name}"
            self.s3_client.upload_file(str(local_path), self.bucket, s3_key)
            print("  ✓ Uploaded to S3")

            checkpoint_info = {
                "name": checkpoint_name,
                "step": step,
                "epoch": epoch,
                "loss": loss,
                "local_path": str(local_path),
                "s3_key": s3_key,
                "size_mb": local_path.stat().st_size / (1024**2),
            }

            self.metadata["checkpoints"].append(checkpoint_info)
            self.metadata["latest"] = checkpoint_info

            duration = time.time() - start_time
            print(f"  ✓ Complete in {duration:.1f}s\n")

            metrics = get_metrics_server()
            metrics.record_checkpoint(duration, success=True)

            return checkpoint_info

        except Exception as e:
            print(f"  ✗ Checkpoint failed: {e}")
            metrics = get_metrics_server()
            metrics.record_checkpoint(0, success=False)
            raise

    def list_checkpoints(self):
        return self.metadata["checkpoints"]
