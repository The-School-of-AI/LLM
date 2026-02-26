import json
from datetime import datetime, timezone
from pathlib import Path

from repro.config import freeze_config
from repro.env import capture_env
from repro.git import get_commit_hash, get_repo_url, is_repo_dirty
from repro.ids import generate_training_run_id
from repro.manifest import GitInfo, RunManifest
from repro.s3 import ImmutableS3Writer
from repro.seeds import capture_seeds

TMP_ROOT = Path(".repro_tmp")


class RunContext:
    def __init__(self, run_id: str, run_dir: Path, s3: ImmutableS3Writer):
        self.run_id = run_id
        self.run_dir = run_dir
        self.s3 = s3

    def save_output(self, rel_path: str, obj):
        path = self.run_dir / rel_path
        path.parent.mkdir(parents=True, exist_ok=True)

        if isinstance(obj, (dict, list)):
            path.write_text(json.dumps(obj, indent=2))
        else:
            raise ValueError("Unsupported output type")

    def finalize(self):
        for p in self.run_dir.rglob("*"):
            if p.is_file():
                self.s3.upload_file(p, p.relative_to(self.run_dir).as_posix())


def start_training_run(config: dict, seed: int, bucket="experiment-registry"):
    run_id = generate_training_run_id()
    run_dir = TMP_ROOT / "runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=False)

    config_hash = freeze_config(config, run_dir / "frozen_config.yaml")
    capture_env(run_dir / "env.json")
    capture_seeds(seed, run_dir / "seeds.json")

    manifest = RunManifest(
        run_id=run_id,
        pipeline="training",
        created_at=datetime.now(timezone.utc),
        git=GitInfo(
            repo_url=get_repo_url(),
            commit_hash=get_commit_hash(),
            dirty=is_repo_dirty(),
        ),
        config_hash=config_hash,
        seed=seed,
        status="STARTED",
    )

    (run_dir / "manifest.json").write_text(manifest.model_dump_json(indent=2))

    s3 = ImmutableS3Writer(bucket, f"runs/{run_id}")
    return RunContext(run_id, run_dir, s3)


def finalize_run(ctx: RunContext, status: str = "COMPLETED"):
    assert status in {"COMPLETED", "FAILED", "ABORTED"}

    manifest_path = ctx.run_dir / "manifest.json"
    manifest = RunManifest.model_validate_json(manifest_path.read_text())
    manifest.status = status

    # Rewrite manifest locally
    manifest_path.write_text(manifest.model_dump_json(indent=2))

    # Upload all artifacts EXCEPT manifest
    for p in sorted(ctx.run_dir.rglob("*")):
        if p.is_file() and p.name != "manifest.json":
            ctx.s3.upload_file(p, p.relative_to(ctx.run_dir).as_posix())

    # Upload manifest LAST (atomic completion)
    ctx.s3.upload_file(manifest_path, "manifest.json")
