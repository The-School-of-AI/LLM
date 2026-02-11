"""
run_tracker -- Lightweight run tracking and reproducibility for the synth pipeline.

Every pipeline command execution gets:
  - A unique run_id (UUID4)
  - A frozen config snapshot (all CLI args + computed params)
  - A config hash (SHA-256 of the frozen config JSON)
  - A git commit hash (HEAD at run start)
  - A synthetic generator version string
  - An environment fingerprint
  - A structured log file (JSON Lines)
  - Output file hashes for integrity verification

Usage:
    tracker = RunTracker(command_name="generate-bank", args=args)
    tracker.log("Starting generation for 5 skills", level="INFO")
    tracker.record_output("path/to/injection.jsonl")
    tracker.finish(status="success")
"""

import hashlib
import json
import os
import platform
import subprocess
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

SCRIPT_DIR = Path(__file__).resolve().parent.parent
RUNS_DIR = SCRIPT_DIR / "runs"


class RunTracker:
    """Tracks a single pipeline command execution for reproducibility."""

    def __init__(
        self,
        command_name: str,
        args: Any,
        runs_dir: Path | None = None,
    ):
        self._run_id = str(uuid.uuid4())
        self._command = command_name
        self._start_time = datetime.now(timezone.utc)
        self._status = "running"

        # Resolve runs directory
        self._runs_dir = runs_dir or RUNS_DIR
        self._run_dir = self._runs_dir / self._run_id
        self._run_dir.mkdir(parents=True, exist_ok=True)

        # Open log file handle
        self._log_path = self._run_dir / "run.log"
        self._log_handle = open(self._log_path, "w", encoding="utf-8")

        # Freeze config
        self._frozen_config = self._freeze_config(command_name, args)
        self._config_hash = self._compute_config_hash(self._frozen_config)

        # Capture version info
        self._git_hash = self._get_git_hash()
        model_name = getattr(args, "model", None)
        self._model_name = model_name
        self._ollama_model_digest = (
            self._get_ollama_model_digest(model_name) if model_name else None
        )
        self._env_fingerprint = self._get_environment_fingerprint()

        # Seed handling
        self._ollama_seed = getattr(args, "seed", None)
        self._python_random_seed = getattr(args, "seed", None)

        # Output file hashes (populated via record_output)
        self._output_hashes: dict[str, str] = {}

        # Log run start
        self.log(
            f"Run started: {self._run_id}",
            data={
                "command": command_name,
                "config_hash": self._config_hash,
                "git_hash": self._git_hash,
            },
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def run_id(self) -> str:
        return self._run_id

    @property
    def run_dir(self) -> Path:
        return self._run_dir

    @property
    def config_hash(self) -> str:
        return self._config_hash

    def log(
        self,
        message: str,
        level: str = "INFO",
        data: dict | None = None,
    ) -> None:
        """Append a structured log line to run.log."""
        entry: dict = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": level,
            "message": message,
        }
        if data:
            entry["data"] = data
        line = json.dumps(entry, ensure_ascii=False)
        self._log_handle.write(line + "\n")
        self._log_handle.flush()

    def record_output(self, file_path: str | Path) -> str:
        """Hash an output file and record it. Returns the SHA-256 hex digest."""
        file_path = Path(file_path)
        sha = hashlib.sha256()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                sha.update(chunk)
        digest = sha.hexdigest()
        # Use posix paths as keys for cross-platform consistency
        key = file_path.as_posix()
        self._output_hashes[key] = digest
        self.log(
            f"Output recorded: {file_path.name}",
            data={
                "file": key,
                "sha256": digest,
                "size_bytes": file_path.stat().st_size,
            },
        )
        return digest

    def finish(
        self,
        status: Literal["success", "error", "interrupted"] = "success",
        summary: dict | None = None,
    ) -> dict:
        """Finalize the run record. Writes config.json and closes log."""
        self._status = status
        end_time = datetime.now(timezone.utc)
        duration = (end_time - self._start_time).total_seconds()

        run_record = {
            "run_id": self._run_id,
            "command": self._command,
            "status": status,
            "start_time": self._start_time.isoformat(),
            "end_time": end_time.isoformat(),
            "duration_seconds": round(duration, 2),
            "config": self._frozen_config,
            "config_hash": self._config_hash,
            "version": {
                "git_hash": self._git_hash,
                "model": self._model_name,
                "ollama_model_digest": self._ollama_model_digest,
            },
            "seeds": {
                "ollama_seed": self._ollama_seed,
                "python_random_seed": self._python_random_seed,
            },
            "environment": self._env_fingerprint,
            "output_hashes": self._output_hashes,
        }
        if summary:
            run_record["summary"] = summary

        # Write config.json
        config_path = self._run_dir / "config.json"
        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(run_record, f, indent=2, ensure_ascii=False)

        self.log(
            f"Run finished: {status}",
            data={
                "duration_seconds": round(duration, 2),
                "output_files": len(self._output_hashes),
            },
        )

        # Close log handle
        self._log_handle.close()

        return run_record

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _get_git_hash() -> str | None:
        """Get current git HEAD hash."""
        try:
            result = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                capture_output=True,
                text=True,
                timeout=5,
                cwd=str(SCRIPT_DIR),
            )
            if result.returncode == 0:
                return result.stdout.strip()
        except (FileNotFoundError, subprocess.TimeoutExpired):
            pass
        return None

    @staticmethod
    def _get_ollama_version() -> str | None:
        """Get Ollama server version."""
        import urllib.request

        try:
            base = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
            url = f"{base}/api/version"
            req = urllib.request.Request(url, method="GET")
            with urllib.request.urlopen(req, timeout=5) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                return data.get("version")
        except Exception:
            return None

    @staticmethod
    def _get_ollama_model_digest(model_name: str) -> str | None:
        """Get Ollama model digest (weights hash)."""
        import urllib.request

        try:
            base = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
            url = f"{base}/api/show"
            payload = json.dumps({"name": model_name}).encode("utf-8")
            req = urllib.request.Request(
                url,
                data=payload,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=10) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                return data.get("digest")
        except Exception:
            return None

    @staticmethod
    def _get_environment_fingerprint() -> dict:
        """Capture environment info for reproducibility."""
        fp: dict = {
            "python_version": sys.version,
            "platform": platform.platform(),
            "machine": platform.machine(),
            "ollama_version": RunTracker._get_ollama_version(),
        }
        # Check key package versions
        for pkg in ("pyarrow", "pandas", "torch", "transformers"):
            try:
                mod = __import__(pkg)
                fp[f"{pkg}_version"] = getattr(mod, "__version__", "unknown")
            except ImportError:
                fp[f"{pkg}_version"] = None
        return fp

    @staticmethod
    def _freeze_config(command_name: str, args: Any) -> dict:
        """Convert argparse Namespace to a serializable frozen config dict."""
        args_dict = {}
        for k, v in vars(args).items():
            if k.startswith("_"):
                continue
            if isinstance(v, Path):
                v = str(v)
            args_dict[k] = v
        return {
            "command": command_name,
            "args": args_dict,
            "frozen_at": datetime.now(timezone.utc).isoformat(),
        }

    @staticmethod
    def _compute_config_hash(frozen_config: dict) -> str:
        """SHA-256 of deterministic JSON serialization of frozen config."""
        config_bytes = json.dumps(
            frozen_config, sort_keys=True, ensure_ascii=True
        ).encode("utf-8")
        return hashlib.sha256(config_bytes).hexdigest()
