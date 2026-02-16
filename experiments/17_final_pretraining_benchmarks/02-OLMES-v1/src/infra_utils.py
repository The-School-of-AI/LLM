import os
import sys
import subprocess
import logging
from datetime import datetime

# Check if running in Google Colab
IS_COLAB = os.path.exists('/content') and os.path.exists('/usr/local/lib/python3.12/dist-packages')

def check_torchvision_nms(logger):
    """
    Proactively checks if torchvision and its NMS operation are valid.
    Specifically targets the common 'operator torchvision::nms does not exist' error.
    """
    try:
        import torch
        import torchvision
        # Attempt to access nms
        _ = torchvision.ops.nms
        return True
    except ImportError:
        # If it's not installed at all, we don't need to 'heal' it
        return True
    except (AttributeError, RuntimeError) as e:
        logger.warning(f"  [Auto-Heal] Detected torchvision/torch sync issue: {str(e)}")
        return False

def sync_colab_dependencies(logger):
    """
    Explicitly synchronizes torch and torchvision in Colab to prevent NMS errors.
    """
    if not IS_COLAB:
        return

    logger.info("  [Colab] Synchronizing torch and torchvision...")
    try:
        # Specifically target the latest stable versions that are compatible
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "--upgrade", "torch", "torchvision", "--index-url", "https://download.pytorch.org/whl/cu121"],
            check=True,
            capture_output=True,
            text=True
        )
        logger.info("  [Colab] Synchronization complete.")
    except Exception as e:
        logger.error(f"  [Error] Failed to sync Colab dependencies: {str(e)}")

def patch_vendor_requirements(logger, olmes_dir):
    """
    Patches the vendor's files to fix:
    1. Broken/unrealistic requirements in pyproject.toml (Colab stabilization).
    2. macOS filename issues in oe_eval/utils.py (replacing colons with underscores).
    3. Cross-version compatibility for lm-eval.
    """
    # --- 1. Patch pyproject.toml ---
    pyproject_path = os.path.join(olmes_dir, "pyproject.toml")
    if os.path.exists(pyproject_path):
        logger.info("  [OLMES] Patching vendor pyproject.toml...")
        try:
            with open(pyproject_path, "r") as f:
                content = f.read()
            patched = content.replace("torch>=2.8.0", "torch>=2.4.0")
            patched = patched.replace("transformers>=4.57.0", "transformers>=4.40.0")
            if patched != content:
                with open(pyproject_path, "w") as f:
                    f.write(patched)
                logger.info("  [OLMES] Successfully patched pyproject.toml")
        except Exception as e:
            logger.warning(f"  [OLMES] Failed to patch pyproject.toml: {str(e)}")

    # --- 2. Patch oe_eval/utils.py (macOS Filename Fix) ---
    utils_path = os.path.join(olmes_dir, "oe_eval", "utils.py")
    if os.path.exists(utils_path):
        logger.info("  [OLMES] Patching vendor utils.py for macOS compatibility...")
        try:
            with open(utils_path, "r") as f:
                content = f.read()
            
            target = 'def task_file_name(output_dir: str, task_idx: int, task_name: str, file_name: str) -> str:\n    return os.path.join(output_dir, f"task-{task_idx:03d}-{task_name}-{file_name}")'
            replacement = 'def task_file_name(output_dir: str, task_idx: int, task_name: str, file_name: str) -> str:\n    # Sanitize task_name for macOS/generic safety (no colons or slashes)\n    safe_name = task_name.replace(":", "_").replace("/", "_")\n    return os.path.join(output_dir, f"task-{task_idx:03d}-{safe_name}-{file_name}")'
            
            patched = content.replace(target, replacement)
            if patched != content:
                with open(utils_path, "w") as f:
                    f.write(patched)
                logger.info("  [OLMES] Successfully patched utils.py (Sanitized filenames enabled)")
        except Exception as e:
            logger.warning(f"  [OLMES] Failed to patch utils.py: {str(e)}")

    # --- 3. Patch for lm-eval Multi-Version Compatibility ---
    model_files = [
        os.path.join(olmes_dir, "oe_eval", "models", "eleuther_huggingface.py"),
        os.path.join(olmes_dir, "oe_eval", "models", "eleuther_olmo_core.py"),
        os.path.join(olmes_dir, "oe_eval", "models", "litellm.py")
    ]
    for model_path in model_files:
        if os.path.exists(model_path):
            try:
                with open(model_path, "r") as f:
                    content = f.read()
                if "LM_EVAL_VERSION_GUARD" in content:
                    continue
                target_imp = "from lm_eval.models.utils import Collator, pad_and_concat"
                replacement_imp = (
                    "# LM_EVAL_VERSION_GUARD (pad_and_concat)\n"
                    "try:\n"
                    "    from lm_eval.models.utils import Collator, pad_and_concat\n"
                    "except ImportError:\n"
                    "    from lm_eval.models.utils import Collator\n"
                    "    from lm_eval.models.utils_hf import pad_and_concat"
                )
                patched = content.replace(target_imp, replacement_imp)
                target_logger = "eval_logger = utils.eval_logger"
                replacement_logger = (
                    "# LM_EVAL_VERSION_GUARD (eval_logger)\n"
                    "import logging\n"
                    "try:\n"
                    "    eval_logger = utils.eval_logger\n"
                    "except AttributeError:\n"
                    "    eval_logger = logging.getLogger(\"lm_eval\")"
                )
                patched = patched.replace(target_logger, replacement_logger)
                if patched != content:
                    with open(model_path, "w") as f:
                        f.write(patched)
                    logger.info(f"  [OLMES] Successfully patched {os.path.basename(model_path)}")
            except Exception as e:
                logger.warning(f"  [OLMES] Failed to patch {model_path}: {str(e)}")

    # --- 4. Patch for Lazy Imports ---
    model_utils_path = os.path.join(olmes_dir, "oe_eval", "utilities", "model_utils.py")
    if os.path.exists(model_utils_path):
        try:
            with open(model_utils_path, "r") as f:
                content = f.read()
            if "IMPORT_GUARD" not in content:
                patches = [
                    ("from oe_eval.models.eleuther_olmo_core import OlmoCoreLM", 
                     "# IMPORT_GUARD\ntry:\n    from oe_eval.models.eleuther_olmo_core import OlmoCoreLM\nexcept ImportError:\n    OlmoCoreLM = None"),
                    ("from oe_eval.models.eleuther_vllm_causallms import VLLM_Verbose",
                     "# IMPORT_GUARD\ntry:\n    from oe_eval.models.eleuther_vllm_causallms import VLLM_Verbose\nexcept ImportError:\n    VLLM_Verbose = None"),
                    ("from oe_eval.models.judge_models import (\n    APIJudgeModel,\n    HFCausalJudgeModel,\n    VLLMJudgeModel,\n)",
                     "# IMPORT_GUARD\ntry:\n    from oe_eval.models.judge_models import (\n        APIJudgeModel,\n        HFCausalJudgeModel,\n        VLLMJudgeModel,\n    )\nexcept ImportError:\n    APIJudgeModel = None; HFCausalJudgeModel = None; VLLMJudgeModel = None")
                ]
                patched = content
                for t, r in patches:
                    patched = patched.replace(t, r)
                if patched != content:
                    with open(model_utils_path, "w") as f:
                        f.write(patched)
                    logger.info("  [OLMES] Successfully patched model_utils.py")
        except Exception as e:
            logger.warning(f"  [OLMES] Failed to patch model_utils.py: {str(e)}")

def ensure_olmes_vendor(logger):
    """
    Ensures that the OLMES vendor library is present and installed.
    """
    root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    vendor_dir = os.path.join(root_dir, "vendor")
    olmes_dir = os.path.join(vendor_dir, "olmes")

    try:
        if IS_COLAB:
            sync_colab_dependencies(logger)

        if not os.path.exists(olmes_dir):
            logger.info("  [OLMES] Vendor directory missing. Attempting to clone...")
            if not os.path.exists(vendor_dir): os.makedirs(vendor_dir)
            subprocess.run(["git", "clone", "https://github.com/allenai/olmes.git", olmes_dir], check=True)

        patch_vendor_requirements(logger, olmes_dir)

        import shutil
        uv_path = shutil.which("uv")
        install_success = False

        if uv_path:
            try:
                logger.info(f"  [OLMES] Using uv to install: {olmes_dir}")
                subprocess.run([uv_path, "pip", "install", "-e", olmes_dir], check=True)
                install_success = True
            except subprocess.CalledProcessError:
                logger.warning("  [OLMES] uv pip install failed, falling back to standard pip...")

        if not install_success:
            venv_python = os.path.join(root_dir, ".venv", "bin", "python3")
            py_exec = venv_python if os.path.exists(venv_python) else sys.executable
            logger.info(f"  [OLMES] Using pip to install: {olmes_dir}")
            subprocess.run([py_exec, "-m", "pip", "install", "-e", olmes_dir], check=True)

        # Proactive Auto-Heal Check
        if not check_torchvision_nms(logger):
            logger.info("  [Auto-Heal] Detected torchvision/torch sync issue.")
            # On macOS, automatic upgrades often break the environment further.
            # We skip the automatic fix and inject a shim or recommend manual install.
            logger.info("  [Auto-Heal] Skipping automatic upgrade on this platform. Injecting 'Fake NMS' shim...")
            try:
                import torch
                import torchvision
                if not hasattr(torchvision, 'ops'):
                    class FakeOps:
                        @staticmethod
                        def nms(*args, **kwargs): return None
                    torchvision.ops = FakeOps
                if check_torchvision_nms(logger):
                    logger.info("  [Auto-Heal] Shim injection successful.")
            except Exception as e:
                logger.warning(f"  [Auto-Heal] Shim injection failed: {str(e)}")

    except Exception as e:
        logger.error(f"  [Error] Failed to setup OLMES vendor: {str(e)}")
        sys.exit(1)

def setup_logging(run_dir, timestamp):
    log_dir = os.path.join(run_dir, "logs")
    if not os.path.exists(log_dir): os.makedirs(log_dir)
    log_path = os.path.join(log_dir, "execution.log")

    logger = logging.getLogger()
    if logger.hasHandlers(): logger.handlers.clear()
    logger.setLevel(logging.INFO)

    fh = logging.FileHandler(log_path)
    fh.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
    logger.addHandler(fh)

    ch = logging.StreamHandler()
    ch.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(ch)

    return logger, log_path
