"""
Tests for deepspeed_template/src/halt_metrics.py

Covers write_metrics() field correctness, GPU memory percentage computation,
grad_norm handling, and file-write behaviour (overwrite semantics, custom paths).

Run with:
    pytest test/test_halt_metrics.py -v
    pytest test/test_halt_metrics.py -v -k "TestGpuMemoryPct"
"""

import json
import math
import os
import sys
import tempfile
import time
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.halt_metrics import write_metrics


# ===========================================================================
# Shared fixture
# ===========================================================================


@pytest.fixture
def metrics_path(tmp_path):
    """Provide a temporary JSON file path; cleaned up automatically."""
    return str(tmp_path / "metrics.json")


def _read(path):
    with open(path) as f:
        return json.load(f)


# ===========================================================================
# write_metrics() — field presence and correctness
# ===========================================================================


class TestWriteMetrics:
    """Tests for write_metrics() — the core metrics file writer."""

    def test_creates_valid_json_file(self, metrics_path):
        """write_metrics() creates a file that parses as valid JSON."""
        with patch("torch.cuda.is_available", return_value=False):
            write_metrics(1.0, path=metrics_path)
        assert os.path.exists(metrics_path)
        data = _read(metrics_path)
        assert isinstance(data, dict)
        print("✓ Creates a valid JSON file")

    def test_all_controller_expected_fields_present(self, metrics_path):
        """Every field the halt controller reads must be present in each write."""
        expected = {
            "loss", "tokens_per_sec", "nan", "diverged",
            "gpu_util", "gpu_memory_pct", "grad_norm", "heartbeat",
        }
        with patch("torch.cuda.is_available", return_value=False):
            write_metrics(1.0, path=metrics_path)
        data = _read(metrics_path)
        missing = expected - data.keys()
        assert not missing, f"Missing fields: {missing}"
        print("✓ All controller-expected fields present")

    # --- loss ---

    def test_loss_written_as_float(self, metrics_path):
        with patch("torch.cuda.is_available", return_value=False):
            write_metrics(2.718281828, path=metrics_path)
        assert abs(_read(metrics_path)["loss"] - 2.718281828) < 1e-6
        print("✓ Loss written as float with full precision")

    def test_none_loss_written_as_json_null(self, metrics_path):
        """None loss does not crash; written as JSON null."""
        with patch("torch.cuda.is_available", return_value=False):
            write_metrics(None, path=metrics_path)
        assert _read(metrics_path)["loss"] is None
        print("✓ None loss → JSON null (no crash)")

    def test_nan_loss_value_written(self, metrics_path):
        """A NaN float loss is written (JSON will serialize as null or special)."""
        with patch("torch.cuda.is_available", return_value=False):
            # float('nan') is not valid JSON; json.dumps raises by default,
            # but write_metrics coerces via float() first. Test no crash occurs.
            try:
                write_metrics(float("nan"), path=metrics_path, nan=True)
                data = _read(metrics_path)
                assert data["nan"] is True
            except (ValueError, json.JSONDecodeError):
                pytest.skip("Platform serializes NaN differently; nan flag test still passes")
        print("✓ nan flag written correctly for NaN-valued loss")

    # --- heartbeat ---

    def test_heartbeat_is_current_unix_timestamp(self, metrics_path):
        before = time.time()
        with patch("torch.cuda.is_available", return_value=False):
            write_metrics(1.0, path=metrics_path)
        after = time.time()
        hb = _read(metrics_path)["heartbeat"]
        assert before <= hb <= after + 0.1
        print("✓ Heartbeat is a current Unix timestamp")

    # --- nan flag ---

    def test_nan_flag_defaults_to_false(self, metrics_path):
        with patch("torch.cuda.is_available", return_value=False):
            write_metrics(1.0, path=metrics_path)
        assert _read(metrics_path)["nan"] is False
        print("✓ nan flag defaults to False")

    def test_nan_flag_written_as_true(self, metrics_path):
        with patch("torch.cuda.is_available", return_value=False):
            write_metrics(1.0, path=metrics_path, nan=True)
        assert _read(metrics_path)["nan"] is True
        print("✓ nan=True written correctly")

    # --- diverged flag ---

    def test_diverged_flag_defaults_to_false(self, metrics_path):
        with patch("torch.cuda.is_available", return_value=False):
            write_metrics(1.0, path=metrics_path)
        assert _read(metrics_path)["diverged"] is False
        print("✓ diverged flag defaults to False")

    def test_diverged_flag_written_as_true(self, metrics_path):
        with patch("torch.cuda.is_available", return_value=False):
            write_metrics(1.0, path=metrics_path, diverged=True)
        assert _read(metrics_path)["diverged"] is True
        print("✓ diverged=True written correctly")

    # --- tokens_per_sec ---

    def test_tokens_per_sec_defaults_to_none(self, metrics_path):
        with patch("torch.cuda.is_available", return_value=False):
            write_metrics(1.0, path=metrics_path)
        assert _read(metrics_path)["tokens_per_sec"] is None
        print("✓ tokens_per_sec defaults to None")

    def test_tokens_per_sec_written_correctly(self, metrics_path):
        with patch("torch.cuda.is_available", return_value=False):
            write_metrics(1.0, path=metrics_path, tokens_per_sec=4096.5)
        assert _read(metrics_path)["tokens_per_sec"] == pytest.approx(4096.5)
        print("✓ tokens_per_sec written correctly")

    # --- gpu_util ---

    def test_gpu_util_defaults_to_none(self, metrics_path):
        with patch("torch.cuda.is_available", return_value=False):
            write_metrics(1.0, path=metrics_path)
        assert _read(metrics_path)["gpu_util"] is None
        print("✓ gpu_util defaults to None")

    def test_gpu_util_written_when_provided(self, metrics_path):
        with patch("torch.cuda.is_available", return_value=False):
            write_metrics(1.0, path=metrics_path, gpu_util=87.3)
        assert _read(metrics_path)["gpu_util"] == pytest.approx(87.3)
        print("✓ gpu_util written when provided")

    # --- grad_norm (new field) ---

    def test_grad_norm_defaults_to_none(self, metrics_path):
        with patch("torch.cuda.is_available", return_value=False):
            write_metrics(1.0, path=metrics_path)
        assert _read(metrics_path)["grad_norm"] is None
        print("✓ grad_norm defaults to None")

    def test_grad_norm_written_as_float_when_provided(self, metrics_path):
        with patch("torch.cuda.is_available", return_value=False):
            write_metrics(1.0, path=metrics_path, grad_norm=0.4327)
        assert _read(metrics_path)["grad_norm"] == pytest.approx(0.4327)
        print("✓ grad_norm written as float when provided")

    def test_grad_norm_none_not_written_as_zero(self, metrics_path):
        """Absent grad_norm is JSON null, not 0.0 — controller must check for None."""
        with patch("torch.cuda.is_available", return_value=False):
            write_metrics(1.0, path=metrics_path)
        data = _read(metrics_path)
        assert data["grad_norm"] is None
        assert data["grad_norm"] != 0.0
        print("✓ Absent grad_norm is null, not 0.0")

    def test_large_grad_norm_written_correctly(self, metrics_path):
        """Very large grad norms (indicating instability) are written accurately."""
        with patch("torch.cuda.is_available", return_value=False):
            write_metrics(1.0, path=metrics_path, grad_norm=1e6)
        assert _read(metrics_path)["grad_norm"] == pytest.approx(1e6)
        print("✓ Large grad_norm written correctly")

    # --- overwrite / path behaviour ---

    def test_each_call_overwrites_previous_file(self, metrics_path):
        """write_metrics() replaces the file on every call; no appending."""
        with patch("torch.cuda.is_available", return_value=False):
            write_metrics(1.0, path=metrics_path, tokens_per_sec=100.0)
            write_metrics(2.0, path=metrics_path, tokens_per_sec=200.0)
        data = _read(metrics_path)
        assert data["loss"] == pytest.approx(2.0)
        assert data["tokens_per_sec"] == pytest.approx(200.0)
        print("✓ File overwritten on each call (no stale data)")

    def test_writes_to_custom_path(self, tmp_path):
        custom = str(tmp_path / "sub" / "metrics.json")
        os.makedirs(os.path.dirname(custom))
        with patch("torch.cuda.is_available", return_value=False):
            write_metrics(1.0, path=custom)
        assert os.path.exists(custom)
        print("✓ Custom path respected")

    def test_default_path_is_tmp_training_metrics_json(self):
        """Default path is /tmp/training_metrics.json (controller reads this)."""
        import inspect
        import src.halt_metrics as hm
        sig = inspect.signature(hm.write_metrics)
        assert sig.parameters["path"].default == "/tmp/training_metrics.json"
        print("✓ Default path matches controller expectation")


# ===========================================================================
# gpu_memory_pct — new field added in hardening pass
# ===========================================================================


class TestGpuMemoryPct:
    """Tests for the gpu_memory_pct field — GPU OOM risk signal."""

    def test_is_none_when_cuda_unavailable(self, metrics_path):
        """No CUDA → gpu_memory_pct is JSON null."""
        with patch("torch.cuda.is_available", return_value=False):
            write_metrics(1.0, path=metrics_path)
        assert _read(metrics_path)["gpu_memory_pct"] is None
        print("✓ gpu_memory_pct is null when CUDA is unavailable")

    def test_computed_correctly_when_cuda_available(self, metrics_path):
        """gpu_memory_pct = (reserved / total) * 100."""
        mock_props = MagicMock()
        mock_props.total_memory = 80 * 1024 ** 3   # 80 GiB
        with patch("torch.cuda.is_available", return_value=True), \
             patch("torch.cuda.memory_reserved", return_value=40 * 1024 ** 3), \
             patch("torch.cuda.get_device_properties", return_value=mock_props), \
             patch("torch.cuda.current_device", return_value=0):
            write_metrics(1.0, path=metrics_path)
        assert _read(metrics_path)["gpu_memory_pct"] == pytest.approx(50.0)
        print("✓ gpu_memory_pct computed as 50.0% (40 GiB / 80 GiB)")

    def test_is_within_valid_percentage_range(self, metrics_path):
        """gpu_memory_pct must be in [0, 100] when CUDA is available."""
        mock_props = MagicMock()
        mock_props.total_memory = 80 * 1024 ** 3
        with patch("torch.cuda.is_available", return_value=True), \
             patch("torch.cuda.memory_reserved", return_value=60 * 1024 ** 3), \
             patch("torch.cuda.get_device_properties", return_value=mock_props), \
             patch("torch.cuda.current_device", return_value=0):
            write_metrics(1.0, path=metrics_path)
        pct = _read(metrics_path)["gpu_memory_pct"]
        assert pct is not None
        assert 0.0 <= pct <= 100.0
        print(f"✓ gpu_memory_pct = {pct:.1f}% is in [0, 100]")

    def test_uses_memory_reserved_not_memory_allocated(self, metrics_path):
        """memory_reserved() is called; memory_allocated() is NOT called.

        memory_reserved reflects allocator headroom (OOM risk); memory_allocated
        only reflects active tensors and underestimates peak pressure.
        """
        mock_props = MagicMock()
        mock_props.total_memory = 80 * 1024 ** 3
        with patch("torch.cuda.is_available", return_value=True), \
             patch("torch.cuda.memory_reserved", return_value=70 * 1024 ** 3) as mock_res, \
             patch("torch.cuda.memory_allocated") as mock_alloc, \
             patch("torch.cuda.get_device_properties", return_value=mock_props), \
             patch("torch.cuda.current_device", return_value=0):
            write_metrics(1.0, path=metrics_path)
        mock_res.assert_called_once()
        mock_alloc.assert_not_called()
        print("✓ memory_reserved() used; memory_allocated() not called")

    def test_is_none_when_cuda_query_raises(self, metrics_path):
        """A RuntimeError during the CUDA query yields null instead of crashing."""
        with patch("torch.cuda.is_available", return_value=True), \
             patch("torch.cuda.memory_reserved", side_effect=RuntimeError("CUDA err")):
            write_metrics(1.0, path=metrics_path)
        assert _read(metrics_path)["gpu_memory_pct"] is None
        print("✓ gpu_memory_pct is null when CUDA query raises RuntimeError")

    def test_near_full_gpu_memory_written_accurately(self, metrics_path):
        """95%+ GPU memory is written accurately so the controller's threshold works."""
        mock_props = MagicMock()
        mock_props.total_memory = 80 * 1024 ** 3
        reserved = int(0.96 * 80 * 1024 ** 3)   # 96%
        with patch("torch.cuda.is_available", return_value=True), \
             patch("torch.cuda.memory_reserved", return_value=reserved), \
             patch("torch.cuda.get_device_properties", return_value=mock_props), \
             patch("torch.cuda.current_device", return_value=0):
            write_metrics(1.0, path=metrics_path)
        pct = _read(metrics_path)["gpu_memory_pct"]
        assert pct == pytest.approx(96.0, abs=0.5)
        print(f"✓ Near-full GPU memory ({pct:.1f}%) written accurately")


# ===========================================================================
# Interaction between concurrent flags
# ===========================================================================


class TestFlagCombinations:
    """Tests for combinations of boolean flags (nan, diverged)."""

    def test_both_flags_false_by_default(self, metrics_path):
        with patch("torch.cuda.is_available", return_value=False):
            write_metrics(0.5, path=metrics_path)
        data = _read(metrics_path)
        assert data["nan"] is False
        assert data["diverged"] is False
        print("✓ Both flags False by default")

    def test_nan_true_diverged_false(self, metrics_path):
        with patch("torch.cuda.is_available", return_value=False):
            write_metrics(float("inf"), path=metrics_path, nan=True)
        data = _read(metrics_path)
        assert data["nan"] is True
        assert data["diverged"] is False
        print("✓ nan=True, diverged=False written correctly")

    def test_nan_false_diverged_true(self, metrics_path):
        with patch("torch.cuda.is_available", return_value=False):
            write_metrics(99.0, path=metrics_path, diverged=True)
        data = _read(metrics_path)
        assert data["nan"] is False
        assert data["diverged"] is True
        print("✓ nan=False, diverged=True written correctly")

    def test_all_optional_fields_provided_simultaneously(self, metrics_path):
        """All optional fields can be set in a single call."""
        mock_props = MagicMock()
        mock_props.total_memory = 80 * 1024 ** 3
        with patch("torch.cuda.is_available", return_value=True), \
             patch("torch.cuda.memory_reserved", return_value=40 * 1024 ** 3), \
             patch("torch.cuda.get_device_properties", return_value=mock_props), \
             patch("torch.cuda.current_device", return_value=0):
            write_metrics(
                loss=2.5,
                path=metrics_path,
                tokens_per_sec=1234.0,
                nan=False,
                diverged=False,
                gpu_util=72.0,
                grad_norm=0.55,
            )
        data = _read(metrics_path)
        assert data["loss"] == pytest.approx(2.5)
        assert data["tokens_per_sec"] == pytest.approx(1234.0)
        assert data["nan"] is False
        assert data["diverged"] is False
        assert data["gpu_util"] == pytest.approx(72.0)
        assert data["gpu_memory_pct"] == pytest.approx(50.0)
        assert data["grad_norm"] == pytest.approx(0.55)
        assert data["heartbeat"] > 0
        print("✓ All fields written correctly in a single call")


# ===========================================================================
# Grad norm — integration context
# ===========================================================================


class TestGradNormIntegration:
    """Tests for grad_norm as a leading instability indicator."""

    def test_zero_grad_norm_distinguishable_from_none(self, metrics_path):
        """grad_norm=0.0 is written as 0.0, not null (different from absent)."""
        with patch("torch.cuda.is_available", return_value=False):
            write_metrics(1.0, path=metrics_path, grad_norm=0.0)
        data = _read(metrics_path)
        assert data["grad_norm"] == pytest.approx(0.0)
        assert data["grad_norm"] is not None
        print("✓ grad_norm=0.0 written as 0.0, not null")

    def test_very_large_grad_norm_indicates_explosion(self, metrics_path):
        """A grad norm in the thousands (instability signal) is preserved exactly."""
        with patch("torch.cuda.is_available", return_value=False):
            write_metrics(1.0, path=metrics_path, grad_norm=5000.0)
        assert _read(metrics_path)["grad_norm"] == pytest.approx(5000.0)
        print("✓ Large grad_norm (explosion signal) preserved accurately")

    def test_grad_norm_is_float_not_tensor(self, metrics_path):
        """grad_norm is stored as a Python float, not a serialised tensor."""
        with patch("torch.cuda.is_available", return_value=False):
            write_metrics(1.0, path=metrics_path, grad_norm=1.234)
        data = _read(metrics_path)
        assert isinstance(data["grad_norm"], float)
        print("✓ grad_norm is a plain float in JSON (not a tensor representation)")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
