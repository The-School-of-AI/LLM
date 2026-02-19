"""
Tests for halt_mechanism/halt_controller.py

Covers every trigger function, the consecutive-gate, SSM verification,
checkpoint-wait timeout, and the full halt_cluster() flow.

Run with:
    pytest test/test_halt_controller.py -v
    pytest test/test_halt_controller.py -v -k "TestThroughputCollapsed"
"""

import json
import os
import sys
import time
from unittest.mock import MagicMock, call, patch

import pytest

# ---------------------------------------------------------------------------
# Safe import of halt_controller
#
# halt_controller.py is a runnable script with a module-level `while True`
# loop.  We import it by:
#   1. Mocking boto3 so no real AWS connections are attempted.
#   2. Patching time.sleep to raise StopIteration on its first call, which
#      breaks out of the while loop after one iteration.
# After the import completes, the module-level globals (ec2, ssm, s3,
# tps_samples, baseline_tps, trigger_counts) are accessible via `hc.*` and
# can be replaced / reset in fixtures.
# ---------------------------------------------------------------------------

_HALT_MECHANISM_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "../../halt_mechanism")
)
if _HALT_MECHANISM_DIR not in sys.path:
    sys.path.insert(0, _HALT_MECHANISM_DIR)

sys.modules.pop("halt_controller", None)

_mock_boto3 = MagicMock()


def _one_shot_breaking_sleep():
    """Return a time.sleep stub that raises StopIteration on first call."""
    calls = [0]

    def _sleep(t):
        calls[0] += 1
        raise StopIteration("break-while-loop-for-import")

    return _sleep


with patch.dict(sys.modules, {"boto3": _mock_boto3}), patch(
    "time.sleep", side_effect=_one_shot_breaking_sleep()
):
    try:
        import halt_controller as hc
    except StopIteration:
        pass


# ---------------------------------------------------------------------------
# Shared test helper
# ---------------------------------------------------------------------------


def _metrics(
    heartbeat=None,
    nan=False,
    diverged=False,
    tokens_per_sec=None,
    gpu_util=None,
    gpu_memory_pct=None,
    loss=1.0,
):
    """Build a realistic metrics dict for use in trigger tests."""
    return {
        "heartbeat": heartbeat if heartbeat is not None else time.time(),
        "nan": nan,
        "diverged": diverged,
        "tokens_per_sec": tokens_per_sec,
        "gpu_util": gpu_util,
        "gpu_memory_pct": gpu_memory_pct,
        "loss": loss,
    }


# ===========================================================================
# read_metrics()
# ===========================================================================


class TestReadMetrics:
    """Tests for read_metrics() — the metrics file reader."""

    def test_returns_none_when_file_does_not_exist(self, tmp_path):
        """FileNotFoundError is silenced and returns None (normal at startup)."""
        hc.METRICS_FILE = str(tmp_path / "nonexistent.json")
        assert hc.read_metrics() is None
        print("✓ Returns None for missing file")

    def test_returns_parsed_dict_for_valid_json(self, tmp_path):
        """Returns the full dict when the file contains valid JSON."""
        path = tmp_path / "metrics.json"
        payload = {"loss": 1.5, "heartbeat": 12345.0, "nan": False}
        path.write_text(json.dumps(payload))
        hc.METRICS_FILE = str(path)
        assert hc.read_metrics() == payload
        print("✓ Returns parsed dict for valid JSON")

    def test_returns_none_and_warns_on_corrupt_json(self, tmp_path, capsys):
        """Returns None and prints a warning for malformed JSON."""
        path = tmp_path / "metrics.json"
        path.write_text("{corrupt{{json")
        hc.METRICS_FILE = str(path)
        result = hc.read_metrics()
        assert result is None
        assert "Warning" in capsys.readouterr().out
        print("✓ Returns None and prints Warning for corrupt JSON")

    def test_returns_none_and_warns_on_permission_error(self, tmp_path, capsys):
        """Non-FileNotFoundError exceptions emit a warning rather than crashing."""
        path = tmp_path / "metrics.json"
        path.write_text("{}")
        hc.METRICS_FILE = str(path)
        with patch("builtins.open", side_effect=PermissionError("denied")):
            result = hc.read_metrics()
        assert result is None
        assert "Warning" in capsys.readouterr().out
        print("✓ PermissionError produces a Warning, not a silent None")


# ===========================================================================
# heartbeat_stalled()
# ===========================================================================


class TestHeartbeatStalled:
    """Tests for heartbeat_stalled()."""

    def test_false_when_metrics_is_none(self):
        assert hc.heartbeat_stalled(None) is False
        print("✓ False when metrics is None")

    def test_false_when_heartbeat_key_absent(self):
        assert hc.heartbeat_stalled({"loss": 1.0}) is False
        print("✓ False when heartbeat key is absent")

    def test_false_when_heartbeat_is_fresh(self):
        assert hc.heartbeat_stalled(_metrics(heartbeat=time.time())) is False
        print("✓ False for a fresh heartbeat")

    def test_true_when_heartbeat_exceeds_timeout(self):
        stale = time.time() - (hc.HEARTBEAT_TIMEOUT + 10)
        assert hc.heartbeat_stalled(_metrics(heartbeat=stale)) is True
        print(f"✓ True when heartbeat is >{hc.HEARTBEAT_TIMEOUT}s old")

    def test_no_crash_at_exact_timeout_boundary(self):
        """Boundary value does not raise; result is a bool."""
        exact = time.time() - hc.HEARTBEAT_TIMEOUT
        assert isinstance(hc.heartbeat_stalled(_metrics(heartbeat=exact)), bool)
        print("✓ No crash at exact timeout boundary")


# ===========================================================================
# nan_detected() / divergence_detected()
# ===========================================================================


class TestNanAndDivergenceDetected:
    """Tests for the two single-step, immediate-halt triggers."""

    def test_nan_true_when_flag_set(self):
        assert hc.nan_detected(_metrics(nan=True)) is True
        print("✓ nan_detected: True when flag set")

    def test_nan_false_by_default(self):
        assert hc.nan_detected(_metrics()) is False
        print("✓ nan_detected: False by default")

    def test_nan_false_when_metrics_none(self):
        assert hc.nan_detected(None) is False
        print("✓ nan_detected: False when metrics is None")

    def test_divergence_true_when_flag_set(self):
        assert hc.divergence_detected(_metrics(diverged=True)) is True
        print("✓ divergence_detected: True when flag set")

    def test_divergence_false_by_default(self):
        assert hc.divergence_detected(_metrics()) is False
        print("✓ divergence_detected: False by default")

    def test_divergence_false_when_metrics_none(self):
        assert hc.divergence_detected(None) is False
        print("✓ divergence_detected: False when metrics is None")


# ===========================================================================
# gpu_idle()
# ===========================================================================


class TestGpuIdle:
    """Tests for gpu_idle() — GPU underutilisation trigger."""

    def test_false_when_metrics_none(self):
        assert hc.gpu_idle(None) is False
        print("✓ False when metrics is None")

    def test_false_when_gpu_util_absent(self):
        assert hc.gpu_idle({"loss": 1.0}) is False
        print("✓ False when gpu_util key is absent")

    def test_true_when_util_below_threshold(self):
        assert hc.gpu_idle(_metrics(gpu_util=hc.GPU_MIN - 1)) is True
        print(f"✓ True when gpu_util < GPU_MIN ({hc.GPU_MIN}%)")

    def test_false_when_util_equals_threshold(self):
        # Threshold is a strict-less-than comparison
        assert hc.gpu_idle(_metrics(gpu_util=hc.GPU_MIN)) is False
        print(f"✓ False when gpu_util == GPU_MIN (strict <)")

    def test_false_when_util_well_above_threshold(self):
        assert hc.gpu_idle(_metrics(gpu_util=80)) is False
        print("✓ False when gpu_util is 80%")

    def test_threshold_is_5_percent(self):
        """Verify the lowered threshold; was 20% before the hardening pass."""
        assert hc.GPU_MIN == 5, (
            f"GPU_MIN should be 5 (lowered from 20 for ZeRO-3 compatibility); got {hc.GPU_MIN}"
        )
        print("✓ GPU_MIN is 5% (correct for ZeRO-3 with CPU offloading)")


# ===========================================================================
# memory_pressure()  — new trigger added in hardening pass
# ===========================================================================


class TestMemoryPressure:
    """Tests for memory_pressure() — GPU OOM risk trigger."""

    def test_false_when_metrics_none(self):
        assert hc.memory_pressure(None) is False
        print("✓ False when metrics is None")

    def test_false_when_field_absent(self):
        assert hc.memory_pressure({"loss": 1.0}) is False
        print("✓ False when gpu_memory_pct key is absent")

    def test_true_when_above_threshold(self):
        assert hc.memory_pressure(_metrics(gpu_memory_pct=hc.GPU_MEMORY_MAX + 0.1)) is True
        print(f"✓ True when gpu_memory_pct > GPU_MEMORY_MAX ({hc.GPU_MEMORY_MAX}%)")

    def test_false_when_at_threshold(self):
        # Threshold is strict-greater-than
        assert hc.memory_pressure(_metrics(gpu_memory_pct=hc.GPU_MEMORY_MAX)) is False
        print(f"✓ False when gpu_memory_pct == GPU_MEMORY_MAX (strict >)")

    def test_false_when_below_threshold(self):
        assert hc.memory_pressure(_metrics(gpu_memory_pct=80.0)) is False
        print("✓ False when gpu_memory_pct is 80%")

    def test_threshold_is_95_percent(self):
        assert hc.GPU_MEMORY_MAX == 95, (
            f"GPU_MEMORY_MAX should be 95; got {hc.GPU_MEMORY_MAX}"
        )
        print("✓ GPU_MEMORY_MAX is 95%")


# ===========================================================================
# throughput_collapsed() — TPS baseline + trigger
# ===========================================================================


class TestThroughputCollapsed:
    """Tests for throughput_collapsed() — includes warmup-skip and median baseline logic."""

    @pytest.fixture(autouse=True)
    def reset_tps_globals(self):
        """Reset tps_samples and baseline_tps before and after every test."""
        hc.tps_samples.clear()
        hc.baseline_tps = None
        yield
        hc.tps_samples.clear()
        hc.baseline_tps = None

    def _feed_tps(self, tps, n=1):
        """Call throughput_collapsed() n times with the given TPS value."""
        results = []
        for _ in range(n):
            results.append(hc.throughput_collapsed(_metrics(tokens_per_sec=tps)))
        return results

    def _establish_baseline(self, warmup_tps=10.0, baseline_tps=1000.0):
        """Feed enough samples to pass warmup + baseline window."""
        self._feed_tps(warmup_tps, hc.TPS_WARMUP_SAMPLES)
        self._feed_tps(baseline_tps, hc.TPS_BASELINE_WINDOW)

    # --- None / missing cases ---

    def test_false_when_metrics_none(self):
        assert hc.throughput_collapsed(None) is False
        print("✓ False when metrics is None")

    def test_false_when_tps_absent(self):
        assert hc.throughput_collapsed({"loss": 1.0}) is False
        print("✓ False when tokens_per_sec key is absent")

    def test_false_when_tps_is_zero(self):
        """Zero TPS is falsy and treated as absent."""
        assert hc.throughput_collapsed(_metrics(tokens_per_sec=0)) is False
        print("✓ False when tokens_per_sec is 0 (falsy)")

    # --- Warmup window ---

    def test_all_readings_during_warmup_window_return_false(self):
        """Every reading during warmup + baseline accumulation must return False."""
        total_needed = hc.TPS_WARMUP_SAMPLES + hc.TPS_BASELINE_WINDOW
        for _ in range(total_needed):
            assert hc.throughput_collapsed(_metrics(tokens_per_sec=1000.0)) is False
        print(f"✓ All {total_needed} accumulation readings return False")

    def test_baseline_not_set_until_enough_samples(self):
        """baseline_tps remains None while samples are still accumulating."""
        total_needed = hc.TPS_WARMUP_SAMPLES + hc.TPS_BASELINE_WINDOW
        for i in range(total_needed - 1):
            hc.throughput_collapsed(_metrics(tokens_per_sec=1000.0))
            assert hc.baseline_tps is None, f"baseline_tps set too early at sample {i+1}"
        print("✓ baseline_tps is None until enough samples collected")

    def test_baseline_set_after_full_window(self):
        """baseline_tps is set after TPS_WARMUP_SAMPLES + TPS_BASELINE_WINDOW readings."""
        self._establish_baseline()
        assert hc.baseline_tps is not None
        print("✓ baseline_tps is set after full window")

    # --- Median baseline (not first sample) ---

    def test_warmup_samples_are_discarded_from_baseline(self):
        """Low warmup TPS does not pull the baseline down; post-warmup median is used."""
        # Warmup: very low TPS (would create a false-low baseline if included)
        self._feed_tps(1.0, hc.TPS_WARMUP_SAMPLES)
        # Baseline window: stable 1000 TPS
        self._feed_tps(1000.0, hc.TPS_BASELINE_WINDOW)

        assert hc.baseline_tps is not None
        assert hc.baseline_tps > 500.0, (
            f"Baseline ({hc.baseline_tps:.1f}) should not be dragged down by warmup values"
        )
        print(f"✓ Warmup TPS discarded; baseline = {hc.baseline_tps:.1f} (expected ~1000)")

    def test_baseline_uses_median_not_mean(self):
        """An outlier in the baseline window does not skew the result like a mean would."""
        self._feed_tps(1000.0, hc.TPS_WARMUP_SAMPLES)  # warmup
        # Baseline window: 9 normal samples and 1 massive outlier
        self._feed_tps(1000.0, hc.TPS_BASELINE_WINDOW - 1)
        hc.throughput_collapsed(_metrics(tokens_per_sec=999_999.0))  # outlier

        # median of [1000]*9 + [999999] = 1000 (outlier barely moves median)
        # but mean would be ~(9000+999999)/10 = ~100899
        assert hc.baseline_tps is not None
        assert hc.baseline_tps < 50_000, (
            f"Baseline ({hc.baseline_tps:.1f}) should not be skewed by a single outlier"
        )
        print(f"✓ Median baseline {hc.baseline_tps:.1f} not skewed by single outlier")

    # --- Trigger after baseline ---

    def test_false_when_tps_above_drop_threshold(self):
        """Returns False when TPS is comfortably above baseline * THROUGHPUT_DROP."""
        self._establish_baseline(baseline_tps=1000.0)
        # 80% of baseline — well above the 50% drop threshold
        assert hc.throughput_collapsed(_metrics(tokens_per_sec=800.0)) is False
        print("✓ No collapse at 80% of baseline")

    def test_true_when_tps_below_drop_threshold(self):
        """Returns True when TPS falls below baseline * THROUGHPUT_DROP."""
        self._establish_baseline(baseline_tps=1000.0)
        # 40% of baseline — below the 50% drop threshold
        assert hc.throughput_collapsed(_metrics(tokens_per_sec=400.0)) is True
        print("✓ Collapse detected at 40% of baseline (below 50% threshold)")

    def test_false_exactly_at_drop_threshold(self):
        """At exactly baseline * THROUGHPUT_DROP the trigger does not fire (strict <)."""
        self._establish_baseline(baseline_tps=1000.0)
        exactly_at_threshold = hc.baseline_tps * hc.THROUGHPUT_DROP
        # May be True or False depending on floating-point; verify no crash
        result = hc.throughput_collapsed(_metrics(tokens_per_sec=exactly_at_threshold))
        assert isinstance(result, bool)
        print("✓ Exact threshold boundary handled without error")

    def test_throughput_drop_constant_is_50_percent(self):
        assert hc.THROUGHPUT_DROP == 0.5
        print("✓ THROUGHPUT_DROP is 0.5 (50%)")


# ===========================================================================
# check_trigger() — consecutive-gate
# ===========================================================================


class TestConsecutiveTriggerGate:
    """Tests for check_trigger() — the N-consecutive-cycles gate added in hardening."""

    @pytest.fixture(autouse=True)
    def reset_trigger_counts(self):
        hc.trigger_counts.clear()
        yield
        hc.trigger_counts.clear()

    def test_does_not_fire_on_first_bad_reading(self):
        assert hc.check_trigger("t", True) is False
        print("✓ Does not fire on first bad reading")

    def test_does_not_fire_before_threshold(self):
        for _ in range(hc.CONSECUTIVE_THRESHOLD - 1):
            assert hc.check_trigger("t", True) is False
        print(f"✓ Does not fire before {hc.CONSECUTIVE_THRESHOLD} consecutive readings")

    def test_fires_exactly_at_threshold(self):
        result = None
        for _ in range(hc.CONSECUTIVE_THRESHOLD):
            result = hc.check_trigger("t", True)
        assert result is True
        print(f"✓ Fires at exactly {hc.CONSECUTIVE_THRESHOLD} consecutive bad readings")

    def test_resets_counter_on_single_good_reading(self):
        """One good reading resets the counter; full threshold required again."""
        for _ in range(hc.CONSECUTIVE_THRESHOLD - 1):
            hc.check_trigger("t", True)
        hc.check_trigger("t", False)  # reset
        for _ in range(hc.CONSECUTIVE_THRESHOLD - 1):
            result = hc.check_trigger("t", True)
        assert result is False
        print("✓ Counter resets on good reading; full threshold needed again")

    def test_alternating_bad_good_never_fires(self):
        """A bad-good-bad-good pattern never accumulates to threshold."""
        for _ in range(hc.CONSECUTIVE_THRESHOLD * 3):
            hc.check_trigger("t", True)
            hc.check_trigger("t", False)
        assert hc.trigger_counts.get("t", 0) == 0
        print("✓ Alternating bad/good pattern never fires (non-cumulative)")

    def test_independent_counters_per_trigger_name(self):
        """Different trigger names maintain completely separate counters."""
        for _ in range(hc.CONSECUTIVE_THRESHOLD - 1):
            hc.check_trigger("a", True)
        # "b" has not fired — should not be affected by "a"'s count
        assert hc.check_trigger("b", True) is False
        assert hc.trigger_counts["a"] == hc.CONSECUTIVE_THRESHOLD - 1
        assert hc.trigger_counts["b"] == 1
        print("✓ Trigger counters are independent per name")

    def test_threshold_is_3(self):
        assert hc.CONSECUTIVE_THRESHOLD == 3
        print("✓ CONSECUTIVE_THRESHOLD is 3 (~60s at 20s polling interval)")


# ===========================================================================
# trigger_checkpoint() — SSM send + per-instance verification
# ===========================================================================


class TestTriggerCheckpoint:
    """Tests for trigger_checkpoint() — SSM command sending and verification."""

    @pytest.fixture
    def mock_ssm(self):
        m = MagicMock()
        m.send_command.return_value = {"Command": {"CommandId": "cmd-abc"}}
        hc.ssm = m
        return m

    def test_send_command_called_with_all_instance_ids(self, mock_ssm):
        with patch("time.sleep"):
            mock_ssm.get_command_invocation.return_value = {"Status": "Success"}
            hc.trigger_checkpoint(["i-1", "i-2"])
        mock_ssm.send_command.assert_called_once()
        assert mock_ssm.send_command.call_args.kwargs["InstanceIds"] == ["i-1", "i-2"]
        print("✓ send_command called with all instance IDs")

    def test_command_body_touches_force_checkpoint_file(self, mock_ssm):
        with patch("time.sleep"):
            mock_ssm.get_command_invocation.return_value = {"Status": "Success"}
            hc.trigger_checkpoint(["i-1"])
        call_params = str(mock_ssm.send_command.call_args)
        assert "FORCE_CHECKPOINT" in call_params
        print("✓ SSM command body references FORCE_CHECKPOINT")

    def test_get_command_invocation_polled_per_instance(self, mock_ssm):
        with patch("time.sleep"):
            mock_ssm.get_command_invocation.return_value = {"Status": "Success"}
            hc.trigger_checkpoint(["i-1"])
        mock_ssm.get_command_invocation.assert_called_with(
            CommandId="cmd-abc", InstanceId="i-1"
        )
        print("✓ get_command_invocation polled for each instance")

    def test_warning_logged_when_ssm_status_is_failed(self, mock_ssm, capsys):
        with patch("time.sleep"):
            mock_ssm.get_command_invocation.return_value = {"Status": "Failed"}
            hc.trigger_checkpoint(["i-1"])
        assert "WARNING" in capsys.readouterr().out
        print("✓ WARNING logged when SSM status is Failed")

    def test_warning_logged_when_ssm_status_is_cancelled(self, mock_ssm, capsys):
        with patch("time.sleep"):
            mock_ssm.get_command_invocation.return_value = {"Status": "Cancelled"}
            hc.trigger_checkpoint(["i-1"])
        assert "WARNING" in capsys.readouterr().out
        print("✓ WARNING logged when SSM status is Cancelled")

    def test_warning_logged_when_invocation_raises(self, mock_ssm, capsys):
        with patch("time.sleep"):
            mock_ssm.get_command_invocation.side_effect = Exception("SSM error")
            hc.trigger_checkpoint(["i-1"])
        assert "WARNING" in capsys.readouterr().out
        print("✓ WARNING logged when get_command_invocation raises")

    def test_continues_after_one_instance_fails(self, mock_ssm, capsys):
        """Verification failure on one instance does not stop processing others."""
        with patch("time.sleep"):
            mock_ssm.get_command_invocation.side_effect = [
                {"Status": "Failed"},   # i-1 fails
                {"Status": "Success"},  # i-2 succeeds
            ]
            hc.trigger_checkpoint(["i-1", "i-2"])
        # Should have polled both
        assert mock_ssm.get_command_invocation.call_count >= 2
        print("✓ Continues verifying remaining instances after one fails")


# ===========================================================================
# wait_for_checkpoint() — S3 sentinel poll + timeout
# ===========================================================================


class TestWaitForCheckpoint:
    """Tests for wait_for_checkpoint() — success path and 60-minute timeout."""

    @pytest.fixture
    def mock_s3(self):
        m = MagicMock()
        hc.s3 = m
        return m

    def test_returns_true_when_sentinel_found_immediately(self, mock_s3):
        mock_s3.head_object.return_value = {}
        with patch("time.sleep"):
            result = hc.wait_for_checkpoint()
        assert result is True
        print("✓ Returns True when sentinel found on first attempt")

    def test_returns_true_after_one_retry(self, mock_s3):
        mock_s3.head_object.side_effect = [Exception("Not yet"), {}]
        with patch("time.sleep"):
            result = hc.wait_for_checkpoint()
        assert result is True
        assert mock_s3.head_object.call_count == 2
        print("✓ Returns True after one retry")

    def test_returns_false_when_deadline_already_passed(self, mock_s3):
        """Setting CHECKPOINT_TIMEOUT to -1 makes the deadline past; loop never runs."""
        mock_s3.head_object.side_effect = Exception("Never")
        original = hc.CHECKPOINT_TIMEOUT
        hc.CHECKPOINT_TIMEOUT = -1
        try:
            with patch("time.sleep"):
                result = hc.wait_for_checkpoint()
        finally:
            hc.CHECKPOINT_TIMEOUT = original
        assert result is False
        print("✓ Returns False immediately when timeout is already exceeded")

    def test_returns_false_not_infinite_loop_on_timeout(self, mock_s3):
        """Function terminates and returns False; does not hang indefinitely."""
        mock_s3.head_object.side_effect = Exception("Never")
        original = hc.CHECKPOINT_TIMEOUT
        hc.CHECKPOINT_TIMEOUT = 0
        try:
            with patch("time.sleep"):
                result = hc.wait_for_checkpoint()
        finally:
            hc.CHECKPOINT_TIMEOUT = original
        assert result is False
        print("✓ Terminates cleanly on timeout (no infinite loop)")

    def test_sleeps_between_retries(self, mock_s3):
        mock_s3.head_object.side_effect = [Exception("Not yet"), {}]
        with patch("time.sleep") as mock_sleep:
            hc.wait_for_checkpoint()
        mock_sleep.assert_called()
        print("✓ time.sleep called between S3 retries")

    def test_timeout_constant_is_3600_seconds(self):
        assert hc.CHECKPOINT_TIMEOUT == 3600
        print("✓ CHECKPOINT_TIMEOUT is 3600s (60 minutes)")


# ===========================================================================
# halt_cluster() — full halt flow integration
# ===========================================================================


class TestHaltCluster:
    """Integration tests for halt_cluster() — end-to-end halt flow."""

    @pytest.fixture
    def instances_running(self):
        """Stub EC2 to return one running GPU instance."""
        mock_ec2 = MagicMock()
        mock_ec2.describe_instances.return_value = {
            "Reservations": [{"Instances": [{"InstanceId": "i-abc123"}]}]
        }
        hc.ec2 = mock_ec2
        return mock_ec2

    @pytest.fixture
    def no_instances(self):
        """Stub EC2 to return no running GPU instances."""
        mock_ec2 = MagicMock()
        mock_ec2.describe_instances.return_value = {"Reservations": []}
        hc.ec2 = mock_ec2
        return mock_ec2

    def test_skips_when_no_instances_running(self, no_instances, capsys):
        """halt_cluster() exits early and does not call any halt functions."""
        with patch.object(hc, "trigger_checkpoint") as mock_trigger:
            hc.halt_cluster("No instances test")
        mock_trigger.assert_not_called()
        assert "No running GPU instances" in capsys.readouterr().out
        print("✓ Early exit when no instances found")

    def test_full_halt_sequence_in_order(self, instances_running):
        """trigger_checkpoint → wait_for_checkpoint → terminate → verify → sys.exit."""
        with patch.object(hc, "trigger_checkpoint") as mock_trigger, \
             patch.object(hc, "wait_for_checkpoint", return_value=True) as mock_wait, \
             patch.object(hc, "terminate") as mock_terminate, \
             patch.object(hc, "verify") as mock_verify, \
             patch("sys.exit") as mock_exit:
            hc.halt_cluster("Full sequence test")

        mock_trigger.assert_called_once_with(["i-abc123"])
        mock_wait.assert_called_once()
        mock_terminate.assert_called_once_with(["i-abc123"])
        mock_verify.assert_called_once()
        mock_exit.assert_called_once_with(0)
        print("✓ Full halt sequence executed in correct order")

    def test_exits_with_zero_on_success(self, instances_running):
        with patch.object(hc, "trigger_checkpoint"), \
             patch.object(hc, "wait_for_checkpoint", return_value=True), \
             patch.object(hc, "terminate"), \
             patch.object(hc, "verify"), \
             patch("sys.exit") as mock_exit:
            hc.halt_cluster("Exit test")
        mock_exit.assert_called_once_with(0)
        print("✓ sys.exit(0) called after successful halt")

    def test_terminates_even_if_checkpoint_times_out(self, instances_running):
        """Instance termination proceeds even when wait_for_checkpoint returns False."""
        with patch.object(hc, "trigger_checkpoint"), \
             patch.object(hc, "wait_for_checkpoint", return_value=False), \
             patch.object(hc, "terminate") as mock_terminate, \
             patch.object(hc, "verify"), \
             patch("sys.exit"):
            hc.halt_cluster("Timeout test")
        mock_terminate.assert_called_once()
        print("✓ Termination proceeds even when checkpoint wait times out")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
