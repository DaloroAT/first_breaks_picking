"""Tests for SGY write functionality."""

from pathlib import Path

import numpy as np
import pytest

from first_breaks.sgy.reader import SGY
from first_breaks.utils.utils import calc_hash


class TestSGYWriter:
    """Test SGY write functionality."""

    def test_roundtrip_ieee(self, logs_dir_for_tests: Path) -> None:
        """Test read->write->read roundtrip for IEEE float format."""
        # Use the ieee benchmark file
        source_path = Path("/workspace/data_pack/benchmark/ieee")
        if not source_path.exists():
            pytest.skip("IEEE benchmark file not found")

        # Read the original file
        sgy_original = SGY(source_path)
        traces_original = sgy_original.read()
        file_headers_original = sgy_original.general_headers.copy()
        traces_headers_original = sgy_original.traces_headers.copy()

        # Create new SGY from numpy array and headers
        sgy_new = SGY(
            traces_original,
            dt_mcs=sgy_original.dt_mcs,
            file_headers=file_headers_original,
            traces_headers=traces_headers_original,
        )

        # Write to new file
        output_path = logs_dir_for_tests / "roundtrip_ieee.sgy"
        sgy_new.write(output_path, data_format=5)  # IEEE float

        # Read back and compare
        sgy_roundtrip = SGY(output_path)
        traces_roundtrip = sgy_roundtrip.read()

        # Compare traces
        assert traces_original.shape == traces_roundtrip.shape
        assert np.allclose(traces_original, traces_roundtrip, rtol=1e-6)

        # Compare file headers (essential ones)
        assert sgy_original.dt_mcs == sgy_roundtrip.dt_mcs
        assert sgy_original.num_samples == sgy_roundtrip.num_samples
        assert sgy_original.num_traces == sgy_roundtrip.num_traces

        # Compare file hashes (should be identical for IEEE)
        hash_original = calc_hash(source_path)
        hash_roundtrip = calc_hash(output_path)
        assert hash_original == hash_roundtrip, "File hashes should match for IEEE format"

    def test_roundtrip_ibm(self, logs_dir_for_tests: Path) -> None:
        """Test read->write->read roundtrip for IBM float format."""
        # Use the ibm32 benchmark file
        source_path = Path("/workspace/data_pack/benchmark/ibm32")
        if not source_path.exists():
            pytest.skip("IBM32 benchmark file not found")

        # Read the original file
        sgy_original = SGY(source_path)
        traces_original = sgy_original.read()
        file_headers_original = sgy_original.general_headers.copy()
        traces_headers_original = sgy_original.traces_headers.copy()

        # Create new SGY from numpy array and headers
        sgy_new = SGY(
            traces_original,
            dt_mcs=sgy_original.dt_mcs,
            file_headers=file_headers_original,
            traces_headers=traces_headers_original,
        )

        # Write to new file
        output_path = logs_dir_for_tests / "roundtrip_ibm.sgy"
        sgy_new.write(output_path, data_format=1)  # IBM float

        # Read back and compare
        sgy_roundtrip = SGY(output_path)
        traces_roundtrip = sgy_roundtrip.read()

        # Compare traces
        assert traces_original.shape == traces_roundtrip.shape
        assert np.allclose(traces_original, traces_roundtrip, rtol=1e-6)

        # Compare file headers (essential ones)
        assert sgy_original.dt_mcs == sgy_roundtrip.dt_mcs
        assert sgy_original.num_samples == sgy_roundtrip.num_samples
        assert sgy_original.num_traces == sgy_roundtrip.num_traces

        # Compare file hashes (should be identical for IBM)
        hash_original = calc_hash(source_path)
        hash_roundtrip = calc_hash(output_path)
        assert hash_original == hash_roundtrip, "File hashes should match for IBM format"

    def test_write_with_default_headers(self, logs_dir_for_tests: Path) -> None:
        """Test writing SGY with auto-generated default headers."""
        # Create synthetic data
        np.random.seed(42)
        num_samples = 1000
        num_traces = 50
        traces = np.random.randn(num_samples, num_traces).astype(np.float32)
        dt_mcs = 1000  # 1 ms

        # Create SGY without providing headers
        sgy = SGY(traces, dt_mcs=dt_mcs)

        # Write to file
        output_path = logs_dir_for_tests / "default_headers.sgy"
        sgy.write(output_path, data_format=5)

        # Read back
        sgy_read = SGY(output_path)
        traces_read = sgy_read.read()

        # Verify
        assert sgy_read.num_samples == num_samples
        assert sgy_read.num_traces == num_traces
        assert sgy_read.dt_mcs == dt_mcs
        assert np.allclose(traces, traces_read, rtol=1e-6)

    def test_write_preserves_trace_headers(self, logs_dir_for_tests: Path) -> None:
        """Test that trace headers are preserved through write/read cycle."""
        source_path = Path("/workspace/data_pack/benchmark/ieee")
        if not source_path.exists():
            pytest.skip("IEEE benchmark file not found")

        # Read original
        sgy_original = SGY(source_path)
        traces = sgy_original.read()
        headers_original = sgy_original.traces_headers.copy()

        # Create new SGY and write
        sgy_new = SGY(
            traces,
            dt_mcs=sgy_original.dt_mcs,
            file_headers=sgy_original.general_headers.copy(),
            traces_headers=headers_original,
        )
        output_path = logs_dir_for_tests / "preserved_headers.sgy"
        sgy_new.write(output_path, data_format=5)

        # Read back and compare headers
        sgy_read = SGY(output_path)
        _ = sgy_read.read()
        headers_read = sgy_read.traces_headers

        # Compare key headers (non-scalar ones for simplicity)
        for col in ["TRACENO", "FFID", "CHAN"]:
            if col in headers_original.columns and col in headers_read.columns:
                assert np.allclose(
                    headers_original[col].values,
                    headers_read[col].values,
                    rtol=1e-6,
                ), f"Header {col} mismatch"

    def test_dt_mcs_validation(self) -> None:
        """Test that dt_mcs must match file header if provided."""
        traces = np.random.randn(100, 10).astype(np.float32)
        dt_mcs = 1000

        # File headers with different dt
        file_headers = {"dt": 2000, "ns": 100, "data_sample_format": 5}

        with pytest.raises(Exception):  # SGYInitParamsError
            SGY(traces, dt_mcs=dt_mcs, file_headers=file_headers)
