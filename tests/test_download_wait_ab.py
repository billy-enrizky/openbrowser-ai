from pathlib import Path

import pytest

from benchmarks.interactive import download_wait_ab
from benchmarks.interactive.adapters import openbrowser


def test_download_wait_benchmark_accepts_a_short_tmp_directory():
    assert download_wait_ab.validate_download_wait_ab_workdir(Path("/tmp/ob")) == Path("/tmp/ob").resolve()


def test_download_wait_benchmark_rejects_a_long_socket_path():
    workdir = Path("/tmp") / ("x" * 90)

    with pytest.raises(ValueError, match="shorter path"):
        download_wait_ab.validate_download_wait_ab_workdir(workdir)


def test_download_wait_benchmark_rejects_a_non_tmp_directory():
    with pytest.raises(ValueError, match="under /tmp"):
        download_wait_ab.validate_download_wait_ab_workdir(Path("/var/tmp/openbrowser-bench"))


def test_download_wait_benchmark_restores_the_adapter_builder():
    original = openbrowser._fast_download

    with download_wait_ab.instrument_download_wait_ab_timing():
        program = openbrowser._fast_download("http://127.0.0.1/download")

    assert "click_to_terminal_ms" in program
    assert "terminal_to_wait_return_ms" in program
    assert openbrowser._fast_download is original
