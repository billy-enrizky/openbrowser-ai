"""Tests for the scraping resource lane."""

from __future__ import annotations

import time

from benchmarks.scraping.resource_compare import ResourceSampler, _group


def test_resource_sampler_reports_peak_rss_and_cpu():
    sampler = ResourceSampler(interval=0.01)
    sampler.start()
    block = bytearray(8 * 1024 * 1024)
    time.sleep(0.05)
    metrics = sampler.stop()

    assert block
    assert metrics["peak_rss_mb"] > 0
    assert metrics["live_tree_cpu_ms"] >= 0
    assert metrics["reaped_child_cpu_ms"] >= 0
    assert metrics["child_maxrss_mb"] >= 0


def test_resource_group_summarizes_peak_and_cpu():
    records = [
        {
            "task": "table",
            "arm": "http",
            "ok": True,
            "latency_ms": 1.0,
            "peak_rss_mb": 10.0,
            "live_tree_cpu_ms": 2.0,
            "reaped_child_cpu_ms": 1.0,
            "child_maxrss_mb": 4.0,
            "requests": 1,
            "response_bytes": 100,
            "detail": "",
        },
        {
            "task": "table",
            "arm": "http",
            "ok": True,
            "latency_ms": 3.0,
            "peak_rss_mb": 20.0,
            "live_tree_cpu_ms": 4.0,
            "reaped_child_cpu_ms": 3.0,
            "child_maxrss_mb": 6.0,
            "requests": 1,
            "response_bytes": 100,
            "detail": "",
        },
    ]

    groups = _group(records)

    assert groups[0]["ok"] is True
    assert groups[0]["peak_rss_mb_max"] == 20.0
    assert groups[0]["peak_rss_mb_p50"] == 15.0
    assert groups[0]["live_tree_cpu_ms_p50"] == 3.0
    assert groups[0]["child_maxrss_mb_max"] == 6.0
