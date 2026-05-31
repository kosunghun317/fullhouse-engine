"""Tests for adaptive-threshold benchmark reporting helpers."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tools.benchmark_adaptive_threshold_rollout import render_results_table


def test_render_results_table_groups_by_suite_and_ranks_by_mean():
    rows = [
        {
            "candidate": "slow",
            "suite": "heads_up_shark",
            "matches": 4,
            "hands": 80,
            "mean_delta": 100.0,
            "median_delta": 50.0,
            "p10_delta": -25,
            "ci95": 12.5,
            "positive_rate": 0.5,
            "bust_rate": 0.25,
            "bot_error_count": 0,
        },
        {
            "candidate": "fast",
            "suite": "heads_up_shark",
            "matches": 4,
            "hands": 80,
            "mean_delta": 250.0,
            "median_delta": 200.0,
            "p10_delta": 100,
            "ci95": 10.0,
            "positive_rate": 0.75,
            "bust_rate": 0.0,
            "bot_error_count": 1,
        },
    ]

    table = render_results_table(rows)

    assert "| suite" in table
    assert "| candidate" in table
    assert "75.0%" in table
    assert "25.0%" in table
    assert table.index("fast") < table.index("slow")
