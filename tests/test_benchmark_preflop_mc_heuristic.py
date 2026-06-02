"""Tests for the focused preflop-MC benchmark wrapper."""

from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tools.benchmark_preflop_mc_heuristic import (
    DEFAULT_BASELINE_NAME,
    DEFAULT_BASELINE_PATH,
    DEFAULT_SUITES,
    default_suite_ids,
    maybe_add_baseline,
)
from tools.benchmark_adaptive_threshold_rollout import SUITES


def test_default_suite_ids_are_focused_matrix():
    assert default_suite_ids() == DEFAULT_SUITES
    assert default_suite_ids(all_suites=True) == list(SUITES)


def test_maybe_add_baseline_preserves_candidate_order_after_baseline():
    candidates = {"preflop": "runs/preflop_mc_heuristic/preflop-mc-4096-1sec/bot"}

    assert maybe_add_baseline(candidates, include_baseline=False) is candidates
    assert maybe_add_baseline(candidates, include_baseline=True) == {
        DEFAULT_BASELINE_NAME: DEFAULT_BASELINE_PATH,
        "preflop": "runs/preflop_mc_heuristic/preflop-mc-4096-1sec/bot",
    }
