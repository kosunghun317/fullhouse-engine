"""Unit tests for named heuristic threshold tuning guardrails."""

from __future__ import annotations

import os
import sys
from types import SimpleNamespace

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from tools.select_heuristic_config import validate_budget as validate_select_budget
from tools.tune_heuristic_thresholds import validate_budget


def test_threshold_tuning_rejects_tiny_non_smoke_budget():
    args = SimpleNamespace(
        allow_smoke=False,
        hands=20,
        min_tasks_per_config=512,
    )

    with pytest.raises(SystemExit):
        validate_budget(args, ["reference_6max"], [101, 202])


def test_threshold_tuning_allows_explicit_smoke_budget():
    args = SimpleNamespace(
        allow_smoke=True,
        hands=20,
        min_tasks_per_config=512,
    )

    validate_budget(args, ["reference_6max"], [101])


def test_select_config_rejects_tiny_non_smoke_promotion_budget():
    args = SimpleNamespace(
        allow_smoke=False,
        preset="promotion",
        hands=400,
        min_tasks_per_config=512,
    )

    with pytest.raises(SystemExit):
        validate_select_budget(args, ["reference_6max"], [101, 202])
