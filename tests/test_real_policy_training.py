"""Unit tests for real-policy training guardrails."""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from tools.strong_mocks.train_real_policy import should_export_policy


def test_should_export_policy_allows_new_outputs_below_threshold():
    assert should_export_policy(
        selected_mean_delta=-500.0,
        output_exists=False,
        min_export_mean_delta=1500.0,
    )


def test_should_export_policy_blocks_existing_output_below_threshold():
    assert not should_export_policy(
        selected_mean_delta=1100.0,
        output_exists=True,
        min_export_mean_delta=1500.0,
    )


def test_should_export_policy_allows_existing_output_above_threshold():
    assert should_export_policy(
        selected_mean_delta=1600.0,
        output_exists=True,
        min_export_mean_delta=1500.0,
    )
