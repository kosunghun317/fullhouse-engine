"""Tests for submission heuristic benchmark helper."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pytest

from tools.benchmark_submission_heuristic import build_candidates, parse_candidate


def test_parse_candidate_requires_name_and_path():
    assert parse_candidate("rollout=bots/rollout_search_submission") == (
        "rollout",
        "bots/rollout_search_submission",
    )

    with pytest.raises(ValueError):
        parse_candidate("rollout")

    with pytest.raises(ValueError):
        parse_candidate("=bots/rollout_search_submission")


def test_build_candidates_defaults_to_submission_zip_and_adds_extras():
    candidates = build_candidates(
        bot_path="dist/heuristic_bot.zip",
        candidate_name="heuristic_submission",
        extra_candidates=[
            "rollout=bots/rollout_search_submission",
            "adaptive=bots/adaptive_threshold_rollout_submission",
        ],
    )

    assert candidates == {
        "heuristic_submission": "dist/heuristic_bot.zip",
        "rollout": "bots/rollout_search_submission",
        "adaptive": "bots/adaptive_threshold_rollout_submission",
    }
