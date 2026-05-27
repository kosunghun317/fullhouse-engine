"""Parity tests for the unlimited offline fast match runner."""

from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from sandbox.match import run_match
from tools.parallel import resolve_workers
from training.fast_match import run_fast_match, run_fast_match_task, run_fast_matches_parallel


ROOT = Path(__file__).resolve().parents[1]
FAST_BOTS = ROOT / "tests" / "fast_bots"


def _bot_paths() -> dict[str, str]:
    return {
        "call_check": str(FAST_BOTS / "call_check"),
        "fold_check": str(FAST_BOTS / "fold_check"),
        "min_raise": str(FAST_BOTS / "min_raise"),
    }


def _compact_hands(result: dict) -> list[dict]:
    return [
        {
            "hand_num": hand["hand_num"],
            "community_cards": hand["community_cards"],
            "winners": hand["winners"],
            "showdown": hand["showdown"],
            "final_stacks": hand["final_stacks"],
        }
        for hand in result["hands"]
    ]


def test_fast_match_matches_sandbox_for_deterministic_bots():
    bot_paths = _bot_paths()
    official = run_match("parity", bot_paths, n_hands=12, seed=101)
    fast = run_fast_match("parity", bot_paths, n_hands=12, seed=101)

    assert fast["n_hands"] == official["n_hands"]
    assert fast["final_stacks"] == official["final_stacks"]
    assert fast["chip_delta"] == official["chip_delta"]
    assert fast["bot_errors"] == official["bot_errors"]
    assert _compact_hands(fast) == _compact_hands(official)


def test_fast_match_parallel_matches_sequential():
    tasks = [
        {
            "match_id": f"parallel_{seed}",
            "bot_paths": _bot_paths(),
            "n_hands": 6,
            "seed": seed,
        }
        for seed in (31, 32, 33)
    ]
    sequential = [run_fast_match_task(task) for task in tasks]
    parallel = run_fast_matches_parallel(tasks, workers=0, backend="process")

    assert [result["final_stacks"] for result in parallel] == [
        result["final_stacks"] for result in sequential
    ]
    assert [result["chip_delta"] for result in parallel] == [
        result["chip_delta"] for result in sequential
    ]


def test_workers_zero_auto_selects_parallelism():
    assert resolve_workers(0, 1) == 1
    assert 1 <= resolve_workers(0, 3) <= 3


def test_fast_match_has_no_action_timeout():
    bot_paths = {
        "sleepy": str(FAST_BOTS / "sleepy"),
        "call_check": str(FAST_BOTS / "call_check"),
    }
    result = run_fast_match("sleepy", bot_paths, n_hands=2, seed=202)

    assert result["n_hands"] == 2
    assert result["bot_errors"] == {"sleepy": [], "call_check": []}
    assert sum(result["final_stacks"].values()) == 20000
