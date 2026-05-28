from types import SimpleNamespace
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tools.paired_heuristic_gate import _summarize_pair


def args(**overrides):
    defaults = {
        "min_mean_diff": 200.0,
        "min_mean_ci_low": 0.0,
        "min_median_diff": 0.0,
        "min_p10_diff": -1000.0,
        "min_win_rate": 0.55,
        "min_paired_runs": 4,
        "max_extra_busts": 0,
        "bootstrap_samples": 1000,
        "bootstrap_seed": 17,
        "ci_level": 0.95,
    }
    defaults.update(overrides)
    return SimpleNamespace(**defaults)


def rows(deltas, *, bust_at=None, errors_at=None):
    bust_at = set(bust_at or [])
    errors_at = set(errors_at or [])
    return {
        ("suite", index): {
            "suite": "suite",
            "seed": index,
            "delta": delta,
            "busted": index in bust_at,
            "error_count": 1 if index in errors_at else 0,
            "errors": ["error"] if index in errors_at else [],
        }
        for index, delta in enumerate(deltas)
    }


def test_pair_gate_promotes_only_when_ci_lower_bound_is_positive():
    incumbent = rows([0, 0, 0, 0, 0, 0])
    candidate = rows([500, 700, 650, 800, 900, 600])

    summary = _summarize_pair(incumbent, candidate, args())

    assert summary["mean_diff_ci"][0] > 0
    assert summary["promotable"] is True


def test_pair_gate_rejects_noisy_positive_mean_without_positive_ci():
    incumbent = rows([0, 0, 0, 0])
    candidate = rows([-1000, 600, 600, 600])

    summary = _summarize_pair(
        incumbent,
        candidate,
        args(min_mean_diff=100.0, min_p10_diff=-1000.0),
    )

    assert summary["mean_diff"] > 0
    assert summary["median_diff"] > 0
    assert summary["win_rate"] > 0.55
    assert summary["mean_diff_ci"][0] < 0
    assert summary["promotable"] is False


def test_pair_gate_rejects_smoke_sample_below_minimum_runs():
    incumbent = rows([0])
    candidate = rows([5000])

    summary = _summarize_pair(incumbent, candidate, args(min_paired_runs=2))

    assert summary["mean_diff_ci"][0] > 0
    assert summary["promotable"] is False
