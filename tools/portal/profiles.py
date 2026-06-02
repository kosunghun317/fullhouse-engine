"""Replay-derived opponent profile construction."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from statistics import median

from .analysis import build_report

PROFILE_DESCRIPTIONS = {
    "large_size_jammer": "Large sizing or all-in frequency is high enough to require tighter call-off rules.",
    "loose_passive_overfolder": "Loose entry and passive calling, but folds often when facing pressure.",
    "pressure_overfolder": "Aggressive preflop and postflop, then folds often when re-pressured.",
    "tight_overfolder": "Tight range construction with a high fold rate against pressure.",
    "sticky_station": "High call or showdown frequency; value-heavy counterstrategy is preferred.",
    "balanced_aggressor": "Meaningful initiative without the most extreme pressure-fold leak.",
    "passive_caller": "Low initiative with enough calling to punish careless bluffs.",
    "unknown_mixed": "Mixed or low-signal behavior that does not match a stronger heuristic profile.",
}

NUMERIC_FIELDS = (
    "actions",
    "all_in_rate",
    "avg_raise_to_pot",
    "call_rate",
    "check_rate",
    "fold_rate",
    "hands_alive",
    "official_delta",
    "official_rank",
    "pfr",
    "preflop_all_in_per_hand",
    "pressure_call_rate",
    "pressure_fold_rate",
    "raise_rate",
    "sample_delta",
    "showdown_rate",
    "vpip",
    "win_hand_rate",
)


def _float(row: dict, field: str) -> float:
    value = row.get(field, 0.0)
    if isinstance(value, bool):
        return 0.0
    if isinstance(value, (int, float)):
        return float(value)
    return 0.0


def _round(value: float | None, places: int = 4) -> float | None:
    return round(value, places) if value is not None else None


def clamp(value: float, lo: float, hi: float) -> float:
    return min(hi, max(lo, value))


def percentile(values: list[float], pct: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    rank = (len(ordered) - 1) * pct
    lo = int(rank)
    hi = min(lo + 1, len(ordered) - 1)
    frac = rank - lo
    return ordered[lo] * (1.0 - frac) + ordered[hi] * frac


def assign_profile(row: dict) -> str:
    """Assign a deterministic behavior profile from portal strategy metrics."""

    hands = _float(row, "hands_alive")
    if hands < 100:
        return "unknown_mixed"

    vpip = _float(row, "vpip")
    pfr = _float(row, "pfr")
    raise_rate = _float(row, "raise_rate")
    call_rate = _float(row, "call_rate")
    pressure_fold = _float(row, "pressure_fold_rate")
    pressure_call = _float(row, "pressure_call_rate")
    showdown = _float(row, "showdown_rate")
    all_in = _float(row, "all_in_rate")
    preflop_all_in = _float(row, "preflop_all_in_per_hand")
    raise_to_pot = _float(row, "avg_raise_to_pot")

    if all_in >= 0.012 or preflop_all_in >= 0.012 or raise_to_pot >= 5.0:
        return "large_size_jammer"
    if vpip >= 0.30 and pfr <= 0.18 and call_rate >= 0.16 and pressure_fold >= 0.72:
        return "loose_passive_overfolder"
    if (pfr >= 0.22 or raise_rate >= 0.22) and pressure_fold >= 0.70 and showdown <= 0.13:
        return "pressure_overfolder"
    if vpip <= 0.24 and pfr <= 0.20 and pressure_fold >= 0.75:
        return "tight_overfolder"
    if call_rate >= 0.18 or pressure_call >= 0.34 or showdown >= 0.14:
        return "sticky_station"
    if pfr >= 0.22 or raise_rate >= 0.22:
        return "balanced_aggressor"
    if pfr <= 0.16 and call_rate >= 0.13:
        return "passive_caller"
    return "unknown_mixed"


def summarize_numeric(rows: list[dict], field: str) -> dict:
    values = [_float(row, field) for row in rows if row.get(field) is not None]
    if not values:
        return {"mean": None, "p10": None, "p50": None, "p90": None}
    return {
        "mean": _round(sum(values) / len(values)),
        "p10": _round(percentile(values, 0.10)),
        "p50": _round(float(median(values))),
        "p90": _round(percentile(values, 0.90)),
    }


def profile_mock_config(profile: str, rows: list[dict]) -> dict:
    metrics = {field: summarize_numeric(rows, field)["mean"] for field in NUMERIC_FIELDS}
    vpip = float(metrics.get("vpip") or 0.25)
    pfr = float(metrics.get("pfr") or 0.14)
    raise_rate = float(metrics.get("raise_rate") or 0.12)
    call_rate = float(metrics.get("call_rate") or 0.12)
    pressure_fold = float(metrics.get("pressure_fold_rate") or 0.65)
    showdown = float(metrics.get("showdown_rate") or 0.09)
    all_in = float(metrics.get("all_in_rate") or 0.002)
    raise_to_pot = float(metrics.get("avg_raise_to_pot") or 1.0)

    aggression_bias = clamp((pfr + raise_rate) - call_rate, -0.30, 0.45)
    bluff_bias = clamp((pressure_fold - 0.62) + aggression_bias * 0.35, -0.35, 0.45)
    value_bias = clamp(call_rate + showdown - pfr * 0.35, -0.10, 0.40)

    return {
        "profile": profile,
        "description": PROFILE_DESCRIPTIONS[profile],
        "vpip": _round(clamp(vpip, 0.06, 0.65)),
        "pfr": _round(clamp(pfr, 0.01, 0.55)),
        "raise_rate": _round(clamp(raise_rate, 0.01, 0.55)),
        "call_rate": _round(clamp(call_rate, 0.02, 0.45)),
        "pressure_fold_rate": _round(clamp(pressure_fold, 0.05, 0.95)),
        "showdown_rate": _round(clamp(showdown, 0.01, 0.35)),
        "all_in_rate": _round(clamp(all_in, 0.0, 0.08)),
        "avg_raise_to_pot": _round(clamp(raise_to_pot, 0.25, 8.0)),
        "aggression_bias": _round(aggression_bias),
        "bluff_bias": _round(bluff_bias),
        "value_bias": _round(value_bias),
    }


def profile_summary(profile: str, rows: list[dict]) -> dict:
    ordered = sorted(
        rows,
        key=lambda row: (
            row.get("official_rank") is None,
            row.get("official_rank") if row.get("official_rank") is not None else 10**9,
            -_float(row, "sample_delta"),
        ),
    )
    return {
        "profile": profile,
        "description": PROFILE_DESCRIPTIONS[profile],
        "bot_count": len(rows),
        "bot_ids": [row["bot_id"] for row in ordered],
        "bot_names": [row["bot_name"] for row in ordered],
        "top_official_rank": ordered[0].get("official_rank") if ordered else None,
        "official_delta_mean": summarize_numeric(rows, "official_delta")["mean"],
        "metrics": {field: summarize_numeric(rows, field) for field in NUMERIC_FIELDS},
        "mock_config": profile_mock_config(profile, rows),
    }


def load_or_build_report(directory: Path, report_json: Path | None, top: int) -> dict:
    if report_json and report_json.exists():
        return json.loads(report_json.read_text(encoding="utf-8"))
    default_report = directory / "strategy_report.json"
    if default_report.exists():
        return json.loads(default_report.read_text(encoding="utf-8"))
    return build_report(directory, top)


def build_profiles(
    directory: Path,
    report_json: Path | None = None,
    min_hands: int = 200,
    top: int = 100_000,
) -> dict:
    report = load_or_build_report(directory, report_json, top)
    bots = [row for row in report.get("bots", []) if _float(row, "hands_alive") >= min_hands]
    grouped: dict[str, list[dict]] = defaultdict(list)
    bot_rows = []
    for row in bots:
        profile = assign_profile(row)
        grouped[profile].append(row)
        bot_rows.append({
            "bot_id": row.get("bot_id"),
            "bot_name": row.get("bot_name"),
            "profile": profile,
            "official_rank": row.get("official_rank"),
            "official_delta": row.get("official_delta"),
            "hands_alive": row.get("hands_alive"),
            "vpip": row.get("vpip"),
            "pfr": row.get("pfr"),
            "call_rate": row.get("call_rate"),
            "pressure_fold_rate": row.get("pressure_fold_rate"),
            "showdown_rate": row.get("showdown_rate"),
        })

    profiles = [
        profile_summary(profile, rows)
        for profile, rows in sorted(grouped.items(), key=lambda item: (-len(item[1]), item[0]))
    ]
    return {
        "source_directory": str(directory),
        "source_report": str(report_json or directory / "strategy_report.json"),
        "min_hands": min_hands,
        "input_bot_count": len(report.get("bots", [])),
        "profiled_bot_count": len(bot_rows),
        "profile_count": len(profiles),
        "profiles": profiles,
        "bots": sorted(
            bot_rows,
            key=lambda row: (
                row.get("official_rank") is None,
                row.get("official_rank") if row.get("official_rank") is not None else 10**9,
                row.get("bot_name") or "",
            ),
        ),
    }


def write_profile_artifacts(summary: dict, output: Path) -> None:
    output.mkdir(parents=True, exist_ok=True)
    (output / "profile_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    for profile in summary["profiles"]:
        profile_dir = output / profile["profile"]
        profile_dir.mkdir(parents=True, exist_ok=True)
        (profile_dir / "profile.json").write_text(
            json.dumps(profile, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        (profile_dir / "mock_config.json").write_text(
            json.dumps(profile["mock_config"], indent=2, sort_keys=True),
            encoding="utf-8",
        )
