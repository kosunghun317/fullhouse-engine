"""Bounded tunable constants for heuristic expert-arm selection."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import numpy as np


@dataclass(frozen=True)
class ParamSpec:
    name: str
    default: float
    lower: float
    upper: float

    def clip(self, value: float) -> float:
        return float(max(self.lower, min(self.upper, value)))

    def normalize(self, value: float) -> float:
        if self.upper <= self.lower:
            return 0.0
        return (self.clip(value) - self.lower) / (self.upper - self.lower)

    def denormalize(self, value: float) -> float:
        return self.clip(self.lower + float(value) * (self.upper - self.lower))


PARAM_SPECS: tuple[ParamSpec, ...] = (
    ParamSpec("classify_min_actions", 7.0, 4.0, 24.0),
    ParamSpec("maniac_allin_rate", 0.06, 0.02, 0.16),
    ParamSpec("maniac_raise_rate", 0.33, 0.20, 0.55),
    ParamSpec("station_call_rate", 0.42, 0.25, 0.65),
    ParamSpec("station_fold_rate_max", 0.30, 0.12, 0.45),
    ParamSpec("folder_fold_rate", 0.42, 0.25, 0.70),
    ParamSpec("folder_raise_rate_max", 0.20, 0.08, 0.35),
    ParamSpec("strong_raise_rate_max", 0.17, 0.06, 0.32),
    ParamSpec("strong_call_rate_max", 0.28, 0.12, 0.42),
    ParamSpec("strong_fold_rate_min", 0.34, 0.18, 0.58),
    ParamSpec("fold_pressure_folder_bonus", 0.18, 0.00, 0.35),
    ParamSpec("fold_pressure_station_penalty", 0.18, 0.00, 0.35),
    ParamSpec("station_score_bonus", 0.20, 0.00, 0.40),
    ParamSpec("maniac_allin_weight", 1.50, 0.50, 3.00),
    ParamSpec("maniac_score_bonus", 0.20, 0.00, 0.45),
    ParamSpec("equity_preflop_position_bonus", 0.03, 0.00, 0.08),
    ParamSpec("equity_active_penalty", 0.055, 0.015, 0.10),
    ParamSpec("equity_postflop_base", 0.17, 0.05, 0.30),
    ParamSpec("equity_strength_weight", 0.44, 0.25, 0.70),
    ParamSpec("equity_made_weight", 0.20, 0.05, 0.35),
    ParamSpec("equity_draw_weight", 0.11, 0.00, 0.25),
    ParamSpec("equity_broadway_weight", 0.04, 0.00, 0.12),
    ParamSpec("equity_wet_penalty", 0.045, 0.00, 0.12),
    ParamSpec("equity_paired_penalty", 0.025, 0.00, 0.10),
    ParamSpec("value_threshold_base", 0.66, 0.52, 0.82),
    ParamSpec("value_threshold_multiway_bonus", 0.07, 0.00, 0.16),
    ParamSpec("value_threshold_station_discount", 0.04, 0.00, 0.12),
    ParamSpec("call_margin_base", 0.075, 0.02, 0.16),
    ParamSpec("call_margin_multiway_bonus", 0.04, 0.00, 0.12),
    ParamSpec("call_margin_river_bonus", 0.035, 0.00, 0.12),
    ParamSpec("call_margin_strong_bonus", 0.035, 0.00, 0.12),
    ParamSpec("call_margin_maniac_discount", 0.045, 0.00, 0.12),
    ParamSpec("call_margin_min", 0.02, 0.00, 0.08),
    ParamSpec("default_value_fraction", 0.55, 0.33, 0.90),
    ParamSpec("default_value_conf_base", 0.66, 0.45, 0.85),
    ParamSpec("default_value_conf_equity_weight", 0.25, 0.00, 0.45),
    ParamSpec("default_value_hint", 0.15, -0.05, 0.30),
    ParamSpec("default_semibluff_equity", 0.39, 0.22, 0.58),
    ParamSpec("default_semibluff_fold_pressure", 0.38, 0.20, 0.62),
    ParamSpec("default_semibluff_fraction", 0.45, 0.25, 0.75),
    ParamSpec("default_raise_value_equity", 0.80, 0.66, 0.92),
    ParamSpec("default_raise_owed_pot_max", 0.35, 0.15, 0.65),
    ParamSpec("default_raise_fraction", 0.75, 0.45, 1.20),
    ParamSpec("pot_control_call_margin_floor", 0.055, 0.00, 0.14),
    ParamSpec("pot_control_call_margin_discount", 0.025, 0.00, 0.10),
    ParamSpec("pot_control_owed_pot_max", 0.45, 0.15, 0.80),
    ParamSpec("value_station_strong_equity", 0.61, 0.45, 0.78),
    ParamSpec("value_station_strong_strength", 0.78, 0.60, 0.92),
    ParamSpec("value_station_bet_equity", 0.54, 0.38, 0.72),
    ParamSpec("value_station_high_equity", 0.72, 0.58, 0.88),
    ParamSpec("value_station_big_fraction", 0.85, 0.55, 1.30),
    ParamSpec("value_station_normal_fraction", 0.67, 0.40, 1.00),
    ParamSpec("value_station_call_margin", 0.08, 0.00, 0.18),
    ParamSpec("value_station_raise_equity", 0.83, 0.68, 0.95),
    ParamSpec("value_station_raise_owed_pot_max", 0.30, 0.10, 0.60),
    ParamSpec("anti_maniac_value_equity", 0.82, 0.65, 0.94),
    ParamSpec("anti_maniac_value_spr_max", 3.00, 1.20, 6.00),
    ParamSpec("anti_maniac_value_fraction", 0.75, 0.45, 1.25),
    ParamSpec("anti_maniac_call_margin_discount", 0.05, 0.00, 0.12),
    ParamSpec("anti_maniac_raise_equity", 0.84, 0.68, 0.96),
    ParamSpec("anti_maniac_raise_owed_pot_max", 0.40, 0.12, 0.75),
    ParamSpec("pressure_dry_weight", 0.25, 0.00, 0.45),
    ParamSpec("pressure_broadway_weight", 0.20, 0.00, 0.40),
    ParamSpec("pressure_position_weight", 0.18, 0.00, 0.40),
    ParamSpec("pressure_value_equity", 0.61, 0.42, 0.78),
    ParamSpec("pressure_value_fraction", 0.67, 0.40, 1.05),
    ParamSpec("pressure_bluff_threshold", 0.63, 0.35, 0.90),
    ParamSpec("pressure_bluff_equity", 0.26, 0.10, 0.45),
    ParamSpec("pressure_bluff_fraction", 0.56, 0.35, 0.85),
    ParamSpec("pot_odds_value_equity", 0.62, 0.45, 0.78),
    ParamSpec("pot_odds_value_made_equity", 0.55, 0.35, 0.72),
    ParamSpec("pot_odds_value_fraction", 0.48, 0.30, 0.65),
    ParamSpec("pot_odds_bluff_fold_pressure", 0.38, 0.20, 0.65),
    ParamSpec("pot_odds_bluff_equity", 0.24, 0.08, 0.45),
    ParamSpec("pot_odds_bluff_fraction", 0.56, 0.35, 0.85),
    ParamSpec("pot_odds_call_margin", 0.065, 0.00, 0.16),
    ParamSpec("spr_commit_spr_max", 2.50, 1.00, 5.00),
    ParamSpec("spr_commit_equity", 0.67, 0.50, 0.84),
    ParamSpec("spr_commit_strength", 0.83, 0.65, 0.95),
    ParamSpec("spr_commit_draw_equity", 0.52, 0.34, 0.72),
    ParamSpec("spr_jam_stack_pot_max", 1.15, 0.50, 2.50),
    ParamSpec("spr_jam_equity", 0.76, 0.60, 0.92),
    ParamSpec("spr_raise_fraction", 0.85, 0.50, 1.40),
    ParamSpec("spr_call_margin", 0.05, 0.00, 0.14),
    ParamSpec("spr_call_owed_stack_max", 0.28, 0.10, 0.55),
    ParamSpec("short_stack_bb", 15.0, 6.0, 25.0),
    ParamSpec("short_stack_push_base", 0.62, 0.42, 0.80),
    ParamSpec("short_stack_position_discount", 0.09, 0.00, 0.20),
    ParamSpec("short_stack_facing_bonus", 0.08, 0.00, 0.18),
    ParamSpec("short_stack_postflop_equity", 0.62, 0.45, 0.80),
    ParamSpec("cbet_dry_fraction", 0.33, 0.20, 0.55),
    ParamSpec("cbet_wet_fraction", 0.45, 0.25, 0.75),
    ParamSpec("cbet_value_equity", 0.58, 0.40, 0.75),
    ParamSpec("cbet_conf_base", 0.58, 0.40, 0.75),
    ParamSpec("cbet_broadway_conf_weight", 0.12, 0.00, 0.30),
    ParamSpec("cbet_fold_conf_weight", 0.08, 0.00, 0.25),
    ParamSpec("blocker_strength_threshold", 0.55, 0.35, 0.78),
    ParamSpec("blocker_fold_pressure", 0.36, 0.18, 0.65),
    ParamSpec("blocker_equity_min", 0.24, 0.08, 0.42),
    ParamSpec("blocker_equity_max", 0.62, 0.42, 0.82),
    ParamSpec("blocker_fraction", 0.67, 0.35, 1.05),
    ParamSpec("blocker_call_margin", 0.035, 0.00, 0.12),
    ParamSpec("showdown_value_equity", 0.69, 0.50, 0.85),
    ParamSpec("showdown_value_fraction", 0.45, 0.25, 0.70),
    ParamSpec("showdown_call_margin", 0.045, 0.00, 0.14),
    ParamSpec("showdown_owed_pot_max", 0.65, 0.25, 1.00),
    ParamSpec("strong_unknown_value_equity", 0.76, 0.58, 0.92),
    ParamSpec("strong_unknown_value_fraction", 0.55, 0.30, 0.85),
    ParamSpec("strong_unknown_call_margin", 0.12, 0.04, 0.24),
    ParamSpec("strong_unknown_owed_stack_max", 0.22, 0.08, 0.45),
    ParamSpec("score_confidence_weight", 0.95, 0.50, 1.40),
    ParamSpec("score_equity_weight", 0.55, 0.20, 0.95),
    ParamSpec("score_risk_weight", 0.42, 0.10, 0.90),
    ParamSpec("score_value_bonus", 0.08, -0.05, 0.18),
    ParamSpec("score_semibluff_bonus", 0.04, -0.08, 0.15),
    ParamSpec("score_trap_bonus", 0.05, -0.08, 0.16),
    ParamSpec("score_pot_control_bonus", 0.03, -0.08, 0.14),
    ParamSpec("score_commit_bonus", 0.02, -0.12, 0.14),
    ParamSpec("score_bluff_bonus", -0.01, -0.16, 0.10),
    ParamSpec("score_check_call_bonus", 0.02, -0.08, 0.12),
    ParamSpec("score_legal_bonus", 0.02, -0.05, 0.08),
    ParamSpec("score_illegal_penalty", -0.15, -0.40, -0.02),
)

PARAM_NAMES = tuple(spec.name for spec in PARAM_SPECS)
_SPEC_BY_NAME = {spec.name: spec for spec in PARAM_SPECS}
DEFAULT_PARAMS = {spec.name: spec.default for spec in PARAM_SPECS}


def coerce_params(params: Mapping[str, float] | None = None) -> dict[str, float]:
    source = DEFAULT_PARAMS if params is None else {**DEFAULT_PARAMS, **dict(params)}
    return {name: _SPEC_BY_NAME[name].clip(float(source[name])) for name in PARAM_NAMES}


def vector_from_params(params: Mapping[str, float] | None = None, normalized: bool = False) -> np.ndarray:
    values = coerce_params(params)
    if normalized:
        return np.asarray([_SPEC_BY_NAME[name].normalize(values[name]) for name in PARAM_NAMES], dtype=np.float64)
    return np.asarray([values[name] for name in PARAM_NAMES], dtype=np.float64)


def params_from_vector(vector: np.ndarray, normalized: bool = False) -> dict[str, float]:
    arr = np.asarray(vector, dtype=np.float64).reshape(-1)
    if arr.size != len(PARAM_NAMES):
        raise ValueError(f"expected {len(PARAM_NAMES)} params, got {arr.size}")
    if normalized:
        return {name: _SPEC_BY_NAME[name].denormalize(float(arr[index])) for index, name in enumerate(PARAM_NAMES)}
    return {name: _SPEC_BY_NAME[name].clip(float(arr[index])) for index, name in enumerate(PARAM_NAMES)}


def load_params_npz(path: str | Path) -> dict[str, float]:
    data = np.load(path, allow_pickle=False)
    if "param_names" in data.files and "values" in data.files:
        names = [str(item) for item in data["param_names"]]
        values = {name: float(value) for name, value in zip(names, data["values"])}
        return coerce_params(values)
    if "normalized_values" in data.files:
        return params_from_vector(data["normalized_values"], normalized=True)
    raise ValueError(f"unsupported params artifact: {path}")


def load_params(data_dir: str | Path | None = None, filename: str = "params.npz") -> dict[str, float]:
    if data_dir is None:
        return coerce_params()
    path = Path(data_dir) / filename
    if not path.is_file():
        return coerce_params()
    try:
        return load_params_npz(path)
    except Exception:
        return coerce_params()


def save_params_npz(path: str | Path, params: Mapping[str, float] | None = None, **metadata) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    values = coerce_params(params)
    payload = {
        "param_names": np.asarray(PARAM_NAMES),
        "values": vector_from_params(values, normalized=False).astype(np.float32),
        "normalized_values": vector_from_params(values, normalized=True).astype(np.float32),
        "defaults": vector_from_params(DEFAULT_PARAMS, normalized=False).astype(np.float32),
        "lower_bounds": np.asarray([spec.lower for spec in PARAM_SPECS], dtype=np.float32),
        "upper_bounds": np.asarray([spec.upper for spec in PARAM_SPECS], dtype=np.float32),
    }
    for key, value in metadata.items():
        if isinstance(value, str):
            payload[key] = np.asarray([value])
        elif np.isscalar(value):
            payload[key] = np.asarray([value])
    np.savez_compressed(path, **payload)

