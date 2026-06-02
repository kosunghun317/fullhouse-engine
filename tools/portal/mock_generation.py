"""Generate and evaluate local mocks from portal profile artifacts."""

from __future__ import annotations

import json
import random
import shutil
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from sandbox.match import run_match

DEFAULT_TEMPLATE = ROOT / "bots" / "mock_competitors" / "portal_profile" / "bot.py"
DEFAULT_CANDIDATE = "bots/heuristic/bot.py"
VARIANT_NUMERIC_BOUNDS = {
    "vpip": (0.04, 0.70),
    "pfr": (0.01, 0.60),
    "raise_rate": (0.01, 0.60),
    "call_rate": (0.01, 0.50),
    "pressure_fold_rate": (0.02, 0.96),
    "showdown_rate": (0.005, 0.40),
    "all_in_rate": (0.0, 0.10),
    "avg_raise_to_pot": (0.20, 9.0),
    "aggression_bias": (-0.40, 0.55),
    "bluff_bias": (-0.45, 0.55),
    "value_bias": (-0.25, 0.50),
}


def safe_profile_name(name: str) -> str:
    return "".join(ch if ch.isalnum() or ch in {"_", "-"} else "_" for ch in name)


def load_profile_summary(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def profile_summary_path(source: Path) -> Path:
    if source.is_dir():
        candidate = source / "profiles" / "profile_summary.json"
        if candidate.exists():
            return candidate
        return source / "profile_summary.json"
    return source


def _clamp(value: float, lo: float, hi: float) -> float:
    return min(hi, max(lo, value))


def variant_mock_config(config: dict, rng: random.Random, spread: float) -> dict:
    adjusted = dict(config)
    for key, bounds in VARIANT_NUMERIC_BOUNDS.items():
        value = adjusted.get(key)
        if not isinstance(value, (int, float)):
            continue
        lo, hi = bounds
        width = max(0.01, abs(float(value)) * spread)
        adjusted[key] = round(_clamp(rng.gauss(float(value), width), lo, hi), 4)
    if isinstance(adjusted.get("pfr"), (int, float)) and isinstance(adjusted.get("vpip"), (int, float)):
        adjusted["pfr"] = round(min(float(adjusted["pfr"]), float(adjusted["vpip"]) * 0.95), 4)
    return adjusted


def profile_variant(profile: dict, variant_index: int, rng: random.Random, spread: float) -> dict:
    if variant_index <= 0:
        variant = dict(profile)
        variant["mock_config"] = dict(profile.get("mock_config", {}))
        variant["variant"] = {"index": 0, "name": "base", "spread": 0.0}
        return variant
    variant = dict(profile)
    variant["mock_config"] = variant_mock_config(profile.get("mock_config", {}), rng, spread)
    variant["variant"] = {
        "index": variant_index,
        "name": f"v{variant_index:02d}",
        "spread": spread,
    }
    return variant


def materialize_profile_mocks(
    profile_summary: dict,
    output: Path,
    template: Path = DEFAULT_TEMPLATE,
    source_profile_summary: Path | None = None,
    variants_per_profile: int = 1,
    variant_seed: int = 1729,
    variant_spread: float = 0.12,
) -> dict:
    output.mkdir(parents=True, exist_ok=True)
    profiles = []
    source_profiles = list(profile_summary.get("profiles", []))
    variant_count = max(1, int(variants_per_profile))
    spread = max(0.0, float(variant_spread))
    rng = random.Random(int(variant_seed))
    for profile in source_profiles:
        name = str(profile["profile"])
        safe_name = safe_profile_name(name)
        for variant_index in range(variant_count):
            variant = profile_variant(profile, 0 if variant_count == 1 else variant_index + 1, rng, spread)
            variant_name = variant["variant"]["name"]
            dir_name = safe_name if variant_count == 1 else f"{safe_name}_{variant_name}"
            bot_id = "portal_" + dir_name
            bot_dir = output / dir_name
            data_dir = bot_dir / "data"
            data_dir.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(template, bot_dir / "bot.py")
            (data_dir / "profile.json").write_text(
                json.dumps(variant, indent=2, sort_keys=True),
                encoding="utf-8",
            )
            profiles.append({
                "profile": name,
                "variant": variant["variant"],
                "bot_id": bot_id,
                "path": str(bot_dir),
                "bot_count": profile.get("bot_count"),
                "top_official_rank": profile.get("top_official_rank"),
                "mock_config": variant.get("mock_config", {}),
            })

    manifest = {
        "source_profile_summary": str(source_profile_summary) if source_profile_summary else None,
        "template": str(template),
        "output": str(output),
        "source_profile_count": len(source_profiles),
        "variants_per_profile": variant_count,
        "variant_seed": int(variant_seed),
        "variant_spread": spread,
        "profile_count": len(profiles),
        "profiles": profiles,
        "suite_bots": {row["bot_id"]: row["path"] for row in profiles},
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return manifest


def load_mock_manifest(mocks_dir: Path) -> dict:
    return json.loads((mocks_dir / "manifest.json").read_text(encoding="utf-8"))


def selected_profile_rows(manifest: dict, profiles: list[str] | None = None) -> list[dict]:
    rows = list(manifest.get("profiles", []))
    if profiles:
        wanted = set(profiles)
        rows = [row for row in rows if row["profile"] in wanted or row["bot_id"] in wanted]
    return rows


def build_match_specs(
    manifest: dict,
    candidate: str = DEFAULT_CANDIDATE,
    profiles: list[str] | None = None,
    mode: str = "sixmax",
    candidate_id: str = "candidate",
    table_count: int | None = None,
    table_seed: int = 1729,
) -> list[dict]:
    rows = selected_profile_rows(manifest, profiles)
    if not rows:
        raise ValueError("no portal profile mocks selected")
    specs = []
    if mode == "heads-up":
        for row in rows:
            specs.append({
                "name": row["bot_id"],
                "bots": {candidate_id: candidate, row["bot_id"]: row["path"]},
            })
        return specs
    if len(rows) <= 5:
        table_rows = [rows]
    else:
        rng = random.Random(int(table_seed))
        shuffled = list(rows)
        rng.shuffle(shuffled)
        count = table_count or max(1, (len(shuffled) + 4) // 5)
        stride = max(1, len(shuffled) // 5)
        table_rows = [
            [shuffled[(index + offset * stride) % len(shuffled)] for offset in range(5)]
            for index in range(count)
        ]
    for index, chunk in enumerate(table_rows):
        bots = {candidate_id: candidate}
        bots.update({row["bot_id"]: row["path"] for row in chunk})
        specs.append({"name": f"portal_profiles_{index + 1}", "bots": bots})
    return specs


def summarize_candidate(results: list[dict]) -> dict:
    deltas = [row["chip_delta"].get("candidate", 0) for row in results]
    errors = []
    for row in results:
        errors.extend(row["bot_errors"].get("candidate", []))
    return {
        "runs": len(results),
        "mean_delta": round(statistics.mean(deltas), 2) if deltas else 0,
        "median_delta": round(statistics.median(deltas), 2) if deltas else 0,
        "min_delta": min(deltas) if deltas else 0,
        "max_delta": max(deltas) if deltas else 0,
        "stdev_delta": round(statistics.pstdev(deltas), 2) if len(deltas) > 1 else 0,
        "positive_runs": sum(1 for delta in deltas if delta > 0),
        "nonnegative_runs": sum(1 for delta in deltas if delta >= 0),
        "candidate_error_count": len(errors),
        "candidate_errors": errors,
    }


def evaluate_profile_mocks(
    mocks_dir: Path,
    candidate: str,
    seeds: list[int],
    hands: int,
    mode: str = "sixmax",
    profiles: list[str] | None = None,
) -> dict:
    manifest = load_mock_manifest(mocks_dir)
    specs = build_match_specs(manifest, candidate=candidate, profiles=profiles, mode=mode)
    suite_reports = []
    all_results = []
    for spec in specs:
        results = []
        for seed in seeds:
            result = run_match(
                match_id=f"{spec['name']}_{seed}",
                bot_paths=spec["bots"],
                n_hands=hands,
                verbose=False,
                seed=seed,
            )
            row = {
                "suite": spec["name"],
                "seed": seed,
                "chip_delta": result["chip_delta"],
                "final_stacks": result["final_stacks"],
                "bot_errors": result["bot_errors"],
                "duration_s": result["duration_s"],
            }
            results.append(row)
            all_results.append(row)
        suite_reports.append({
            "suite": spec["name"],
            "bots": spec["bots"],
            "summary": summarize_candidate(results),
        })
    return {
        "mocks_dir": str(mocks_dir),
        "candidate": candidate,
        "mode": mode,
        "hands": hands,
        "seeds": seeds,
        "summary": summarize_candidate(all_results),
        "suites": suite_reports,
    }
