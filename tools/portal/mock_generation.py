"""Generate and evaluate local mocks from portal profile artifacts."""

from __future__ import annotations

import json
import shutil
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from sandbox.match import run_match

DEFAULT_TEMPLATE = ROOT / "bots" / "mock_competitors" / "portal_profile" / "bot.py"
DEFAULT_CANDIDATE = "bots/heuristic/bot.py"


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


def materialize_profile_mocks(
    profile_summary: dict,
    output: Path,
    template: Path = DEFAULT_TEMPLATE,
    source_profile_summary: Path | None = None,
) -> dict:
    output.mkdir(parents=True, exist_ok=True)
    profiles = []
    for profile in profile_summary.get("profiles", []):
        name = str(profile["profile"])
        safe_name = safe_profile_name(name)
        bot_id = "portal_" + safe_name
        bot_dir = output / safe_name
        data_dir = bot_dir / "data"
        data_dir.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(template, bot_dir / "bot.py")
        (data_dir / "profile.json").write_text(
            json.dumps(profile, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        profiles.append({
            "profile": name,
            "bot_id": bot_id,
            "path": str(bot_dir),
            "bot_count": profile.get("bot_count"),
            "top_official_rank": profile.get("top_official_rank"),
            "mock_config": profile.get("mock_config", {}),
        })

    manifest = {
        "source_profile_summary": str(source_profile_summary) if source_profile_summary else None,
        "template": str(template),
        "output": str(output),
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
    for index in range(0, len(rows), 5):
        chunk = rows[index:index + 5]
        bots = {candidate_id: candidate}
        bots.update({row["bot_id"]: row["path"] for row in chunk})
        specs.append({"name": f"portal_profiles_{index // 5 + 1}", "bots": bots})
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
