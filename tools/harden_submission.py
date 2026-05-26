"""Run submission hardening checks for the heuristic bot.

This is local tooling only. It rebuilds optional data, packages the submission,
validates both directory and zip formats, inspects zip contents, and runs short
sanity matches against the same entrypoints the sandbox accepts.
"""

import argparse
import json
import os
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from sandbox.match import run_match
from sandbox.validator import validate
from tools.build_heuristic_tables import DEFAULT_OUTPUT as TABLES_OUTPUT
from tools.build_heuristic_tables import build as build_tables
from tools.package_heuristic import DEFAULT_OUTPUT as ZIP_OUTPUT
from tools.package_heuristic import package


HEURISTIC_DIR = ROOT / "bots" / "heuristic"
SHARK = ROOT / "bots" / "shark" / "bot.py"


def _zip_report(path):
    with zipfile.ZipFile(path) as zf:
        names = sorted(zf.namelist())
    return {
        "has_bot_py": "bot.py" in names,
        "data_files": [name for name in names if name.startswith("data/")],
        "python_files_in_data": [
            name for name in names
            if name.startswith("data/") and name.endswith(".py")
        ],
        "members": names,
    }


def _match_report(label, heuristic_path, hands, seed, docker=False):
    old = os.environ.get("USE_DOCKER")
    if docker:
        os.environ["USE_DOCKER"] = "true"
    else:
        os.environ.pop("USE_DOCKER", None)
    try:
        result = run_match(
            match_id=f"harden_{label}_{seed}",
            bot_paths={
                "heuristic": str(heuristic_path),
                "shark": str(SHARK),
            },
            n_hands=hands,
            verbose=False,
            seed=seed,
        )
    finally:
        if old is None:
            os.environ.pop("USE_DOCKER", None)
        else:
            os.environ["USE_DOCKER"] = old

    errors = result["bot_errors"].get("heuristic", [])
    return {
        "label": label,
        "hands": hands,
        "seed": seed,
        "docker": docker,
        "chip_delta": result["chip_delta"].get("heuristic", 0),
        "final_stack": result["final_stacks"].get("heuristic"),
        "duration_s": round(result.get("duration_s", 0.0), 2),
        "heuristic_errors": errors,
        "passed": len(errors) == 0,
    }


def harden(output, hands, seed, docker=False):
    table_report = build_tables(TABLES_OUTPUT)
    zip_path = package(output)
    directory_validation = validate(str(HEURISTIC_DIR))
    zip_validation = validate(str(zip_path))
    zip_contents = _zip_report(zip_path)
    matches = [
        _match_report("directory", HEURISTIC_DIR, hands, seed, docker=False),
        _match_report("zip", zip_path, hands, seed + 1, docker=False),
    ]
    if docker:
        matches.append(_match_report("zip_docker", zip_path, hands, seed + 2, docker=True))

    passed = (
        directory_validation["passed"]
        and zip_validation["passed"]
        and zip_contents["has_bot_py"]
        and "data/tables.npz" in zip_contents["data_files"]
        and not zip_contents["python_files_in_data"]
        and all(match["passed"] for match in matches)
    )
    return {
        "passed": passed,
        "tables": table_report,
        "zip": {
            "output": str(zip_path),
            "size_bytes": zip_path.stat().st_size,
            "contents": zip_contents,
        },
        "directory_validation": directory_validation,
        "zip_validation": zip_validation,
        "matches": matches,
    }


def main():
    parser = argparse.ArgumentParser(description="Harden heuristic submission artifacts")
    parser.add_argument("--output", default=str(ZIP_OUTPUT))
    parser.add_argument("--hands", type=int, default=80)
    parser.add_argument("--seed", type=int, default=9901)
    parser.add_argument("--docker", action="store_true", help="Also run a USE_DOCKER=true zip match")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    report = harden(Path(args.output).resolve(), args.hands, args.seed, docker=args.docker)
    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        print("passed:", report["passed"])
        print("zip:", report["zip"]["output"], report["zip"]["size_bytes"], "bytes")
        print("directory validator:", report["directory_validation"]["passed"])
        print("zip validator:", report["zip_validation"]["passed"])
        for match in report["matches"]:
            print(
                f"{match['label']}: passed={match['passed']} "
                f"delta={match['chip_delta']} errors={len(match['heuristic_errors'])}"
            )
    raise SystemExit(0 if report["passed"] else 1)


if __name__ == "__main__":
    main()
