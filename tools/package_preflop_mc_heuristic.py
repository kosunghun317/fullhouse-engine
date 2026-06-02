"""Create a validated submission zip for the preflop-MC heuristic bot."""

from __future__ import annotations

import argparse
import json
import sys
import time
import zipfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from sandbox.validator import validate


DEFAULT_CANDIDATE_ROOT = ROOT / "runs" / "preflop_mc_heuristic"
DEFAULT_OUTPUT = ROOT / "dist" / "preflop_mc_heuristic_bot.zip"
IGNORED_NAMES = {".DS_Store"}
IGNORED_SUFFIXES = {".pyc", ".pyo"}


def discover_latest_candidate(root: Path = DEFAULT_CANDIDATE_ROOT) -> Path:
    """Return the newest generated preflop-MC candidate bot directory."""
    root = root.resolve()
    candidates = [
        path
        for path in root.glob("*/bot")
        if path.is_dir() and (path / "bot.py").is_file()
    ]
    if not candidates:
        raise FileNotFoundError(f"no preflop-MC bot candidates found under {root}")
    return max(candidates, key=lambda path: (path.stat().st_mtime, str(path))).resolve()


def resolve_candidate(path: Path | None) -> Path:
    """Accept either a bot directory or a run directory containing bot/."""
    if path is None:
        return discover_latest_candidate()
    path = path.resolve()
    if (path / "bot.py").is_file():
        return path
    nested = path / "bot"
    if (nested / "bot.py").is_file():
        return nested.resolve()
    raise FileNotFoundError(f"candidate must contain bot.py or bot/bot.py: {path}")


def unique_output_path(output: Path) -> Path:
    """Return output, or a timestamped sibling if output already exists."""
    output = output.resolve()
    if not output.exists():
        return output
    stamp = time.strftime("%Y%m%d-%H%M%S")
    for index in range(1, 1000):
        suffix = f"-{stamp}" if index == 1 else f"-{stamp}-{index}"
        candidate = output.with_name(f"{output.stem}{suffix}{output.suffix}")
        if not candidate.exists():
            return candidate
    raise FileExistsError(f"could not find a free output path near {output}")


def iter_submission_files(candidate_dir: Path):
    bot_py = candidate_dir / "bot.py"
    if not bot_py.is_file():
        raise FileNotFoundError(f"missing bot.py: {bot_py}")
    if bot_py.is_symlink():
        raise ValueError(f"symlinked bot.py is not allowed: {bot_py}")
    yield bot_py, "bot.py"

    data_dir = candidate_dir / "data"
    if not data_dir.exists():
        return
    if not data_dir.is_dir() or data_dir.is_symlink():
        raise ValueError(f"data must be a real directory: {data_dir}")

    for path in sorted(data_dir.rglob("*")):
        if path.is_symlink():
            raise ValueError(f"symlinks are not allowed in submissions: {path}")
        if not path.is_file():
            continue
        if path.name in IGNORED_NAMES or path.suffix in IGNORED_SUFFIXES:
            continue
        if path.suffix == ".py":
            raise ValueError(f"data/ may not contain Python files: {path}")
        yield path, path.relative_to(candidate_dir).as_posix()


def create_submission_zip(
    candidate_dir: Path | None,
    output: Path = DEFAULT_OUTPUT,
    *,
    make_unique: bool = True,
    validate_zip: bool = True,
) -> dict:
    candidate_dir = resolve_candidate(candidate_dir)
    zip_output = unique_output_path(output) if make_unique else output.resolve()
    if zip_output.exists():
        raise FileExistsError(f"refusing to overwrite existing zip: {zip_output}")

    zip_output.parent.mkdir(parents=True, exist_ok=True)
    members = []
    with zipfile.ZipFile(zip_output, "x", compression=zipfile.ZIP_DEFLATED) as zf:
        for source, arcname in iter_submission_files(candidate_dir):
            zf.write(source, arcname)
            members.append(arcname)

    validation = validate(str(zip_output)) if validate_zip else None
    return {
        "candidate": str(candidate_dir),
        "output": str(zip_output),
        "size_bytes": zip_output.stat().st_size,
        "members": members,
        "validator_passed": None if validation is None else validation["passed"],
        "errors": [] if validation is None else validation["errors"],
        "warnings": [] if validation is None else validation["warnings"],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--candidate",
        default=None,
        help="Preflop-MC bot directory, or run directory containing bot/. Defaults to newest run.",
    )
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT), help="Submission zip path.")
    parser.add_argument(
        "--no-unique-output",
        action="store_true",
        help="Fail instead of adding a timestamp suffix when --output already exists.",
    )
    parser.add_argument(
        "--skip-validation",
        action="store_true",
        help="Create the zip without running the validator.",
    )
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    report = create_submission_zip(
        Path(args.candidate) if args.candidate else None,
        Path(args.output),
        make_unique=not args.no_unique_output,
        validate_zip=not args.skip_validation,
    )

    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        print(f"wrote {report['output']} ({report['size_bytes']} bytes)")
        print(f"candidate: {report['candidate']}")
        print("members:", ", ".join(report["members"]))
        if report["validator_passed"] is not None:
            print("validator:", "passed" if report["validator_passed"] else "failed")
            for error in report["errors"]:
                print("error:", error)

    return 0 if report["validator_passed"] is not False else 1


if __name__ == "__main__":
    raise SystemExit(main())
