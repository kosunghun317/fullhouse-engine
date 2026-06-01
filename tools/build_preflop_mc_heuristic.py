"""Build a heuristic candidate with preflop score replaced by Monte Carlo."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import textwrap
import zipfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = ROOT / "dist" / "heuristic_bot.zip"
DEFAULT_OUTPUT_ROOT = ROOT / "runs" / "preflop_mc_heuristic"


def _safe_extract_zip(zip_path: Path, output_dir: Path) -> None:
    with zipfile.ZipFile(zip_path) as zf:
        for member in zf.infolist():
            name = member.filename
            if name.startswith("/") or name.startswith("\\"):
                raise ValueError(f"unsafe absolute zip path: {name!r}")
            target = os.path.normpath(os.path.join(output_dir, name))
            if not (target == str(output_dir) or target.startswith(str(output_dir) + os.sep)):
                raise ValueError(f"unsafe zip traversal path: {name!r}")
            if (member.external_attr >> 16) & 0o170000 == 0o120000:
                raise ValueError(f"unsafe zip symlink path: {name!r}")
        zf.extractall(output_dir)


def _copy_candidate_input(input_path: Path, output_dir: Path) -> None:
    if output_dir.exists():
        raise FileExistsError(f"output already exists: {output_dir}")
    output_dir.mkdir(parents=True)
    if input_path.is_dir():
        for child in input_path.iterdir():
            target = output_dir / child.name
            if child.is_dir():
                shutil.copytree(child, target)
            else:
                shutil.copy2(child, target)
    elif input_path.suffix == ".zip":
        _safe_extract_zip(input_path, output_dir)
    elif input_path.name == "bot.py":
        shutil.copy2(input_path, output_dir / "bot.py")
    else:
        raise ValueError(f"unsupported input path: {input_path}")
    if not (output_dir / "bot.py").is_file():
        raise ValueError(f"candidate input did not contain bot.py: {input_path}")


def preflop_mc_override_source(samples: int, budget_s: float, min_samples: int = 256) -> str:
    return textwrap.dedent(
        f"""

        # ---------------------------------------------------------------------------
        # Generated preflop Monte Carlo override
        # ---------------------------------------------------------------------------

        PREFLOP_MC_SAMPLES = _int_env("HEURISTIC_PREFLOP_MC_SAMPLES", {int(samples)})
        PREFLOP_MC_BUDGET_S = _float_env("HEURISTIC_PREFLOP_MC_BUDGET_S", {float(budget_s):.6f})
        PREFLOP_MC_MIN_SAMPLES = _int_env("HEURISTIC_PREFLOP_MC_MIN_SAMPLES", {int(min_samples)})
        PREFLOP_MC_CACHE = {{}}
        _TABLE_PREFLOP_SCORE = _preflop_score


        def _preflop_mc_seed(hand_class, samples):
            seed = 0
            for ch in str(hand_class) + "|" + str(int(samples)):
                seed = (seed * 131 + ord(ch)) & 0xFFFFFFFF
            return seed


        def _preflop_score(cards):
            cls = _hand_class(cards)
            key = (cls, int(PREFLOP_MC_SAMPLES))
            cached = PREFLOP_MC_CACHE.get(key)
            if cached is not None:
                return cached
            try:
                hero = [eval7.Card(card) for card in cards[:2]]
            except Exception:
                return _TABLE_PREFLOP_SCORE(cards)
            if len(hero) < 2:
                return _TABLE_PREFLOP_SCORE(cards)

            deck = [card for card in FULL_DECK if card not in set(hero)]
            if len(deck) < 7:
                return _TABLE_PREFLOP_SCORE(cards)

            rng = random.Random(_preflop_mc_seed(cls, PREFLOP_MC_SAMPLES))
            start = time.perf_counter()
            wins = 0.0
            trials = 0
            target = max(1, int(PREFLOP_MC_SAMPLES))
            min_samples = max(1, min(target, int(PREFLOP_MC_MIN_SAMPLES)))

            for index in range(target):
                if (
                    index >= min_samples
                    and index % 64 == 0
                    and time.perf_counter() - start > float(PREFLOP_MC_BUDGET_S)
                ):
                    break
                draw = rng.sample(deck, 7)
                opp = draw[:2]
                board = draw[2:]
                hero_score = eval7.evaluate(hero + board)
                opp_score = eval7.evaluate(opp + board)
                if hero_score > opp_score:
                    wins += 1.0
                elif hero_score == opp_score:
                    wins += 0.5
                trials += 1

            if trials < min_samples:
                return _TABLE_PREFLOP_SCORE(cards)
            score = int(round(100.0 * wins / max(1, trials)))
            score = int(_clamp(score, 1, 99))
            if len(PREFLOP_MC_CACHE) > 256:
                PREFLOP_MC_CACHE.clear()
            PREFLOP_MC_CACHE[key] = score
            return score
        """
    )


def build_preflop_mc_candidate(input_path: Path, output_dir: Path, samples: int, budget_s: float, min_samples: int = 256) -> dict:
    input_path = input_path.resolve()
    output_dir = output_dir.resolve()
    _copy_candidate_input(input_path, output_dir)
    bot_path = output_dir / "bot.py"
    source = bot_path.read_text(encoding="utf-8")
    source += preflop_mc_override_source(samples=samples, budget_s=budget_s, min_samples=min_samples)
    bot_path.write_text(source, encoding="utf-8")
    data_files = sorted(str(path.relative_to(output_dir)) for path in (output_dir / "data").glob("*") if path.is_file()) if (output_dir / "data").is_dir() else []
    return {
        "input": str(input_path),
        "output": str(output_dir),
        "bot_py": str(bot_path),
        "samples": int(samples),
        "budget_s": float(budget_s),
        "min_samples": int(min_samples),
        "data_files": data_files,
    }


def zip_candidate(candidate_dir: Path, zip_output: Path) -> Path:
    candidate_dir = candidate_dir.resolve()
    zip_output = zip_output.resolve()
    if zip_output.exists():
        raise FileExistsError(f"zip output already exists: {zip_output}")
    zip_output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_output, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for path in sorted(candidate_dir.rglob("*")):
            if not path.is_file():
                continue
            if "__pycache__" in path.parts or path.suffix == ".pyc":
                continue
            zf.write(path, path.relative_to(candidate_dir).as_posix())
    return zip_output


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default=str(DEFAULT_INPUT), help="Submission zip, bot directory, or bot.py to copy.")
    parser.add_argument("--output", default=None, help="Output bot directory. Must not already exist.")
    parser.add_argument("--samples", type=int, default=4096)
    parser.add_argument("--budget", type=float, default=0.35)
    parser.add_argument("--min-samples", type=int, default=256)
    parser.add_argument("--zip-output", default=None, help="Optional zip output path for the generated candidate.")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    output = Path(args.output) if args.output else DEFAULT_OUTPUT_ROOT / f"preflop-mc-{args.samples}" / "bot"
    report = build_preflop_mc_candidate(
        input_path=Path(args.input),
        output_dir=output,
        samples=args.samples,
        budget_s=args.budget,
        min_samples=args.min_samples,
    )
    if args.zip_output:
        report["zip_output"] = str(zip_candidate(Path(report["output"]), Path(args.zip_output)))
    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        print(f"built preflop-MC heuristic candidate: {report['output']}")
        print(f"samples={report['samples']} budget_s={report['budget_s']} min_samples={report['min_samples']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
