"""Package and validate the heuristic bot submission zip."""

import argparse
import json
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from sandbox.validator import validate
from tools.heuristic_env_overrides import bake_heuristic_source, load_env_overrides


BOT_PATH = ROOT / "bots" / "heuristic" / "bot.py"
DATA_DIR = ROOT / "bots" / "heuristic" / "data"
DEFAULT_OUTPUT = ROOT / "dist" / "heuristic_bot.zip"


def _bot_source(env_file=None):
    source = BOT_PATH.read_text(encoding="utf-8")
    if not env_file:
        return source, None
    overrides = load_env_overrides(env_file)
    baked, report = bake_heuristic_source(source, overrides)
    report["env_file"] = str(env_file)
    return baked, report


def package(output, env_file=None):
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        output.unlink()
    bot_source, _report = _bot_source(env_file)
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("bot.py", bot_source)
        if DATA_DIR.exists():
            for path in DATA_DIR.rglob("*"):
                if path.is_file():
                    zf.write(path, "data/" + str(path.relative_to(DATA_DIR)))
    return output


def main():
    parser = argparse.ArgumentParser(description="Create and validate heuristic bot submission zip")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--env-file", default=None, help="JSON or best_env.sh file whose HEURISTIC_* values are baked into bot.py")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    env_file = Path(args.env_file).resolve() if args.env_file else None
    _source, bake_report = _bot_source(env_file)
    output = package(Path(args.output).resolve(), env_file=env_file)
    result = validate(str(output))
    payload = {
        "output": str(output),
        "size_bytes": output.stat().st_size,
        "validator_passed": result["passed"],
        "errors": result["errors"],
        "warnings": result["warnings"],
        "bake_report": bake_report,
    }

    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        print(f"wrote {payload['output']} ({payload['size_bytes']} bytes)")
        print("validator:", "passed" if payload["validator_passed"] else "failed")
        for error in payload["errors"]:
            print("error:", error)

    raise SystemExit(0 if result["passed"] else 1)


if __name__ == "__main__":
    main()
