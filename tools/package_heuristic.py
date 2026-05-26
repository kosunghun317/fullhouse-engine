"""Package and validate the heuristic bot submission zip."""

import argparse
import json
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from sandbox.validator import validate


BOT_PATH = ROOT / "bots" / "heuristic" / "bot.py"
DATA_DIR = ROOT / "bots" / "heuristic" / "data"
DEFAULT_OUTPUT = ROOT / "dist" / "heuristic_bot.zip"


def package(output):
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        output.unlink()
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.write(BOT_PATH, "bot.py")
        if DATA_DIR.exists():
            for path in DATA_DIR.rglob("*"):
                if path.is_file():
                    zf.write(path, "data/" + str(path.relative_to(DATA_DIR)))
    return output


def main():
    parser = argparse.ArgumentParser(description="Create and validate heuristic bot submission zip")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    output = package(Path(args.output).resolve())
    result = validate(str(output))
    payload = {
        "output": str(output),
        "size_bytes": output.stat().st_size,
        "validator_passed": result["passed"],
        "errors": result["errors"],
        "warnings": result["warnings"],
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
