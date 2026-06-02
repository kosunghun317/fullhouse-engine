"""Analyze downloaded Fullhouse portal strategy tendencies."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from portal.analysis import build_report, print_text


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--top", type=int, default=80)
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--output-json", type=Path)
    args = parser.parse_args()

    report = build_report(args.directory, args.top)
    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        print_text(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
