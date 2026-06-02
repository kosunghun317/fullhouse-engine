"""Materialize local mock competitors from portal profile artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from portal.mock_generation import (
    DEFAULT_TEMPLATE,
    load_profile_summary,
    materialize_profile_mocks,
    profile_summary_path,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path, help="Portal run dir, profiles dir, or profile_summary.json")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--template", type=Path, default=DEFAULT_TEMPLATE)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    summary_path = profile_summary_path(args.source)
    summary = load_profile_summary(summary_path)
    output = args.output
    if output is None:
        source_name = args.source.name if args.source.is_dir() else args.source.parent.name
        output = Path("runs/portal_profile_mocks") / source_name
    manifest = materialize_profile_mocks(
        summary,
        output,
        template=args.template,
        source_profile_summary=summary_path,
    )
    if args.json:
        print(json.dumps(manifest, indent=2, sort_keys=True))
    else:
        print(f"wrote {output} mocks={manifest['profile_count']}")
        for profile in manifest["profiles"]:
            print(f"{profile['bot_id']}\t{profile['path']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
