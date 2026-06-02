"""Build replay-conditioned behavior-clone mocks from portal replay artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from portal.behavior_clone import (
    DEFAULT_TEMPLATE,
    build_behavior_clone_summary,
    materialize_behavior_clones,
)


def default_output(source: Path, rank_gte: int, rank_lte: int, group_by: str) -> Path:
    rank_part = f"r{rank_gte}_to_{rank_lte}"
    return Path("runs/portal_behavior_clones") / f"{source.name}_{rank_part}_{group_by}"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path, help="Portal replay artifact directory")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--template", type=Path, default=DEFAULT_TEMPLATE)
    parser.add_argument("--rank-gte", type=int, default=1)
    parser.add_argument("--rank-lte", type=int, default=64)
    parser.add_argument("--group-by", choices=["all", "profile", "bot"], default="profile")
    parser.add_argument("--min-actions", type=int, default=200)
    parser.add_argument("--min-cell-actions", type=int, default=20)
    parser.add_argument("--smoothing", type=float, default=0.25)
    parser.add_argument("--variants-per-policy", type=int, default=1)
    parser.add_argument("--variant-seed", type=int, default=1729)
    parser.add_argument("--action-noise", type=float, default=0.04)
    parser.add_argument("--size-noise", type=float, default=0.12)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    summary = build_behavior_clone_summary(
        args.directory,
        rank_lte=args.rank_lte,
        rank_gte=args.rank_gte,
        group_by=args.group_by,
        min_actions=args.min_actions,
        min_cell_actions=args.min_cell_actions,
        smoothing=args.smoothing,
    )
    output = args.output or default_output(args.directory, args.rank_gte, args.rank_lte, args.group_by)
    manifest = materialize_behavior_clones(
        summary,
        output,
        template=args.template,
        source=args.directory,
        variants_per_policy=args.variants_per_policy,
        variant_seed=args.variant_seed,
        action_noise=args.action_noise,
        size_noise=args.size_noise,
    )

    if args.json:
        print(json.dumps(manifest, indent=2, sort_keys=True))
    else:
        print(
            f"wrote {output} clones={manifest['profile_count']} "
            f"source_policies={manifest['source_policy_count']}"
        )
        for profile in manifest["profiles"]:
            variant = profile.get("variant", {}).get("name", "base")
            print(
                f"{profile['bot_id']}\t{profile['path']}\t"
                f"group={profile['profile']}\tvariant={variant}\t"
                f"actions={profile.get('action_count')}"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
