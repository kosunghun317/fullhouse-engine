"""Fast in-process match runner for offline training.

This runner deliberately skips Docker, subprocess isolation, action timeouts,
and validator/resource checks. It is for local opponent training and fast
benchmarking only; use sandbox.match for submission-like validation.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import shutil
import sys
import tempfile
import time
import uuid
import zipfile
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from engine.game import STARTING_STACK, PokerEngine
from sandbox.match import MATCH_LOG_MAX_ENTRIES
from tools.parallel import map_parallel


def _prepare_bot_mount(bot_path: str) -> tuple[str, str | None]:
    """Prepare a bot path like sandbox.match, but for in-process import."""
    path = os.path.abspath(bot_path)
    if os.path.isdir(path):
        bot_py = os.path.join(path, "bot.py")
        if not os.path.isfile(bot_py):
            raise ValueError(f"Bot directory must contain bot.py: {bot_path!r}")
        return path, None

    if path.endswith(".py") and os.path.isfile(path):
        tmpdir = tempfile.mkdtemp(prefix="fhfastbot_")
        shutil.copy(path, os.path.join(tmpdir, "bot.py"))
        return tmpdir, tmpdir

    if path.endswith(".zip") and os.path.isfile(path):
        tmpdir = tempfile.mkdtemp(prefix="fhfastbot_")
        with zipfile.ZipFile(path) as zf:
            for member in zf.infolist():
                name = member.filename
                if name.startswith("/") or name.startswith("\\"):
                    shutil.rmtree(tmpdir, ignore_errors=True)
                    raise ValueError(f"Unsafe zip path (absolute): {name!r}")
                norm = os.path.normpath(os.path.join(tmpdir, name))
                if not norm.startswith(tmpdir + os.sep) and norm != tmpdir:
                    shutil.rmtree(tmpdir, ignore_errors=True)
                    raise ValueError(f"Unsafe zip path (traversal): {name!r}")
                if (member.external_attr >> 16) & 0o170000 == 0o120000:
                    shutil.rmtree(tmpdir, ignore_errors=True)
                    raise ValueError(f"Unsafe zip path (symlink): {name!r}")
            zf.extractall(tmpdir)
        if not os.path.isfile(os.path.join(tmpdir, "bot.py")):
            shutil.rmtree(tmpdir, ignore_errors=True)
            raise ValueError("Zip archive must contain bot.py at the root")
        return tmpdir, tmpdir

    raise ValueError(f"Unsupported bot path: {bot_path!r}")


class FastBot:
    """One in-process bot instance with no resource/time limits."""

    def __init__(self, bot_id: str, bot_path: str):
        self.bot_id = bot_id
        self.bot_path = bot_path
        self.errors: list[str] = []
        self._cleanup_dir: str | None = None
        self._mount_src: str | None = None
        self.module: Any | None = None
        try:
            self._mount_src, self._cleanup_dir = _prepare_bot_mount(bot_path)
            self.module = self._load_module()
        except Exception as exc:
            self.errors.append("load_failed: " + str(exc))

    def _load_module(self):
        assert self._mount_src is not None
        bot_py = os.path.join(self._mount_src, "bot.py")
        module_name = "_fullhouse_fastbot_" + uuid.uuid4().hex
        spec = importlib.util.spec_from_file_location(module_name, bot_py)
        if spec is None or spec.loader is None:
            raise ImportError(f"could not load bot module from {bot_py}")
        module = importlib.util.module_from_spec(spec)

        old_bot_path = os.environ.get("BOT_PATH")
        old_data_dir = os.environ.get("BOT_DATA_DIR")
        old_sys_path = list(sys.path)
        try:
            os.environ["BOT_PATH"] = bot_py
            os.environ["BOT_DATA_DIR"] = os.path.join(self._mount_src, "data")
            if self._mount_src not in sys.path:
                sys.path.insert(0, self._mount_src)
            spec.loader.exec_module(module)
        finally:
            if old_bot_path is None:
                os.environ.pop("BOT_PATH", None)
            else:
                os.environ["BOT_PATH"] = old_bot_path
            if old_data_dir is None:
                os.environ.pop("BOT_DATA_DIR", None)
            else:
                os.environ["BOT_DATA_DIR"] = old_data_dir
            sys.path[:] = old_sys_path

        if not hasattr(module, "decide"):
            raise AttributeError("bot.py must define decide(game_state)")
        return module

    def warmup(self) -> None:
        if self.module is None:
            return
        try:
            self.module.decide({"type": "warmup"})
        except Exception:
            # sandbox.runner also discards warmup exceptions at match level.
            return

    def act(self, state: dict) -> dict:
        if self.module is None:
            return {"action": "fold", "error": "no_process"}
        try:
            action = self.module.decide(state)
            if not isinstance(action, dict) or "action" not in action:
                raise ValueError("decide() must return dict with 'action' key")
            if "error" in action:
                self.errors.append(str(action["error"]))
            return action
        except Exception as exc:
            self.errors.append("exception: " + str(exc))
            return {"action": "fold", "error": "exception"}

    def stop(self) -> None:
        if self._cleanup_dir and os.path.isdir(self._cleanup_dir):
            shutil.rmtree(self._cleanup_dir, ignore_errors=True)


def _inject_match_log(state: dict, match_log: list[dict]) -> dict:
    if state.get("type") == "action_request":
        state["match_action_log"] = match_log[-MATCH_LOG_MAX_ENTRIES:]
    return state


def _play_hand(
    engine: PokerEngine,
    bots: dict[str, FastBot],
    active_bots: list[str],
    match_action_log: list[dict],
    hand_num: int,
    verbose: bool,
) -> dict:
    state = _inject_match_log(engine.start_hand(), match_action_log)
    steps = 0
    while state.get("type") == "action_request":
        seat = int(state["seat_to_act"])
        bot_id = active_bots[seat]
        action = bots[bot_id].act(state)
        if verbose:
            print(f"  [{bot_id}] {action}", file=sys.stderr)
        match_action_log.append({
            "hand_num": hand_num,
            "seat": seat,
            "bot_id": bot_id,
            "action": action.get("action"),
            "amount": action.get("amount"),
        })
        state = _inject_match_log(engine.apply_action(seat, action), match_action_log)
        steps += 1
        if steps > 1000:
            raise RuntimeError(f"Hand exceeded 1000 steps: {engine.hand_id}")
    return state


def run_fast_match(
    match_id: str,
    bot_paths: dict[str, str],
    n_hands: int = 400,
    verbose: bool = False,
    seed: int | None = None,
) -> dict:
    """Run a match in-process with the same engine semantics as sandbox.match."""
    bot_ids = list(bot_paths.keys())
    if not 2 <= len(bot_ids) <= 9:
        raise AssertionError(f"Need 2-9 bots, got {len(bot_ids)}")

    bots = {bot_id: FastBot(bot_id, path) for bot_id, path in bot_paths.items()}
    stacks = {bot_id: STARTING_STACK for bot_id in bot_ids}
    hand_log: list[dict] = []
    match_action_log: list[dict] = []
    dealer = 0
    start_ts = time.time()

    for bot in bots.values():
        bot.warmup()

    try:
        for hand_num in range(n_hands):
            alive = [bot_id for bot_id in bot_ids if stacks[bot_id] > 0]
            if len(alive) < 2:
                break

            hand_id = match_id + "_h" + str(hand_num).zfill(4)
            hand_seed = (seed * 1000003 + hand_num) if seed is not None else None
            engine = PokerEngine(
                hand_id=hand_id,
                bot_ids=alive,
                dealer_seat=dealer % len(alive),
                starting_stacks={bot_id: stacks[bot_id] for bot_id in alive},
                seed=hand_seed,
            )
            result = _play_hand(engine, bots, alive, match_action_log, hand_num, verbose)
            hand_log.append({"hand_num": hand_num, "hand_id": hand_id, **result})
            for bot_id, stack in result["final_stacks"].items():
                stacks[bot_id] = stack
            dealer += 1
    finally:
        for bot in bots.values():
            bot.stop()

    return {
        "match_id": match_id,
        "bot_ids": bot_ids,
        "seed": seed,
        "n_hands": len(hand_log),
        "duration_s": round(time.time() - start_ts, 2),
        "final_stacks": stacks,
        "chip_delta": {bot_id: stacks[bot_id] - STARTING_STACK for bot_id in bot_ids},
        "bot_errors": {bot_id: bots[bot_id].errors for bot_id in bot_ids},
        "hands": hand_log,
    }


def run_fast_match_task(task: dict) -> dict:
    return run_fast_match(
        match_id=task.get("match_id") or "fast_" + uuid.uuid4().hex[:8],
        bot_paths=task["bot_paths"],
        n_hands=int(task.get("n_hands", task.get("hands", 400))),
        verbose=bool(task.get("verbose", False)),
        seed=task.get("seed"),
    )


def run_fast_matches_parallel(
    tasks: list[dict],
    workers: int | None = 1,
    backend: str = "process",
) -> list[dict]:
    return map_parallel(run_fast_match_task, tasks, workers=workers, backend=backend)


def _bot_paths_from_cli(paths: list[str]) -> dict[str, str]:
    bots = {}
    for index, path in enumerate(paths):
        p = Path(path)
        bot_id = p.stem if p.suffix in (".py", ".zip") else (p.name or f"bot_{index}")
        if bot_id in bots or bot_id == "bot":
            bot_id = p.parent.name or f"bot_{index}"
        base = bot_id
        suffix = 2
        while bot_id in bots:
            bot_id = f"{base}_{suffix}"
            suffix += 1
        bots[bot_id] = path
    return bots


def _json_summary(result: dict) -> dict:
    return {
        "match_id": result["match_id"],
        "seed": result["seed"],
        "n_hands": result["n_hands"],
        "duration_s": result["duration_s"],
        "final_stacks": result["final_stacks"],
        "chip_delta": result["chip_delta"],
        "bot_errors": result["bot_errors"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run unlimited in-process Fullhouse training matches")
    parser.add_argument("bots", nargs="+", help="Paths to bot.py files, bot directories, or bot.zip archives")
    parser.add_argument("--hands", type=int, default=400)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--seed-start", type=int, default=1)
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--parallel-backend", choices=["process", "thread"], default="process")
    parser.add_argument("--match-id", default=None)
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    bot_paths = _bot_paths_from_cli(args.bots)
    if args.repeat <= 1:
        result = run_fast_match(
            match_id=args.match_id or "fast_" + uuid.uuid4().hex[:8],
            bot_paths=bot_paths,
            n_hands=args.hands,
            verbose=args.verbose,
            seed=args.seed,
        )
        print(json.dumps(_json_summary(result), indent=2) if args.json else _json_summary(result))
        return

    tasks = []
    for index in range(args.repeat):
        seed = (args.seed_start + index) if args.seed is None else (args.seed + index)
        tasks.append({
            "match_id": (args.match_id or "fast_batch") + "_" + str(index).zfill(4),
            "bot_paths": bot_paths,
            "n_hands": args.hands,
            "seed": seed,
            "verbose": args.verbose,
        })
    results = run_fast_matches_parallel(tasks, workers=args.workers, backend=args.parallel_backend)
    summaries = [_json_summary(result) for result in results]
    print(json.dumps(summaries, indent=2) if args.json else summaries)


if __name__ == "__main__":
    main()
