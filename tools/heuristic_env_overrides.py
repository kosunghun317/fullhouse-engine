"""Helpers for loading and baking heuristic HEURISTIC_* overrides."""

from __future__ import annotations

import json
import os
import re
import shlex
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator


ROOT = Path(__file__).resolve().parents[1]
HEURISTIC_BOT = ROOT / "bots" / "heuristic" / "bot.py"

ENV_CALL_PATTERN = re.compile(
    r"_(?P<kind>float|int)_env\(\"(?P<name>HEURISTIC_[A-Z0-9_]+)\",\s*"
    r"(?P<default>[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?)\)"
)


def heuristic_env_names(source: str | None = None) -> set[str]:
    source = source if source is not None else HEURISTIC_BOT.read_text(encoding="utf-8")
    return {match.group("name") for match in ENV_CALL_PATTERN.finditer(source)}


def _coerce_env(raw: dict) -> dict[str, str]:
    return {
        str(key): str(value)
        for key, value in raw.items()
        if str(key).startswith("HEURISTIC_")
    }


def _load_json_env(path: Path) -> dict[str, str]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a JSON object")
    for key in ("best_env", "env"):
        value = payload.get(key)
        if isinstance(value, dict):
            return _coerce_env(value)
    return _coerce_env(payload)


def _load_shell_env(path: Path) -> dict[str, str]:
    env: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        tokens = shlex.split(line, comments=True, posix=True)
        if not tokens:
            continue
        if tokens[0] == "export":
            tokens = tokens[1:]
        if len(tokens) != 1 or "=" not in tokens[0]:
            continue
        key, value = tokens[0].split("=", 1)
        if key.startswith("HEURISTIC_"):
            env[key] = value
    return env


def load_env_overrides(path: str | Path) -> dict[str, str]:
    env_path = Path(path)
    if not env_path.is_file():
        raise FileNotFoundError(env_path)
    if env_path.suffix.lower() == ".json":
        env = _load_json_env(env_path)
    else:
        env = _load_shell_env(env_path)
    if not env:
        raise ValueError(f"{env_path} did not contain HEURISTIC_* overrides")
    return env


def _format_literal(kind: str, value: str) -> str:
    if kind == "int":
        return str(int(round(float(value))))
    return f"{float(value):.12g}"


def bake_heuristic_source(source: str, overrides: dict[str, str]) -> tuple[str, dict]:
    matched: set[str] = set()

    def replace(match: re.Match) -> str:
        name = match.group("name")
        if name not in overrides:
            return match.group(0)
        matched.add(name)
        kind = match.group("kind")
        return f'_{kind}_env("{name}", {_format_literal(kind, overrides[name])})'

    baked = ENV_CALL_PATTERN.sub(replace, source)
    unknown = sorted(set(overrides) - heuristic_env_names(source))
    if unknown:
        raise ValueError("unknown HEURISTIC_* overrides: " + ", ".join(unknown))
    return baked, {
        "override_count": len(overrides),
        "baked_count": len(matched),
        "baked_keys": sorted(matched),
    }


@contextmanager
def heuristic_env_scope(overrides: dict[str, str]) -> Iterator[None]:
    keys = heuristic_env_names() | set(overrides)
    old = {key: os.environ.get(key) for key in keys}
    try:
        for key in keys:
            if key in overrides:
                os.environ[key] = str(overrides[key])
            else:
                os.environ.pop(key, None)
        yield
    finally:
        for key, value in old.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
